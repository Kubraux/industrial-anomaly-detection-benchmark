"""Sequential GANomaly MVTec benchmark; final epoch only, no test-based selection."""
import argparse, csv, hashlib, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score)
from options import Options
from lib.data import load_data
from lib.model import Ganomaly

CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather',
              'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
BASE = Path(__file__).resolve().parent
def stamp():
    return datetime.now(timezone.utc).isoformat()
def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
def make_options(args, category):
    opt = Options().parser.parse_args([])
    for k,v in dict(dataset='mvtec', dataroot=str(Path(args.data).resolve()),
        abnormal_class=category, name=category, outf=str(Path(args.output).resolve()),
        niter=args.epochs, batchsize=8, workers=0, isize=256, nz=100, ngf=64,
        ndf=64, extralayers=0, manualseed=0, device='gpu', gpu_ids=[0],
        display=False, display_id=0, isTrain=True, load_weights=False, resume='',
        lr=0.0002, beta1=0.5, w_adv=1.0, w_con=50.0, w_enc=1.0).items():
        setattr(opt,k,v)
    return opt
def score(model, loader):
    scores, labels = [], []
    model.netg.eval()
    elapsed = 0.0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(model.device)
            sync()
            start = time.perf_counter()
            _, zi, zo = model.netg(x)
            s = (zi-zo).square().flatten(1).mean(1)
            sync()
            elapsed += time.perf_counter()-start
            scores.extend(s.cpu().tolist())
            labels.extend(y.tolist())
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        raise RuntimeError('Non-finite anomaly scores')
    return values, np.asarray(labels), elapsed
def worker(args):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opt = make_options(args, args.category)
    out = Path(opt.outf)/opt.name
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'options.json', vars(opt))
    loaders = load_data(opt)
    # Audit actual files consumed; retain hashes and paths for this exact run.
    manifest = []
    train_hashes = set()
    for split in ['train','test']:
        for p,label in loaders[split].dataset.samples:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if split == 'train':
                train_hashes.add(h)
            elif h in train_hashes:
                raise RuntimeError('Train/test exact duplicate: '+str(p))
            manifest.append(dict(split=split,path=str(p),label=label,sha256=h))
    save_csv(out/'data_manifest.csv',manifest)
    model = Ganomaly(opt,loaders)
    save_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),platform=platform.platform(),
        generator_parameters=sum(p.numel() for p in model.netg.parameters()),
        discriminator_parameters=sum(p.numel() for p in model.netd.parameters())))
    history=[]
    start_train=time.perf_counter()
    model.total_steps=0
    for epoch in range(args.epochs):
        model.epoch=epoch
        model.netg.train()
        model.netd.train()
        sums={}
        count=0
        sync()
        tick=time.perf_counter()
        for batch,(x,y) in enumerate(loaders['train'],1):
            model.total_steps += len(x)
            model.set_input((x,y))
            model.optimize_params()
            errors=model.get_errors()
            if not all(np.isfinite(v) for v in errors.values()):
                raise RuntimeError('Non-finite training loss')
            for k,v in errors.items():
                sums[k]=sums.get(k,0)+v*len(x)
            count+=len(x)
            if batch==1 or batch%10==0:
                save_json(out/'progress.json',dict(status='training',epoch=epoch+1,
                    epochs=args.epochs,batch=batch,batches=len(loaders['train']),updated_utc=stamp()))
        sync()
        row=dict(epoch=epoch+1,seconds=time.perf_counter()-tick,images_seen=count,
                 **{k:v/count for k,v in sums.items()})
        history.append(row)
        save_csv(out/'epoch_losses.csv',history)
        print(json.dumps(dict(category=args.category,**row)),flush=True)
    train_seconds=time.perf_counter()-start_train
    model.save_weights(args.epochs-1)
    # Threshold fitted on normal TRAIN scores only; no test-based model selection.
    calibration = torch.utils.data.DataLoader(loaders['train'].dataset,
                          batch_size=opt.batchsize,shuffle=False,num_workers=0,drop_last=False)
    train_scores,_,_=score(model,calibration)
    threshold=float(np.quantile(train_scores,0.99))
    scores,labels,test_seconds=score(model,loaders['test'])
    pred=(scores>threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(labels,pred,labels=[0,1]).ravel())
    precision,recall,thresholds=precision_recall_curve(labels,scores)
    f1s=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    best=int(np.argmax(f1s))
    result=dict(category=args.category,epochs=args.epochs,seed=0,image_size=256,
        image_auroc=float(roc_auc_score(labels,scores)),
        image_ap=float(average_precision_score(labels,scores)),
        image_f1_max_test_oracle=float(f1s[best]),test_oracle_threshold=float(thresholds[best]),
        threshold_train_q99=threshold,precision_train_q99=float(precision_score(labels,pred,zero_division=0)),
        recall_train_q99=float(recall_score(labels,pred,zero_division=0)),
        f1_train_q99=float(f1_score(labels,pred,zero_division=0)),
        tn=tn,fp=fp,fn=fn,tp=tp,train_images=len(train_scores),test_images=len(scores),
        train_seconds=train_seconds,inference_ms_per_image=test_seconds/len(scores)*1000,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
    save_csv(out/'test_scores.csv',[dict(path=str(p),label=int(y),score=float(s),
        predicted_train_q99=int(v)) for (p,_),y,s,v in zip(loaders['test'].dataset.samples,labels,scores,pred)])
    save_csv(out/'train_scores.csv',[dict(path=str(p),score=float(s))
        for (p,_),s in zip(loaders['train'].dataset.samples,train_scores)])
    save_json(out/'metrics.json',result)
    save_json(out/'progress.json',dict(status='complete',epoch=args.epochs,updated_utc=stamp()))
    print(json.dumps(result),flush=True)
def controller(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'source_snapshot').mkdir()
    for p in [BASE/'run_mvtec_benchmark.py',BASE/'options.py',BASE/'train.py',*sorted((BASE/'lib').glob('*.py'))]:
        dst=out/'source_snapshot'/p.relative_to(BASE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
    save_json(out/'protocol.json',dict(epochs=args.epochs,categories=CATEGORIES,
        image_size=256,batch_size=8,seed=0,nz=100,lr=0.0002,beta1=0.5,
        weights=dict(adversarial=1,contextual=50,encoder=1),device='CUDA',
        train='category/train/good only; shuffle; drop_last=True',
        preprocessing='Resize(256), CenterCrop(256), RGB normalization mean/std 0.5',
        score='mean squared latent difference (repository implementation)',
        selection='final epoch only; test evaluated once after training',
        threshold='99th percentile of training-normal scores; strict greater-than',
        f1_max='test-label-optimized descriptive oracle metric, NOT deployment threshold',
        pixel_metrics='not applicable: no native spatial anomaly map',
        logging='no extra generator forward in train mode for image logging',
        historical_comparison='GPU run; legacy Bottle image logging may have updated BN buffers',
        inference_timing='GPU synchronized forward+score only, excludes loading and transfer; batch=8',
        started_utc=stamp()))
    results=[]
    for category in CATEGORIES:
        save_json(out/'status.json',dict(status='running',category=category,
            completed=len(results),total=15,pid=os.getpid(),updated_utc=stamp()))
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',
                 '--category',category,'--epochs',str(args.epochs),'--data',str(Path(args.data).resolve()),
                 '--output',str(out)]
        with (out/(category+'.log')).open('w',encoding='utf-8') as log:
            done=subprocess.run(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:
            save_json(out/'status.json',dict(status='failed',category=category,
                completed=len(results),exit_code=done.returncode,updated_utc=stamp()))
            raise RuntimeError('Worker failed: '+category+'; see log')
        results.append(json.loads((out/category/'metrics.json').read_text()))
        save_csv(out/'results.csv',results)
        save_json(out/'summary.json',dict(completed=len(results),total=15,
            macro_image_auroc=float(np.mean([r['image_auroc'] for r in results])),
            macro_image_ap=float(np.mean([r['image_ap'] for r in results])),
            macro_image_f1_max_test_oracle=float(np.mean([r['image_f1_max_test_oracle'] for r in results]))))
    save_json(out/'status.json',dict(status='complete',completed=15,total=15,updated_utc=stamp()))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--category',choices=CATEGORIES)
    args=parser.parse_args()
    if args.epochs<1:
        parser.error('epochs must be positive')
    worker(args) if args.worker else controller(args)

