"""Official-style GANomaly train/test flow with explicit protocol and epoch metrics."""
import argparse,csv,hashlib,json,os,platform,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
from lib.model import Ganomaly
import run_mvtec_benchmark as common

class OfficialFlow(Ganomaly):
    def optimize_params(self):
        super().optimize_params()
        errors=self.get_errors()
        if not all(np.isfinite(v) for v in errors.values()):
            raise RuntimeError('Non-finite training loss')
        for key,val in errors.items():
            self.loss_sum[key]=self.loss_sum.get(key,0.)+val
        self.batch_count+=1
        if self.batch_count==1 or self.batch_count%10==0:
            common.save_json(self.out/'progress.json',dict(status='training',epoch=self.epoch+1,
                epochs=self.opt.niter,batch=self.batch_count,batches=len(self.dataloader['train']),updated_utc=common.stamp()))
    def official_test(self):
        # Intentionally preserve training mode, as in upstream BaseModel.test().
        assert self.netg.training
        self.opt.phase='test'
        self.total_steps=0
        n=len(self.dataloader['test'].dataset)
        scores=torch.zeros(n,dtype=torch.float32,device=self.device)
        labels=torch.zeros(n,dtype=torch.long,device=self.device)
        with torch.no_grad():
            for i,data in enumerate(self.dataloader['test']):
                self.total_steps+=self.opt.batchsize
                self.set_input(data)
                self.fake,latent_i,latent_o=self.netg(self.input)
                error=(latent_i-latent_o).square().mean(dim=1)
                lo=i*self.opt.batchsize
                scores[lo:lo+error.size(0)]=error.reshape(-1)
                labels[lo:lo+error.size(0)]=self.gt.reshape(-1)
        raw=scores.detach().cpu().numpy().copy()
        span=scores.max()-scores.min()
        if not torch.isfinite(scores).all() or span.item()<=0:
            raise RuntimeError('Constant/non-finite scores; upstream min-max would be undefined')
        normalized=((scores-scores.min())/span).cpu().numpy()
        labels=labels.cpu().numpy()
        p,r,t=precision_recall_curve(labels,normalized)
        f1=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
        best=int(np.argmax(f1))
        metrics=dict(image_auroc=float(roc_auc_score(labels,normalized)),
            image_ap=float(average_precision_score(labels,normalized)),
            f1_max_test_oracle=float(f1[best]),threshold_test_oracle=float(t[best]))
        rows=[dict(path=str(path),label=int(y),raw_score=float(s),normalized_score=float(z))
              for (path,_),y,s,z in zip(self.dataloader['test'].dataset.samples,labels,raw,normalized)]
        return metrics,rows

def worker(args):
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    opt=common.make_options(args,args.category)
    out=Path(opt.outf)/opt.name
    out.mkdir(parents=True,exist_ok=False)
    common.save_json(out/'options.json',vars(opt))
    loaders=common.load_data(opt)
    manifest=[];train_hashes=set()
    for split in ['train','test']:
        for path,label in loaders[split].dataset.samples:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if split=='train':train_hashes.add(digest)
            elif digest in train_hashes:raise RuntimeError('Train/test duplicate')
            manifest.append(dict(split=split,path=str(path),label=label,sha256=digest))
    common.save_csv(out/'data_manifest.csv',manifest)
    model=OfficialFlow(opt,loaders)
    model.out=out
    model.total_steps=0
    history=[];best_auc=0.;best_row=None
    common.save_json(out/'environment.json',dict(torch=torch.__version__,python=sys.version,
        gpu=torch.cuda.get_device_name(0),platform=platform.platform()))
    begin=time.perf_counter()
    for epoch in range(args.epochs):
        model.epoch=epoch;model.loss_sum={};model.batch_count=0
        tick=time.perf_counter()
        # Exact existing upstream-style epoch loop, including image logging at frequency=100.
        model.train_one_epoch()
        common.sync()
        train_seconds=time.perf_counter()-tick
        common.save_json(out/'progress.json',dict(status='testing_train_mode',epoch=epoch+1,
            epochs=args.epochs,updated_utc=common.stamp()))
        metrics,scores=model.official_test()
        row=dict(epoch=epoch+1,train_seconds=train_seconds,
            **{k:v/model.batch_count for k,v in model.loss_sum.items()},**metrics)
        history.append(row)
        common.save_csv(out/'epoch_metrics.csv',history)
        (out/'epoch_scores').mkdir(exist_ok=True)
        common.save_csv(out/'epoch_scores'/('epoch_%03d.csv'%(epoch+1)),scores)
        if metrics['image_auroc']>best_auc:
            best_auc=metrics['image_auroc'];best_row=dict(row)
            model.save_weights(epoch)
            common.save_json(out/'best_metrics.json',best_row)
            common.save_csv(out/'best_test_scores.csv',scores)
        common.save_json(out/'progress.json',dict(status='epoch_complete',epoch=epoch+1,
            epochs=args.epochs,best_epoch=best_row['epoch'] if best_row else None,
            best_test_auroc=best_auc,last_test_auroc=metrics['image_auroc'],updated_utc=common.stamp()))
        print(json.dumps(dict(category=args.category,best_test_auroc=best_auc,**row)),flush=True)
    final_dir=out/'final_weights';final_dir.mkdir()
    torch.save(dict(epoch=args.epochs,state_dict=model.netg.state_dict()),final_dir/'netG.pth')
    torch.save(dict(epoch=args.epochs,state_dict=model.netd.state_dict()),final_dir/'netD.pth')
    # Save optimizer/RNG and auxiliary input state so later exact-continuation work is possible.
    torch.save(dict(epoch=args.epochs,optimizer_g=model.optimizer_g.state_dict(),
        optimizer_d=model.optimizer_d.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),total_steps=model.total_steps,
        fixed_input=model.fixed_input.detach().cpu(),options=vars(opt)),
        final_dir/'training_state.pth')
    result=dict(category=args.category,epochs=args.epochs,seed=0,image_size=256,nz=100,
        best_epoch=best_row['epoch'],best_test_image_auroc=best_row['image_auroc'],
        ap_at_best_auroc=best_row['image_ap'],f1_oracle_at_best_auroc=best_row['f1_max_test_oracle'],
        final_test_image_auroc=history[-1]['image_auroc'],final_image_ap=history[-1]['image_ap'],
        train_images=len(loaders['train'].dataset),test_images=len(loaders['test'].dataset),
        total_seconds=time.perf_counter()-begin,selection='maximum test AUROC across epochs',
        test_mode='train; BatchNorm test statistics persist into next epoch')
    common.save_json(out/'metrics.json',result)
    common.save_json(out/'progress.json',dict(status='complete',epoch=args.epochs,**result))
    print(json.dumps(result),flush=True)

def controller(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    for path in [Path(__file__).resolve(),common.BASE/'run_mvtec_benchmark.py',
                 common.BASE/'options.py',*sorted((common.BASE/'lib').glob('*.py'))]:
        dst=out/'source_snapshot'/path.relative_to(common.BASE)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    common.save_json(out/'protocol.json',dict(categories=args.categories,epochs=args.epochs,
        image_size=256,nz=100,seed=0,batch_size=8,lr=0.0002,
        upstream='https://github.com/samet-akcay/ganomaly/blob/master/lib/model.py',
        flow='train_one_epoch; test in train mode; save on strictly improved test AUROC',
        image_logging='upstream get_current_images forward retained every 100 total_steps',
        optimizer='upstream Adam and BCELoss; D reinit retains optimizer state',
        data='MVTec adaptation: train/good only; test all; sorted test paths; no test shuffle; batch8',
        caveat='test labels select checkpoint; test BatchNorm statistics persist in model buffers; not independent held-out evaluation',
        checkpoints='best after test; final after test; optimizer/RNG state saved at final',
        score='latent MSE then per-test-set minmax as upstream',
        differences='modern PyTorch/CUDA; MVTec loader; extra metrics/logging; no claim of exact historical paper reproduction',
        started_utc=common.stamp()))
    results=[]
    for cat in args.categories:
        common.save_json(out/'status.json',dict(status='running',category=cat,completed=len(results),
            total=len(args.categories),pid=os.getpid(),updated_utc=common.stamp()))
        cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker','--category',cat,
             '--epochs',str(args.epochs),'--data',str(Path(args.data).resolve()),'--output',str(out)]
        with (out/(cat+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run(cmd,cwd=common.BASE,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            common.save_json(out/'status.json',dict(status='failed',category=cat,
                completed=len(results),exit_code=proc.returncode))
            raise RuntimeError('Failed category '+cat)
        results.append(json.loads((out/cat/'metrics.json').read_text()))
        common.save_csv(out/'results.csv',results)
        common.save_json(out/'summary.json',dict(completed=len(results),total=len(args.categories),
            macro_best_test_auroc=float(np.mean([r['best_test_image_auroc'] for r in results])),
            macro_final_test_auroc=float(np.mean([r['final_test_image_auroc'] for r in results]))))
    common.save_json(out/'status.json',dict(status='complete',completed=len(results),
        total=len(args.categories),updated_utc=common.stamp()))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--category',choices=common.CATEGORIES)
    parser.add_argument('--categories',nargs='+',choices=common.CATEGORIES,default=['capsule','carpet'])
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.epochs<1:parser.error('Positive epochs required')
    worker(args) if args.worker else controller(args)

