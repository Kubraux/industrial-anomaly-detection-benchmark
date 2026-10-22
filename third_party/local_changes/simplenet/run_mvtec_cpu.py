"""Official SimpleNet MVTec run on Windows/CPU; smoke test gates full run."""
import csv
import gc
import json
import logging
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import backbones
import simplenet
import utils
from datasets.mvtec import MVTecDataset, DatasetSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / 'archive'
OUT = ROOT / 'results' / 'mvtec_official_cpu'
CLASSES = 'screw pill capsule carpet grid tile wood zipper cable toothbrush transistor metal_nut bottle hazelnut leather'.split()
PARAMS = dict(backbone='wideresnet50', weights='IMAGENET1K_V1', layers=['layer2','layer3'], pretrain_embed_dimension=1536, target_embed_dimension=1536, patchsize=3, embedding_size=256, meta_epochs=40, gan_epochs=4, noise_std=0.015, dsc_hidden=1024, dsc_layers=2, dsc_margin=0.5, pre_proj=1, dsc_lr=0.0002, projection_lr=0.0001, batch_size=8, resize=329, imagesize=288, seed=0, num_workers=0, cpu_threads=2, train_backbone=False, cos_lr=False, mix_noise=1)

class ScalarLog:
    def __init__(self):
        self.g_iter = 0
        self.logger = self
        self.last_loss = None
    def add_scalar(self, name, value, *args):
        if name == 'loss':
            self.last_loss = float(value)
            if not np.isfinite(self.last_loss):
                raise RuntimeError('Non-finite discriminator loss')
    def step(self):
        self.g_iter += 1

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def dataset(name, split):
    return MVTecDataset(str(DATA), classname=name, split=split, resize=329, imagesize=288)

def loader(ds, train=False):
    return DataLoader(ds, batch_size=8, shuffle=train, num_workers=0, pin_memory=False)

def model():
    utils.fix_seeds(0, with_cuda=False)
    net = simplenet.SimpleNet(torch.device('cpu'))
    backbone = backbones.load('wideresnet50')
    backbone.name, backbone.seed = 'wideresnet50', None
    net.load(backbone=backbone, layers_to_extract_from=['layer2','layer3'], device=torch.device('cpu'), input_shape=(3,288,288), pretrain_embed_dimension=1536, target_embed_dimension=1536, patchsize=3, embedding_size=256, meta_epochs=40, gan_epochs=4, noise_std=0.015, dsc_hidden=1024, dsc_layers=2, dsc_margin=0.5, pre_proj=1)
    net.logger = ScalarLog()
    return net

def weights(net):
    return {name:{k:v.detach().cpu().clone() for k,v in getattr(net,name).state_dict().items()} for name in ['discriminator','pre_projection']}

def restore(net, state):
    for name, value in state.items():
        getattr(net,name).load_state_dict(value)

def evaluate(net, test):
    start = time.perf_counter()
    prediction = net.predict(test)
    elapsed = time.perf_counter()-start
    scores = net._evaluate(test, *prediction)
    if not np.isfinite(scores).all():
        raise RuntimeError('Non-finite evaluation metrics')
    return list(map(float,scores)), elapsed / len(test.dataset), prediction

def smoke():
    print('SMOKE: bottle, 8 normal training images, 8 test images, 1 discriminator epoch', flush=True)
    train, test = dataset('bottle',DatasetSplit.TRAIN), dataset('bottle',DatasetSplit.TEST)
    good = [i for i,x in enumerate(test.data_to_iterate) if x[1]=='good'][:4]
    bad = [i for i,x in enumerate(test.data_to_iterate) if x[1]!='good'][:4]
    net = model()
    before = weights(net)
    net.gan_epochs = 1
    start = time.perf_counter()
    net._train_discriminator(loader(Subset(train,list(range(8))),True))
    training_seconds = time.perf_counter()-start
    for name in before:
        assert any(not torch.equal(v,getattr(net,name).state_dict()[k]) for k,v in before[name].items()), name+' did not update'
    scores, latency, _ = evaluate(net,loader(Subset(test,good+bad)))
    result = dict(status='passed', train_images=8, test_images=8, meta_epochs=1, gan_epochs=1, metrics=scores, training_batch_seconds=training_seconds, inference_seconds_per_image=latency, final_loss=net.logger.last_loss, adapter_updated=True, discriminator_updated=True)
    atomic_json(OUT/'smoke.json',result)
    print('SMOKE PASSED '+json.dumps(result),flush=True)
    del net, before
    gc.collect()

def save_examples(name, test, prediction):
    if name not in ['bottle','hazelnut','carpet','grid']:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores, maps, _, labels, masks = prediction
    lo, hi = float(np.min(maps)), float(np.max(maps))
    folder = OUT/'examples'
    folder.mkdir(exist_ok=True)
    for label in [0,1]:
        i = labels.index(label)
        item = test.dataset[i]
        img = item['image'].numpy().transpose(1,2,0)
        img = np.clip(img*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
        fig,axes = plt.subplots(1,3,figsize=(9,3))
        axes[0].imshow(img); axes[0].set_title(name+(' anomaly' if label else ' good'))
        axes[1].imshow(np.squeeze(masks[i]),cmap='gray',vmin=0,vmax=1); axes[1].set_title('Ground truth')
        im=axes[2].imshow(maps[i],cmap='jet',vmin=lo,vmax=hi); axes[2].set_title('Anomaly score')
        for ax in axes: ax.axis('off')
        fig.colorbar(im,ax=axes[2],fraction=.046)
        fig.savefig(folder/(name+('_anomaly.png' if label else '_good.png')),dpi=120,bbox_inches='tight')
        plt.close(fig)

def write_results(rows):
    fields=['class','I-AUROC','P-AUROC','PRO-AUPRO_official','inference_s_per_image','best_epoch','train_images','test_images','elapsed_s']
    path=OUT/'results.csv'; tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        if len(rows)==15:
            mean={'class':'mean'}
            for key in fields[1:5]: mean[key]=float(np.mean([r[key] for r in rows]))
            writer.writerow(mean)
    tmp.replace(path)
    lines=['# SimpleNet MVTec AD CPU experiment', '', f'Completed classes: {len(rows)}/15. Metrics in results.csv are fractions (0-1).', '', 'Parameters: '+json.dumps(PARAMS), '', 'Compatibility: NumPy bool and pandas append APIs updated; CPU best weights cloned; unused dense feature upsampling skipped; masks retained as NumPy arrays; workers=0 with no prefetch/pinned memory. No architecture, batch size, resolution or full-run epoch reductions.', '', 'Selection follows official code: evaluate after every meta epoch; maximize test I-AUROC, break ties with P-AUROC. Thus test data participates in checkpoint selection.', '', 'PRO follows official implementation: 200 thresholds, 5x5 dilation for region overlap, undilated false positives, FPR<0.3 divided by maximum retained FPR. This is the repository metric, not a claim of equivalence to all standardized AUPRO implementations.', '', 'Inference time includes test image loading, transforms, model, heatmap interpolation/smoothing and prediction collection; excludes metric calculation. Batch size 8, CPU, warm model.', '', 'Paper reports 99.6% detection AUROC. Source: https://arxiv.org/abs/2303.15140 . CPU timings cannot be directly compared to paper GPU FPS. Single seed and current dependencies can differ from published averages.', '', '| Class | I-AUROC % | P-AUROC % | PRO % |', '|---|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['class']} | {100*r['I-AUROC']:.3f} | {100*r['P-AUROC']:.3f} | {100*r['PRO-AUPRO_official']:.3f} |")
    if len(rows)==15:
        means=[100*np.mean([r[k] for r in rows]) for k in fields[1:4]]
        lines += [f'| mean | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |','',f'I-AUROC difference from reported 99.6%: {means[0]-99.6:+.3f} percentage points.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def full_run():
    rows=[]
    if (OUT/'results.csv').exists():
        with (OUT/'results.csv').open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['class']!='mean':
                    rows.append({k:(v if k=='class' else float(v)) for k,v in r.items()})
    done={r['class'] for r in rows}
    write_results(rows)
    for name in CLASSES:
        if name in done: continue
        print('FULL CLASS '+name,flush=True)
        train,test=loader(dataset(name,DatasetSplit.TRAIN),True),loader(dataset(name,DatasetSplit.TEST))
        net=model(); best=None; best_weights=None; best_epoch=0; start_epoch=0; previous_elapsed=0
        folder=OUT/'models'/name; folder.mkdir(parents=True,exist_ok=True)
        resume=folder/'resume.pth'
        if resume.exists():
            state=torch.load(resume,map_location='cpu',weights_only=False)
            restore(net,state['weights']); net.dsc_opt.load_state_dict(state['dsc_opt']); net.proj_opt.load_state_dict(state['proj_opt'])
            best,best_weights,best_epoch,start_epoch=state['best'],state['best_weights'],state['best_epoch'],state['epoch']
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
            previous_elapsed=state['elapsed_s']; del state
        started=time.perf_counter()
        for epoch in range(start_epoch,40):
            atomic_json(OUT/'status.json',dict(status='training',class_name=name,epoch=epoch+1,total_epochs=40,completed_classes=len(rows),pid=os.getpid()))
            net._train_discriminator(train)
            scores,latency,prediction=evaluate(net,test)
            del prediction
            if best is None or scores[0]>best[0] or (scores[0]==best[0] and scores[1]>best[1]):
                best,best_weights,best_epoch=scores,weights(net),epoch+1
            elapsed=previous_elapsed+time.perf_counter()-started
            state=dict(weights=weights(net),dsc_opt=net.dsc_opt.state_dict(),proj_opt=net.proj_opt.state_dict(),best=best,best_weights=best_weights,best_epoch=best_epoch,epoch=epoch+1,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),elapsed_s=elapsed)
            temp=resume.with_suffix('.tmp'); torch.save(state,temp); temp.replace(resume); del state
            print(f'{name} epoch {epoch+1}/40 metrics={scores} best={best} elapsed_s={elapsed:.1f}',flush=True)
        restore(net,best_weights)
        measured,latency,prediction=evaluate(net,test)
        assert np.allclose(measured,best,atol=1e-7), (measured,best)
        save_examples(name,test,prediction)
        torch.save(best_weights,folder/'best.pth')
        row=dict(zip(['I-AUROC','P-AUROC','PRO-AUPRO_official'],best))
        row.update({'class':name,'inference_s_per_image':latency,'best_epoch':best_epoch,'train_images':len(train.dataset),'test_images':len(test.dataset),'elapsed_s':previous_elapsed+time.perf_counter()-started})
        rows.append(row); write_results(rows)
        resume.unlink()
        del net,train,test,best_weights,prediction
        gc.collect()
    atomic_json(OUT/'status.json',dict(status='completed',completed_classes=15,pid=os.getpid()))
    print('ALL 15 CLASSES COMPLETED',flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)
    config=dict(parameters=PARAMS,data_path=str(DATA),python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,source='https://github.com/DonaldRR/SimpleNet/blob/main/run.sh')
    atomic_json(OUT/'config.json',config)
    counts={}
    for name in CLASSES:
        tr,te=dataset(name,DatasetSplit.TRAIN),dataset(name,DatasetSplit.TEST)
        for _,anomaly,image,mask in te.data_to_iterate:
            assert Path(image).is_file()
            if anomaly!='good':
                assert Path(mask).is_file() and Path(mask).stem==Path(image).stem+'_mask'
        counts[name]=dict(train=len(tr),test=len(te))
    atomic_json(OUT/'data_counts.json',counts)
    print('DATA VALIDATED '+json.dumps(counts),flush=True)
    try:
        if not (OUT/'smoke.json').exists(): smoke()
        else: assert json.loads((OUT/'smoke.json').read_text())['status']=='passed'
        full_run()
    except BaseException as exc:
        atomic_json(OUT/'status.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise

if __name__=='__main__': main()
