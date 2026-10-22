"""Reproducible train-only checks followed by one final held-out evaluation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
from lib.evaluate import roc, score_report

def probes(model, images):
    """Train data only; temporary modes cannot alter persistent BN statistics."""
    snapshots = [{k: v.clone() for k,v in net.named_buffers()}
                 for net in (model.netg, model.netd)]
    modes = [net.training for net in (model.netg, model.netd)]
    result = {}
    try:
        with torch.no_grad():
            for training in (False, True):
                model.netg.train(training)
                model.netd.train(training)
                fake = model.netg(images)
                pred, real_feat = model.netd(images)
                fake_pred, fake_feat = model.netd(fake)
                result['train_mode' if training else 'eval_mode'] = dict(
                    l1=float((images-fake).abs().mean()),
                    mse=float((images-fake).square().mean()),
                    latent_mse=float((real_feat-fake_feat).square().mean()),
                    real_probability=float(pred.mean()),
                    fake_probability=float(fake_pred.mean()),
                    fake_std=float(fake.std()), input_std=float(images.std()))
    finally:
        with torch.no_grad():
            for net, snapshot, mode in zip((model.netg,model.netd),snapshots,modes):
                for name, value in net.named_buffers():
                    value.copy_(snapshot[name])
                net.train(mode)
    return result

def main():
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--audit_mode',choices=['diagnostic','smoke','checkpoint'],default='diagnostic')
    extra.add_argument('--checkpoint_dir')
    own,rest=extra.parse_known_args()
    sys.argv=[sys.argv[0]]+rest
    opt=Options().parse()
    assert opt.device=='cpu' and opt.dataset=='mvtec' and opt.abnormal_class=='bottle'
    data=load_data(opt)
    print('ENV',sys.version,torch.__version__,'threads',torch.get_num_threads(), flush=True)
    counts={}
    for split,ds in [('train',data.train.dataset),('test',data.valid.dataset)]:
        counts[split]=dict(Counter(y for _,y in ds.samples))
        print(split,'labels',counts[split], 'transform',ds.transform,flush=True)
        seen=set()
        for path,label in ds.samples:
            folder=Path(path).parent.name
            assert label == (0 if folder=='good' else 1)
            if folder not in seen:
                print('EXAMPLE',path,'label',label,flush=True)
                seen.add(folder)
    assert counts=={'train':{0:209},'test':{1:63,0:20}}
    assert repr(data.train.dataset.transform)==repr(data.valid.dataset.transform)
    assert not ({str(Path(p).resolve()) for p,_ in data.train.dataset.samples}
                & {str(Path(p).resolve()) for p,_ in data.valid.dataset.samples})
    # Detect byte-identical copies crossing the split, not just path overlap.
    hashes=lambda ds: {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p,_ in ds.samples}
    assert not hashes(data.train.dataset) & hashes(data.valid.dataset)
    print('No train/test path or file-content overlap.',flush=True)
    assert roc(torch.tensor([0,1]),torch.tensor([.1,.9],requires_grad=True))==1
    assert roc([0,1],[.9,.1])==0
    assert roc([0,1],[.5,.5])==.5
    report,_=score_report([0,1],[.5,.5])
    assert report['normalized_auc']==.5
    model=load_model(opt,data)
    fixed=torch.stack([data.train.dataset[i][0] for i in range(8)])
    assert fixed.shape==(8,3,256,256) and fixed.min()>=-1 and fixed.max()<=1
    print('G/D parameter counts',sum(p.numel() for p in model.netg.parameters()),
          sum(p.numel() for p in model.netd.parameters()),flush=True)
    if own.audit_mode=='checkpoint':
        for name,net in [('G',model.netg),('D',model.netd)]:
            state=torch.load(Path(own.checkpoint_dir)/('net'+name+'_0.pth'),map_location='cpu')
            net.load_state_dict(state['state_dict'])
        result={'existing_checkpoint_train_probes':probes(model,fixed)}
        print(json.dumps(result,indent=2),flush=True)
    else:
        before=probes(model,fixed)
        print('TRAIN-ONLY PROBE BEFORE',json.dumps(before),flush=True)
        if own.audit_mode=='diagnostic':
            iterator=iter(data.train)
            for i in range(2):
                images,labels=next(iterator)
                if i==1: images,labels=images[:3],labels[:3]
                model.total_steps+=images.size(0)
                model.set_input((images,labels))
                model.diagnostic_active=True
                with torch.autograd.detect_anomaly():
                    model.optimize_params()
            bn={k:v.clone() for k,v in model.netg.named_buffers()}
            images=model.get_current_images()
            assert images[2].shape==(8,3,256,256)
            assert all(torch.equal(bn[k],v) for k,v in model.netg.named_buffers())
            assert model.netg.training and model.netd.training
            print('PASS: 2 batches, short batch=3, gradients, updates, graph isolation, BN and visualization.',flush=True)
        else:
            calls=[]
            original=model.test
            def checked_test(*args,**kwargs):
                assert not calls, 'Held-out evaluation called more than once'
                calls.append(1)
                assert (Path(model.trn_dir)/'weights'/('netG_%d.pth'%opt.niter)).exists()
                states=[{k:v.clone() for k,v in n.named_buffers()} for n in (model.netg,model.netd)]
                value=original(*args,**kwargs)
                assert model.netg.training and model.netd.training
                assert all(torch.equal(states[i][k],v)
                           for i,n in enumerate((model.netg,model.netd)) for k,v in n.named_buffers())
                return value
            model.test=checked_test
            model.train()
            assert len(calls)==1
        after=probes(model,fixed)
        result=dict(before=before,after=after,d_reinitializations=getattr(model,'d_reinitializations',0))
        print('TRAIN-ONLY PROBE AFTER',json.dumps(after),flush=True)
    destination=Path(opt.outf)/opt.name
    (destination/'train_probes.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    main()

