import csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o): Path(p).write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')

def audit_data(data,category):
    base=Path(data)/category
    train=sorted((base/'train'/'good').glob('*.png'))
    test=sorted((base/'test').glob('*/*.png'))
    assert len(train)==209 and len(test)==83
    assert sum(p.parent.name=='good' for p in test)==20
    rows=[];hashsets=[]
    for split,paths in [('train',train),('test',test)]:
        hashes=set()
        for p in paths:
            h=sha(p);hashes.add(h)
            with Image.open(p) as im:im.verify()
            row=dict(split=split,path=str(p.resolve()),label=int(p.parent.name!='good'),sha256=h)
            if split=='test' and p.parent.name!='good':
                mask=base/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
                assert mask.is_file()
                row.update(mask=str(mask.resolve()),mask_sha256=sha(mask))
            rows.append(row)
        hashsets.append(hashes)
    assert not hashsets[0].intersection(hashsets[1])
    assert not set(train).intersection(test)
    return rows

def region_weights(gt):
    weights=np.zeros(gt.shape,np.float64);regions=0
    for i,g in enumerate(gt):
        labels,n=ndimage.label(g,structure=np.ones((3,3),np.uint8))
        sizes=np.bincount(labels.ravel())
        if n:
            lut=np.zeros(n+1);lut[1:]=1/sizes[1:]
            weights[i]=lut[labels];regions+=n
    assert regions>0
    return weights/regions,regions
def integrate(fpr,pro,cap=0.3):
    last=np.searchsorted(fpr,cap,side="right")
    x=fpr[:last];y=pro[:last]
    if x[-1]<cap:
        ycap=np.interp(cap,[fpr[last-1],fpr[last]],[pro[last-1],pro[last]])
        x=np.append(x,cap);y=np.append(y,ycap)
    return float((np.trapezoid(y,x) if hasattr(np, "trapezoid") else np.trapz(y,x))/cap)
def aupro(scores,gt):
    weights,n=region_weights(gt)
    flat=scores.ravel();order=np.argsort(-flat,kind="stable")
    sorted_s=flat[order]
    ends=np.r_[np.flatnonzero(sorted_s[1:]!=sorted_s[:-1]),len(flat)-1]
    background=(~gt).ravel()
    pro=np.r_[0.,np.cumsum(weights.ravel()[order],dtype=np.float64)[ends]]
    fpr=np.r_[0.,np.cumsum(background[order],dtype=np.int64)[ends]/int(background.sum())]
    result=integrate(fpr,pro)
    curve={"fpr":np.linspace(0,.3,1001),"pro":np.interp(np.linspace(0,.3,1001),fpr,pro)}
    return result,n,curve

def evaluate(scores,labels,masks):
    scores=np.asarray(scores);labels=np.asarray(labels);masks=np.asarray(masks,dtype=bool)
    if masks.ndim==4:masks=masks[:,0]
    assert scores.shape==masks.shape and scores.ndim==3
    assert np.isfinite(scores).all() and set(labels.tolist())=={0,1}
    assert not masks[labels==0].any()
    value,regions,curve=aupro(scores,masks)
    return dict(image_auroc=float(roc_auc_score(labels,scores.reshape(len(scores),-1).max(1))),
        pixel_auroc=float(roc_auc_score(masks.ravel(),scores.ravel())),
        aupro_at_030=value,regions=regions,normal_images=int((labels==0).sum()),
        anomaly_images=int((labels==1).sum()),threshold=None,
        image_score='max of bicubic-upsampled sigma4-smoothed raw Mahalanobis map'),curve

def save_evaluation(out,scores,labels,masks,paths):
    metrics,curve=evaluate(scores,labels,masks)
    dump(out/'metrics.json',metrics)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['category','image_auroc','pixel_auroc','aupro_at_030'])
        w.writeheader();w.writerow(dict(category='bottle',**{k:metrics[k] for k in ['image_auroc','pixel_auroc','aupro_at_030']}))
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','label','image_score']);w.writeheader()
        for p,y,s in zip(paths,labels,scores):
            w.writerow(dict(path=p,label=int(y),image_score=float(s.max())))
    save_comparison_panel(out, scores, masks, paths)
    np.savez_compressed(out/'aupro_curve.npz',**curve)


def choose_visual_examples(manifest, seed=1024):
    # Selection uses filenames and defect type only, before any model prediction.
    rng=np.random.default_rng(seed)
    paths=[Path(r['path']) for r in manifest if r['split']=='test']
    groups=['good']+sorted({p.parent.name for p in paths if p.parent.name!='good'})
    chosen=[]
    for group in groups:
        candidates=sorted(str(p) for p in paths if p.parent.name==group)
        n=2 if group=='good' else 1
        chosen.extend(candidates[int(i)] for i in rng.choice(len(candidates),size=n,replace=False))
    return dict(seed=seed,selection='2 normal + 1 per defect type; score-independent',
                paths=chosen)

def save_comparison_panel(out,scores,masks,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=json.loads((out/'visual_selection.json').read_text())
    index={str(Path(p).resolve()):i for i,p in enumerate(paths)}
    if masks.ndim==4:masks=masks[:,0]
    chosen=selected['paths']
    # Shared scale for visualization only; raw metric inputs remain unchanged.
    lo=float(scores.min());hi=float(scores.max())
    if hi<=lo:hi=lo+1.
    fig,axs=plt.subplots(len(chosen),3,figsize=(10,3*len(chosen)),squeeze=False,
                         constrained_layout=True)
    for row,path in enumerate(chosen):
        i=index[str(Path(path).resolve())]
        with Image.open(path) as im:
            rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC).crop((16,16,240,240)))
        assert rgb.shape[:2]==scores[i].shape==masks[i].shape
        axs[row,0].imshow(rgb)
        axs[row,1].imshow(masks[i],cmap='gray',vmin=0,vmax=1)
        axs[row,2].imshow(rgb)
        heat=axs[row,2].imshow(scores[i],cmap='jet',alpha=.5,vmin=lo,vmax=hi)
        name=Path(path).parent.name+'/'+Path(path).name
        for col,title in enumerate(['Input','Ground truth mask','Anomaly heatmap']):
            axs[row,col].axis('off')
            axs[row,col].set_title(name+' | '+title,fontsize=9)
    fig.colorbar(heat,ax=axs[:,2].tolist(),shrink=.6,label='Raw Mahalanobis score (shared scale)')
    fig.suptitle('PaDiM Bottle | fixed random examples | no prediction threshold')
    fig.savefig(out/'bottle_comparison_panel.png',dpi=160)
    plt.close(fig)
    dump(out/'visualization_metadata.json',dict(**selected,shared_vmin=lo,shared_vmax=hi,
        score_scale='all test maps; display only, not calibration or metric normalization',
        threshold=None,heatmap_overlay_alpha=.5,individual_maps_saved=False))
