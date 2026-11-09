"""Common evaluation of unchanged saved final DFR/RD4AD maps. No training."""
import argparse, csv, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"DFR_RD4AD_common_evaluation"
CATS=["bottle","cable","capsule","hazelnut","metal_nut","pill","screw","toothbrush","transistor","zipper","carpet","grid","leather","tile","wood"]
METRICS=["pixel_auroc","pixel_ap","aupro","precision","recall","fpr","f1","predicted_area_over_gt_area"]
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(x,indent=2,default=str,allow_nan=False),encoding="utf-8");t.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def csvread(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locations(cat):
    d=ROOT/("DFR_results" if cat=="bottle" else "DFR_results_all")/cat
    r=ROOT/"RD4AD_results_bottle"/"final" if cat=="bottle" else ROOT/"RD4AD_results_all"/cat
    return d,r
def keys(p):return "/".join(str(p).replace("\\","/").split("/")[-2:])
def shared_gt(cat,names):
    gt=np.zeros((len(names),256,256),bool)
    masks=[]
    for i,key in enumerate(names):
        kind,filename=key.split("/")
        if kind!="good":
            p=ROOT/"archive"/cat/"ground_truth"/kind/(Path(filename).stem+"_mask.png")
            with Image.open(p) as im:
                gt[i]=np.asarray(im.convert("L").resize((256,256),Image.Resampling.NEAREST))>0
            assert gt[i].any(),p
            masks.append({"path":str(p),"sha256":sha(p)})
    return gt,masks
def load_maps(model,cat):
    d,r=locations(cat)
    if model=="DFR":
        with np.load(d/"test_predictions.npz",allow_pickle=False) as z:
            maps=z["scores"]; names=[keys(p) for p in z["names"]]
        provenance={"file":str(d/"test_predictions.npz"),"sha256":sha(d/"test_predictions.npz")}
    else:
        rec=csvread(r/"per_image.csv")
        names=[keys(x["image"]) for x in rec]
        maps=np.stack([np.load(r/x["raw_map"],allow_pickle=False) for x in rec])
        provenance={"files":[{"image":keys(x["image"]),"path":str(r/x["raw_map"]),"sha256":sha(r/x["raw_map"])} for x in rec]}
    assert len(set(names))==len(names) and maps.shape==(len(names),256,256)
    expected={keys(p) for p in (ROOT/"archive"/cat/"test").glob("*/*.png")}
    assert set(names)==expected
    order=np.argsort(names);maps=maps[order];names=[names[i] for i in order]
    assert np.isfinite(maps).all()
    return maps,names,provenance
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
    return float(np.trapz(y,x)/cap)
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
def counts(gt,pred):
    tp=int(np.count_nonzero(gt&pred));fp=int(np.count_nonzero(~gt&pred))
    fn=int(np.count_nonzero(gt&~pred));tn=int(np.count_nonzero(~gt&~pred))
    div=lambda a,b:a/b if b else 0.
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":div(tp,tp+fp),"recall":div(tp,tp+fn),
        "fpr":div(fp,fp+tn),"f1":div(2*tp,2*tp+fp+fn),
        "predicted_area_over_gt_area":div(tp+fp,tp+fn)}
def threshold(scores):
    t=float(np.quantile(scores,0.995,method="higher"))
    achieved=float(np.count_nonzero(scores>t)/scores.size)
    assert achieved<=.005+1e-12
    return t,achieved
def test_evaluator():
    # Exact AUPRO checked against exhaustive thresholds including tied scores.
    rng=np.random.RandomState(12);g=np.zeros((2,8,8),bool)
    g[0,1:3,1:3]=1;g[0,5:7,5:7]=1;g[1,3:6,2:5]=1
    for a in (g.astype(float),1-g.astype(float),rng.randint(0,7,g.shape).astype(float)):
        w,_=region_weights(g);xx=[0.];yy=[0.]
        for t in np.unique(a)[::-1]:
            pred=a>=t;xx.append(float(pred[~g].mean()));yy.append(float(w[pred].sum()))
        expected=integrate(np.array(xx),np.array(yy))
        actual,_,_=aupro(a,g)
        assert abs(actual-expected)<1e-12,(actual,expected)
    assert abs(aupro(g.astype(float),g)[0]-1)<1e-12
    assert abs(aupro(1-g.astype(float),g)[0])<1e-12
    t,f=threshold(np.arange(10000.))
    assert t==9950 and f==.0049
    c=counts(np.array([1,1,0,0],bool),np.array([1,0,1,0],bool))
    assert c["f1"]==.5 and c["fpr"]==.5 and c["predicted_area_over_gt_area"]==1.
    dump(OUT/"evaluator_tests.json",{"passed":True,"checks":["AUPRO exact versus brute force with ties",
        "perfect AUPRO=1","inverted AUPRO=0","conservative target train FPR","confusion matrix"]})
def source_hashes():
    files=list((ROOT/"RD4AD-main").glob("*.py"))+list((ROOT/"DFR-master/DFR-master/DFR-source").glob("*.py"))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}
def calibrate(model,cat,out,test_maps,names):
    import torch
    from torch.utils.data import DataLoader
    d,r=locations(cat)
    if model=="DFR":
        import run_dfr_experiment as lib
        from types import SimpleNamespace
        lib.seed(0)
        cfg=SimpleNamespace(**read(d/"experiment_config.json"))
        with torch.no_grad(): net=lib.official.AnoSegDFR(cfg)
        checkpoint=d/"checkpoint/autoencoder.pth"
        net.autoencoder.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True)["autoencoder"])
        net.extractor.eval();net.autoencoder.eval()
        train=net.train_data
        train_names=list(train.img_files)
        test_ds=net.test_data
        test_index={keys(p):i for i,p in enumerate(test_ds.img_files)}
        score=lambda x:net.score(x).cpu().numpy()
        state_modules=[net.extractor,net.autoencoder]
    else:
        os.environ["TORCH_HOME"]=str(ROOT/"RD4AD_results_bottle/torch_cache")
        sys.path.insert(0,str(ROOT/"RD4AD-main"))
        import main as lib
        import test as rt
        from dataset import get_data_transforms,MVTecDataset
        from torchvision.datasets import ImageFolder
        from scipy.ndimage import gaussian_filter
        lib.setup_seed(111)
        encoder,bn=lib.wide_resnet50_2(pretrained=True)
        decoder=lib.de_wide_resnet50_2(pretrained=False)
        checkpoint=r/("wres50_"+cat+".pth")
        ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
        bn.load_state_dict(ck["bn"]);decoder.load_state_dict(ck["decoder"])
        for module in (encoder,bn,decoder):module.cuda().eval()
        tr,gttr=get_data_transforms(256,256)
        train=ImageFolder(str(ROOT/"archive"/cat/"train"),transform=tr)
        assert train.classes==["good"]
        train_names=[p for p,_ in train.samples]
        test_ds=MVTecDataset(str(ROOT/"archive"/cat),tr,gttr,"test")
        test_index={keys(p):i for i,p in enumerate(test_ds.img_paths)}
        def score(x):
            ins=encoder(x);outs=decoder(bn(ins))
            pre,_=rt.cal_anomaly_map(ins,outs,256,amap_mode="a")
            return gaussian_filter(pre,sigma=4)
        state_modules=[encoder,bn,decoder]
    def state_hash():
        h=hashlib.sha256()
        for mod in state_modules:
            for k,v in mod.state_dict().items():
                h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=state_hash()
    checkpoint_hash=sha(checkpoint)
    # Verify inference uses the saved official pipeline on normal and anomalous test images.
    samples=[next(i for i,k in enumerate(names) if k.startswith("good/")),
        next(i for i,k in enumerate(names) if not k.startswith("good/"))]
    checks=[]
    with torch.no_grad():
        for i in samples:
            x=test_ds[test_index[names[i]]][0].unsqueeze(0).cuda()
            actual=score(x)
            np.testing.assert_allclose(actual,test_maps[i],rtol=1e-5,atol=1e-6)
            checks.append({"image":names[i],"max_absolute_difference":float(np.max(np.abs(actual-test_maps[i])))})
        loader=DataLoader(train,batch_size=1,shuffle=False,num_workers=0)
        target=out/"train_good_scores.npy"
        storage=None
        for i,item in enumerate(loader):
            x=item if model=="DFR" else item[0]
            amap=score(x.cuda())
            assert amap.shape==(256,256) and np.isfinite(amap).all()
            if storage is None:
                storage=np.lib.format.open_memmap(target,mode="w+",dtype=amap.dtype,shape=(len(train),256,256))
            storage[i]=amap
            if (i+1)%50==0:print(model,cat,"calibration",i+1,"/",len(train),flush=True)
        storage.flush()
    assert before==state_hash(),"Inference changed parameters or buffers"
    assert sha(checkpoint)==checkpoint_hash
    train_keys=[Path(p).name for p in train_names]
    expected=sorted(p.name for p in (ROOT/"archive"/cat/"train/good").glob("*.png"))
    assert sorted(train_keys)==expected and len(set(train_keys))==len(train_keys)
    t,f=threshold(storage)
    dump(out/"calibration.json",{"threshold":t,"target_pixel_fpr":.005,"achieved_train_pixel_fpr":f,
        "images":len(train),"pixels":int(storage.size),"quantile_method":"higher","prediction_rule":"score > threshold",
        "training_image_paths":train_names,"checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_hash,
        "model_state_unchanged":True,"saved_test_map_reproduction_checks":checks,
        "train_map_file":str(target),"score_dtype":str(storage.dtype),"normalization":None,
        "all_train_good_images_used":True,"test_labels_used_to_choose_threshold":False})
    del storage
    return t,f
def worker(model,cat):
    out=OUT/model/cat;out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.json").exists():return
    before=source_hashes()
    a,names,provenance=load_maps(model,cat)
    t,f=calibrate(model,cat,out,a,names)
    gt,masks=shared_gt(cat,names)
    print(model,cat,"common metrics",flush=True)
    auc=float(roc_auc_score(gt.ravel(),a.ravel()))
    ap=float(average_precision_score(gt.ravel(),a.ravel()))
    pro,nregions,curve=aupro(a,gt)
    np.savez_compressed(out/"aupro_curve_display.npz",**curve)
    pred=a>t
    result={"model":model,"category":cat,"pixel_auroc":auc,"pixel_ap":ap,"aupro":pro,
        **counts(gt,pred),"threshold":t,"target_train_pixel_fpr":.005,"achieved_train_pixel_fpr":f,
        "test_images":len(names),"gt_regions":nregions,
        "normal_test_pixel_fpr":float(pred[~gt.any(axis=(1,2))].mean())}
    per=[]
    for i,k in enumerate(names):per.append({"model":model,"category":cat,"image":k,**counts(gt[i],pred[i])})
    csvwrite(out/"per_image.csv",per)
    dump(out/"inputs.json",{"test_maps":provenance,"common_gt_sources":masks,"ordered_test_images":names,
        "gt_sha256":hashlib.sha256(gt.tobytes()).hexdigest()})
    assert before==source_hashes()
    dump(out/"verification.json",{"passed":True,"model_source_unchanged":True,"test_maps_unmodified":True,
        "test_maps_normalized":False,"calibration_model_state_unchanged":True})
    dump(out/"metrics.json",result)
    print("DONE",json.dumps(result),flush=True)
def finish():
    data=[read(OUT/m/c/"metrics.json") for c in CATS for m in ("DFR","RD4AD")]
    for c in CATS:
        a=read(OUT/"DFR"/c/"inputs.json");b=read(OUT/"RD4AD"/c/"inputs.json")
        assert a["ordered_test_images"]==b["ordered_test_images"] and a["gt_sha256"]==b["gt_sha256"]
    csvwrite(OUT/"comparison.csv",data)
    stats=[]
    for m in ("DFR","RD4AD"):
        group=[r for r in data if r["model"]==m]
        for name,fn in (("mean",np.mean),("median",np.median),("std",lambda x:np.std(x,ddof=0))):
            stats.append({"model":m,"statistic":name,**{k:float(fn([r[k] for r in group])) for k in METRICS}})
    csvwrite(OUT/"statistics.csv",stats)
    delta=[]
    for c in CATS:
        a=next(r for r in data if r["model"]=="DFR" and r["category"]==c)
        b=next(r for r in data if r["model"]=="RD4AD" and r["category"]==c)
        delta.append({"category":c,**{k:b[k]-a[k] for k in METRICS}})
    csvwrite(OUT/"rd4ad_minus_dfr.csv",delta)
    lines=["# DFR vs RD4AD - common evaluator","",
        "Saved official final test maps only. No score normalization, model changes or retraining.",
        "All train/good images re-scored in eval/no_grad mode with each model's official map pipeline.",
        "Threshold per model/category: higher empirical 99.5% quantile, strict score > threshold.",
        "Identical GT: original binary masks resized to 256x256 with nearest-neighbor, no hole filling.",
        "AUPRO: 8-connected components, equal region weight, all test background pixels including normal images.",
        "Exact unique-score sweep with ties grouped; trapezoidal area at FPR [0,0.30], interpolated endpoint, divided by 0.30.",
        "Display curves are sampled; metric uses exact curve.",
        "Precision/recall/FPR/F1/area are micro-aggregated over category test pixels.",
        "Means below are unweighted across 15 categories. Std ddof=0.","",
        "| Model | Pixel AUROC | Pixel AP | AUPRO | Precision | Recall | FPR | F1 | Pred/GT area |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in stats:
        if r["statistic"]=="mean":lines.append("| "+r["model"]+" | "+" | ".join(f"{r[k]:.6f}" for k in METRICS)+" |")
    lines+=["","Sources: original DFR baseline (not nearest/native128 experiments); final RD4AD epoch 200.",
        "Test FPR need not equal train target 0.005 due to train/test distribution shift.",
        "See comparison.csv, statistics.csv, rd4ad_minus_dfr.csv and each calibration.json/per_image.csv."]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    dump(OUT/"status.json",{"status":"completed","categories":15,"model_category_evaluations":30})
def main():
    p=argparse.ArgumentParser();p.add_argument("--model",choices=["DFR","RD4AD"]);p.add_argument("--category",choices=CATS);p.add_argument("--test-only",action="store_true")
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.test_only:test_evaluator();print("Evaluator tests passed");return
    if args.model:
        try:worker(args.model,args.category)
        except BaseException as exc:
            dump(OUT/args.model/args.category/"failure.json",{"error":repr(exc),"traceback":traceback.format_exc()});raise
        return
    test_evaluator()
    dump(OUT/"protocol.json",{"target_train_pixel_fpr":.005,"train_sampling":"all images, all final-map pixels, batch1",
        "threshold":"empirical 99.5th percentile method=higher, strict >","score_normalization":False,
        "common_gt":"nearest 256x256, original mask >0, no fill","aupro_fpr_limit":.3,
        "aupro":"exact score sweep, ties grouped, 8-connectivity, equal region weight; all normal pixels in FPR",
        "threshold_metrics":"micro within category; mean/median/std across categories; ddof=0",
        "model_source_initial_hashes":source_hashes()})
    for cat in CATS:
        for model in ("DFR","RD4AD"):
            dump(OUT/"status.json",{"status":"running","category":cat,"model":model})
            log=OUT/model/cat/"run.log";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a",encoding="utf-8") as f:
                result=subprocess.run([sys.executable,"-B","-u",__file__,"--model",model,"--category",cat],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            if result.returncode:
                dump(OUT/"status.json",{"status":"failed","category":cat,"model":model,"log":str(log)})
                raise RuntimeError(f"{model} {cat} failed; see {log}")
            print("COMPLETE",model,cat,flush=True)
    finish()
if __name__=="__main__":main()

