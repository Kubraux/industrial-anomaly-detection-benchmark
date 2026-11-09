"""Read-only RD++ fixed-final inference; explicitly separate dry validation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import argparse, csv, hashlib, json, sys, time, traceback, importlib.util
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
BASE=Path(__file__).resolve().parent
BENCH=BASE.parents[1]
ROOT=BENCH.parent.parent
sys.path.insert(0,str(BASE/'source'))
from main import EPOCHS, TEACHER_SHA, seed_all, state_hash, Monitor
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_train import MultiProjectionLayer
from utils.utils_test import cal_anomaly_map, compute_pro
from dataset.dataset import get_data_transforms
spec=importlib.util.spec_from_file_location('common_metrics',ROOT/'compare_dfr_rd4ad_common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
COMMIT='7f2ceb7c87e602617b8600e1a498f7ef7f5247d6'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=str),encoding='utf-8')
def require(ok,message):
    if not ok:raise ValueError(message)
def checkpoint_gate(c,cat,mode):
    require(c['category']==cat,'checkpoint category mismatch')
    require(c['source_commit']==COMMIT and c['teacher_sha256']==TEACHER_SHA,'checkpoint lineage mismatch')
    require(c['epoch_schedule']==EPOCHS and c['seed']==111,'checkpoint schedule/seed mismatch')
    if mode=='benchmark':
        require(c['mode']=='train' and c['selection']=='fixed_final','benchmark requires fixed-final trained checkpoint')
        require(c['epoch']==EPOCHS[cat],'checkpoint is not at prescribed final epoch')
    else:
        require(c['mode']=='smoke' and c['selection']=='bounded_smoke_not_benchmark','dry mode requires labeled smoke checkpoint')
def preprocess(path):
    raw=cv2.imread(str(path));require(raw is not None,'unreadable image '+str(path))
    tr,_=get_data_transforms(256,256)
    return tr(cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)/255.,(256,256)))
def masks_for(image,mask):
    with Image.open(image) as im:size=im.size
    if mask is None:return np.zeros((256,256),bool),np.zeros((256,256),bool),size
    _,gt=get_data_transforms(256,256)
    with Image.open(mask) as m:
        require(m.size==size,'native image-mask size mismatch')
        a=np.asarray(m);require(a.ndim==2 and set(np.unique(a)).issubset({0,255}) and a.max()==255,'invalid binary mask/polarity')
        original=(gt(m).numpy()[0]>.5)
        nearest=np.asarray(m.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
    require(original.any() and nearest.any(),'defect disappeared on resize')
    return original,nearest,size

def inventory(root,cat,mode):
    folder=root/cat/('train' if mode=='dry' else 'test')
    paths=sorted((folder/'good').glob('*.png'))[:2] if mode=='dry' else sorted(folder.glob('*/*.png'))
    require(len(paths)>0,'no input images')
    pairs=[];seen_masks=set();hashes=set()
    for p in paths:
        mask=None if p.parent.name=='good' else root/cat/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
        require(mask is None or mask.is_file(),'missing exact-stem mask '+str(mask))
        if mask:seen_masks.add(mask.resolve())
        h=sha(p);hashes.add(h)
        pairs.append((p,mask,h))
    if mode=='benchmark':
        expected={p.resolve() for p in (root/cat/'ground_truth').glob('*/*.png')}
        require(expected==seen_masks,'orphan/duplicate mask or incomplete pairing')
        train_hashes={sha(p) for p in (root/cat/'train'/'good').glob('*.png')}
        require(bool(train_hashes) and not hashes.intersection(train_hashes),'missing training split or train/test content overlap')
    return pairs

def metrics(labels,scores,maps,masks):
    labels=np.asarray(labels);maps=np.asarray(maps);masks=np.asarray(masks,dtype=bool)
    require(maps.shape==masks.shape and np.isfinite(maps).all() and np.isfinite(scores).all(),'invalid metric arrays')
    def binary(y,s):
        if len(np.unique(y))<2:return {'status':'not_applicable','auroc':None,'ap':None,'reason':'only one label class; validation only'}
        return {'status':'completed','auroc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s))}
    region={'status':'not_applicable','aupro_0_30':None,'reason':'requires anomaly regions and background'}
    if masks.any() and (~masks).any():
        value,n,_=common.aupro(maps,masks);region={'status':'completed','aupro_0_30':value,'regions':n}
    return {'image':binary(labels,scores),'pixel':binary(masks.ravel(),maps.ravel()),'region':region}

def summarize(values):return {'n':len(values),'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))}

def run(args):
    out=args.output.resolve();require(out.is_relative_to((BENCH/'experiments').resolve()),'output must be under benchmark experiments')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    dump(out/'config.yaml',vars(args))
    dump(out/'status.json',{'status':'running','mode':args.mode})
    monitor=Monitor();started=time.perf_counter();monitor.start()
    try:
        seed_all();require(torch.cuda.is_available(),'CUDA required for measured inference')
        from importlib.metadata import distributions
        (out/'environment.txt').write_text(sys.version+'\n'+'\n'.join(sorted(str(d.metadata['Name'])+'=='+d.version for d in distributions())),encoding='utf-8')
        teacher=BASE/'torch_cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth'
        require(sha(teacher)==TEACHER_SHA,'teacher hash mismatch')
        torch.hub.set_dir(str(teacher.parent.parent))
        results=[];resources={}
        for cat in args.classes:
            co=out/cat;co.mkdir()
            checkpoint=args.checkpoint_root/cat/('final.pt' if args.mode=='benchmark' else 'smoke_final.pt')
            before=sha(checkpoint)
            # Only locally trusted checkpoints; RNG metadata requires weights_only=False.
            c=torch.load(checkpoint,map_location='cpu',weights_only=False)
            checkpoint_gate(c,cat,args.mode)
            encoder,bn=wide_resnet50_2(pretrained=True)
            proj=MultiProjectionLayer(base=64);decoder=de_wide_resnet50_2(pretrained=False)
            modules={'encoder':encoder,'proj':proj,'bn':bn,'decoder':decoder}
            for k in ('proj','bn','decoder'):modules[k].load_state_dict(c[k],strict=True)
            del c
            for mod in modules.values():mod.cuda().eval()
            states={k:state_hash(m) for k,m in modules.items()}
            pairs=inventory(args.data_root,cat,args.mode)
            def forward(x):
                f=encoder(x);return f,decoder(bn(proj(f)))
            with torch.no_grad():
                x=preprocess(pairs[0][0]).unsqueeze(0).cuda()
                for _ in range(3):
                    f,y=forward(x);a,_=cal_anomaly_map(f,y,256,amap_mode='a');gaussian_filter(a,sigma=4)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                maps=[];original=[];nearest=[];labels=[];scores=[];rows=[];model_ms=[];pipeline_ms=[];repo_pro=[]
                for i,(p,mask,input_hash) in enumerate(pairs):
                    torch.cuda.synchronize();t=time.perf_counter()
                    x=preprocess(p).unsqueeze(0).cuda();torch.cuda.synchronize();tmodel=time.perf_counter()
                    f,y=forward(x);torch.cuda.synchronize();ms=(time.perf_counter()-tmodel)*1000
                    pre,levels=cal_anomaly_map(f,y,256,amap_mode='a');amap=gaussian_filter(pre,sigma=4)
                    require(amap.shape==(256,256) and np.isfinite(amap).all(),'invalid map')
                    score=float(amap.max());np.savez_compressed(co/f'{i:04d}_prediction.npz',anomaly_map=amap,image_score=score,pre_smoothing=pre,level_maps=np.stack(levels))
                    torch.cuda.synchronize();pipeline=(time.perf_counter()-t)*1000
                    # Ground truth and metric work are excluded from prediction latency.
                    gt,ngt,size=masks_for(p,mask)
                    np.savez_compressed(co/f'{i:04d}_masks.npz',original=gt,common_nearest=ngt)
                    label=int(mask is not None)
                    rows.append({'id':p.relative_to(args.data_root).as_posix(),'image_path':str(p),'image_sha256':input_hash,'mask_path':str(mask) if mask else None,'mask_sha256':sha(mask) if mask else None,'native_size_wh':size,'image_label':label,'image_score':score,'prediction':f'{i:04d}_prediction.npz','prediction_sha256':sha(co/f'{i:04d}_prediction.npz'),'masks':f'{i:04d}_masks.npz','masks_sha256':sha(co/f'{i:04d}_masks.npz'),'model_ms':ms,'pipeline_ms':pipeline})
                    maps.append(amap);original.append(gt);nearest.append(ngt);labels.append(label);scores.append(score);model_ms.append(ms);pipeline_ms.append(pipeline)
                    if label:
                        if amap.max()==amap.min():repo_pro.append(None)
                        else:
                            val=float(compute_pro(gt[None].astype(int),amap[None]));repo_pro.append(val if np.isfinite(val) else None)
            metric={'dataset':'MVTec_AD','method':'RD++','class':cat,'track':'T1','seed':111,'mode':args.mode,'benchmark_result':args.mode=='benchmark','positive_images':sum(labels),'negative_images':len(labels)-sum(labels),'mask_count':sum(labels),'common256':metrics(labels,scores,maps,nearest),'original_mask':metrics(labels,scores,maps,original),'repo_pro_200':{'value':float(np.mean(repo_pro)) if repo_pro and all(v is not None for v in repo_pro) else None,'status':'completed' if repo_pro and all(v is not None for v in repo_pro) else 'not_applicable','definition':'mean per-anomaly official 200-threshold PRO; normalized to largest sampled FPR below0.3; distinct from exact common PRO'},'definitions':{'auroc':'sklearn roc_auc_score','ap':'sklearn average_precision_score noninterpolated','aupro':'common evaluator exact tied thresholds, 8-connectivity, equal-region weighting, pooled backgrounds, interpolated normalized FPR0.30','image_score':'max smoothed map','masks':'original PIL bilinear >.5 and separate common PIL nearest >0'},'checkpoint_sha256':before}
            require(sha(checkpoint)==before and sha(teacher)==TEACHER_SHA,'input checkpoint mutated')
            require(all(state_hash(m)==states[k] for k,m in modules.items()),'model state mutated in inference')
            require(all(sha(p)==h for p,_,h in pairs),'input images mutated')
            resource={'model_only':summarize(model_ms),'pipeline_decode_to_export':summarize(pipeline_ms),'scope':'batch1 FP32;3 warmups; synchronized forward excludes transfer/map, pipeline includes image decode through prediction NPZ export; excludes labels/metrics/hash','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_bytes':checkpoint.stat().st_size,'teacher_bytes':teacher.stat().st_size,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32}
            dump(co/'per_image.json',rows);dump(co/'metrics.json',metric);dump(co/'resource_usage.json',resource)
            source_files=[Path(__file__),ROOT/'compare_dfr_rd4ad_common.py']+list((BASE/'source').rglob('*.py'))
            dump(co/'provenance.json',{'commit':COMMIT,'checkpoint':str(checkpoint),'checkpoint_sha256_before':before,'checkpoint_sha256_after':sha(checkpoint),'teacher_sha256':TEACHER_SHA,'state_unchanged':True,'sources':{str(p):sha(p) for p in source_files},'test_images_read':0 if args.mode=='dry' else len(rows),'manifest_sha256':sha(co/'per_image.json')})
            results.append(metric);resources[cat]=resource
            del encoder,bn,proj,decoder,modules,x,f,y
            torch.cuda.empty_cache()
        macro={}
        for kind in ('image','pixel','region'):
            for metric in (('aupro_0_30',) if kind=='region' else ('auroc','ap')):
                vals=[r['common256'][kind][metric] for r in results]
                macro[kind+'_'+metric]=float(np.mean(vals)) if args.mode=='benchmark' and all(v is not None for v in vals) else None
        dump(out/'metrics.json',{'status':'completed','benchmark_result':args.mode=='benchmark','categories':results,'equal_category_macro_common256':macro,'coverage':len(results),'expected_mvtec_categories':15,'full_dataset_macro':len(results)==15 and args.mode=='benchmark'})
        dump(out/'resource_usage.json',{'categories':resources,'wall_seconds':time.perf_counter()-started,'sampled_peak_ram_bytes':monitor.peak_ram})
        dump(out/'status.json',{'status':'completed','mode':args.mode,'training_started':False})
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');dump(out/'status.json',{'status':'failed','mode':args.mode});raise
    finally:monitor.finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dry','benchmark'],default='dry')
    p.add_argument('--classes',nargs='+',choices=list(EPOCHS),default=['carpet'])
    p.add_argument('--checkpoint-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'archive');a=p.parse_args()
    require(len(a.classes)==len(set(a.classes)),'duplicate categories')
    run(a)
if __name__=='__main__':main()
