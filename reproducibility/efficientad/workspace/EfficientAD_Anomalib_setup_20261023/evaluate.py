from experiment import *
import numpy as np

def exact_metrics(scores,gt,weights,regions,chunk=1000000):
 # Equal-score ties form one threshold; global background FPR and equal 8-connected region weights.
 order=np.argsort(scores,kind='stable')[::-1];neg=int((~gt).sum());pos=int(gt.sum());assert neg and pos and regions
 count_fp=0;count_tp=0;count_pro=0.;lastx=0.;lasty=0.;lastp=0.;roc=0.;pro=0.
 for a in range(0,len(order),chunk):
  b=min(a+chunk,len(order));ids=order[a:b];s=scores[ids];g=gt[ids]
  fp=np.cumsum(~g,dtype=np.int64)+count_fp;tp=np.cumsum(g,dtype=np.int64)+count_tp;pr=np.cumsum(weights[ids],dtype=np.float64)+count_pro
  end=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
  if b<len(order) and s[-1]==scores[order[b]]:end=end[:-1]
  if len(end):
   x=np.r_[lastx,fp[end]/neg];y=np.r_[lasty,tp[end]/pos];p=np.r_[lastp,pr[end]/regions]
   roc+=float(np.trapz(y,x))
   j=np.flatnonzero((x[:-1]<.3)&(x[1:]>x[:-1]));stop=np.minimum(.3,x[j+1]);q=p[j]+(p[j+1]-p[j])*(stop-x[j])/(x[j+1]-x[j]);pro+=float(np.sum((stop-x[j])*(p[j]+q)/2))
   lastx,lasty,lastp=x[-1],y[-1],p[-1]
  count_fp=int(fp[-1]);count_tp=int(tp[-1]);count_pro=float(pr[-1])
 return roc,pro/.3

def metric_smoke():
 from scipy import ndimage
 sys.path.insert(0,str(ROOT/'DRAEM_setup'));from metrics import pro_reference
 from sklearn.metrics import roc_auc_score
 g=np.zeros((2,9,10),bool);g[0,1:3,1:3]=1;g[0,5:8,6:9]=1;g[1,1,1]=g[1,2,2]=1
 w=np.zeros(g.shape);n=0
 for i,m in enumerate(g):
  lab,k=ndimage.label(m,np.ones((3,3)));counts=np.bincount(lab.ravel());lut=np.r_[0.,1/counts[1:]];w[i]=lut[lab];n+=k
 for s in [np.random.default_rng(3).random(g.shape),g.astype(float),np.ones(g.shape),1-g.astype(float)]:
  a,p=exact_metrics(s.ravel(),g.ravel(),w.ravel(),n,chunk=7)
  assert np.isclose(a,roc_auc_score(g.ravel(),s.ravel()),atol=1e-12)
  assert np.isclose(p,pro_reference(s,g),atol=1e-12),(p,pro_reference(s,g))
 dump(P/'metric_smoke.json',dict(passed=True,ties=True,chunk_boundaries=True,connectivity=8,fpr_limit=.3))

def evaluate(size,cat):
 torch,np=runtime()
 from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel,EfficientAdModelSize
 from torchvision.transforms.v2 import Compose,ToTensor,Resize
 from PIL import Image
 from scipy import ndimage
 from sklearn.metrics import roc_auc_score
 import torch.nn.functional as F
 tr,res=paths(size);out=res/cat;out.mkdir(parents=True,exist_ok=True);status(out/'status.json','evaluating')
 ck=torch.load(tr/cat/'final.pt',map_location='cuda',weights_only=False);assert ck['step']==70000
 model=EfficientAdModel(model_size=EfficientAdModelSize.S if size=='s' else EfficientAdModelSize.M,padding=False,pad_maps=True).cuda().eval();model.load_state_dict(ck['model']);del ck
 tf=Compose([ToTensor(),Resize((256,256),antialias=True)])
 split=json.loads((P/'splits'/f'{cat}.json').read_text());warm=DATA/split['fit'][0]['path']
 with Image.open(warm) as im:x=tf(im.convert('RGB')).unsqueeze(0).cuda()
 with torch.inference_mode():
  for _ in range(30):model(x)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 files=sorted((DATA/cat/'test').glob('*/*.png'));lengths=[];records=[]
 for f in files:
  with Image.open(f) as im:lengths.append(im.width*im.height)
 total=sum(lengths)
 scores=np.memmap(out/'scores.f32',mode='w+',dtype=np.float32,shape=(total,));gt=np.memmap(out/'masks.bool',mode='w+',dtype=bool,shape=(total,));weights=np.memmap(out/'region_weights.f64',mode='w+',dtype=np.float64,shape=(total,))
 labels=[];image_scores=[];timings=[];offset=0;regions=0;gallery=False
 with torch.inference_mode():
  for f,n in zip(files,lengths):
   t=time.perf_counter()
   with Image.open(f) as im:rgb=im.convert('RGB');orig=np.array(rgb);x=tf(rgb).unsqueeze(0).cuda()
   torch.cuda.synchronize();pre=time.perf_counter()-t
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);t=time.perf_counter();a.record();prediction=model(x);b.record();torch.cuda.synchronize();forward=time.perf_counter()-t;gpu=a.elapsed_time(b)/1000
   t=time.perf_counter();native=F.interpolate(prediction.anomaly_map,size=orig.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy();score=float(prediction.pred_score.item());torch.cuda.synchronize();post=time.perf_counter()-t
   assert np.isfinite(native).all() and np.isfinite(score)
   label=int(f.parent.name!='good');mask_path=None
   if label:
    mask_path=DATA/cat/'ground_truth'/f.parent.name/(f.stem+'_mask.png')
    with Image.open(mask_path) as mi:mask=np.array(mi.convert('L'))>0
    assert mask.shape==native.shape and mask.any()
   else:mask=np.zeros(native.shape,bool)
   lab,k=ndimage.label(mask,np.ones((3,3)));counts=np.bincount(lab.ravel());lut=np.r_[0.,1/counts[1:]];regions+=k
   scores[offset:offset+n]=native.ravel();gt[offset:offset+n]=mask.ravel();weights[offset:offset+n]=lut[lab].ravel();offset+=n
   labels.append(label);image_scores.append(score);timings.append([pre,gpu,forward,post])
   records.append(dict(path=f.relative_to(DATA).as_posix(),sha256=sha(f),mask=str(mask_path) if mask_path else None,mask_sha256=sha(mask_path) if mask_path else None,shape=list(native.shape),score=score))
   if label and not gallery:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,title in zip(axes,['Input','Ground Truth','Anomaly Map','Overlay']):ax.set_title(title);ax.axis('off')
    axes[0].imshow(orig);axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[2].imshow(native,cmap='jet');axes[3].imshow(orig);axes[3].imshow(native,cmap='jet',alpha=.45)
    fig.savefig(out/'comparison.png',dpi=140);plt.close(fig);gallery=True
 scores.flush();gt.flush();weights.flush();status(out/'status.json','metrics',test_images=len(files),pixels=total)
 pixel,pro=exact_metrics(scores,gt,weights,regions)
 train=json.loads((tr/cat/'training.json').read_text());tm=np.array(timings)*1000
 result=dict(model=size,category=cat,image_auroc=float(roc_auc_score(labels,image_scores)),pixel_auroc=pixel,aupro_030=pro,inference_gpu_ms=float(tm[:,1].mean()),inference_wall_ms=float(tm[:,2].mean()),fps_model_gpu=float(1000/tm[:,1].mean()),preprocess_ms=float(tm[:,0].mean()),postprocess_ms=float(tm[:,3].mean()),fps_pipeline=float(1000/(tm[:,0]+tm[:,2]+tm[:,3]).mean()),inference_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,inference_peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,training_seconds=train['training_seconds'],training_peak_allocated_mb=train['training_peak_allocated_mb'],test_images=len(files),pixels=total,final_checkpoint=train['final_checkpoint'])
 dump(out/'metrics.json',result);dump(out/'test_manifest.json',records);dump(out/'timings.json',dict(columns=['preprocess_ms','model_gpu_ms','model_wall_ms','postprocess_ms'],rows=tm.tolist(),warmup=30,batch=1))
 # Temporary exact-metric buffers are only removed from this newly created category folder.
 del scores,gt,weights
 for name in ['scores.f32','masks.bool','region_weights.f64']:(out/name).unlink()
 status(out/'status.json','complete');report()

def report():
 REPORT.mkdir(parents=True,exist_ok=True);lines=['# EfficientAD-S / EfficientAD-M — Anomalib','', 'Results are generated only after fixed-final training and held-out normal calibration. Missing values are pending, not estimates. Metrics are fractions (0–1).','']
 means={}
 for size in ['s','m']:
  tr,res=paths(size);res.mkdir(parents=True,exist_ok=True);rows=[]
  for cat in CATS:
   p=res/cat/'metrics.json'
   rows.append(json.loads(p.read_text()) if p.exists() else dict(model=size,category=cat,status='pending'))
  complete=[r for r in rows if 'image_auroc' in r];keys=['image_auroc','pixel_auroc','aupro_030','inference_gpu_ms','fps_model_gpu','preprocess_ms','postprocess_ms','fps_pipeline','training_seconds','training_peak_allocated_mb','inference_peak_allocated_mb']
  if len(complete)==15:
   avg=dict(model=size,category='MEAN_15',**{k:float(np.mean([r[k] for r in complete])) for k in keys});rows.append(avg);means[size]=avg
  fields=list(dict.fromkeys(k for r in rows for k in r))
  with (res/'results.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
  lines += [f'## EfficientAD-{size.upper()} ({len(complete)}/15 evaluated)','', '| Category | Image AUROC | Pixel AUROC | AUPRO@.30 | GPU ms | FPS | Train seconds | Train MB | Infer MB |','|---|---|---|---|---|---|---|---|---|']
  for r in rows:lines.append('| '+r['category']+' | '+' | '.join(f'{r[k]:.6f}' if k in r else 'pending' for k in ['image_auroc','pixel_auroc','aupro_030','inference_gpu_ms','fps_model_gpu','training_seconds','training_peak_allocated_mb','inference_peak_allocated_mb'])+' |')
  lines.append('')
 if len(means)==2:
  lines+=['## M minus S','']+[f'- {k}: {means["m"][k]-means["s"][k]:+.6f}' for k in keys]
  lines += ['', f'M/S model latency ratio: {means["m"]["inference_gpu_ms"]/means["s"]["inference_gpu_ms"]:.3f}; M/S inference allocated memory ratio: {means["m"]["inference_peak_allocated_mb"]/means["s"]["inference_peak_allocated_mb"]:.3f}. Positive AUROC/AUPRO deltas favor M; ratios above 1 mean higher cost.', '']
 for size,avg in means.items():
  lines += [f'Paper difference for {size.upper()}: image AUROC {100*avg["image_auroc"]-({"s":98.8,"m":99.1}[size]):+.3f} percentage points; AUPRO@.30 {100*avg["aupro_030"]-({"s":93.1,"m":93.5}[size]):+.3f} points. Protocol differences preclude causal attribution.', '']
 lines+=['## Protocol and comparison limits','','See [PROTOCOL.md](../../EfficientAD_Anomalib_setup_20261023/PROTOCOL.md) and pinned package/config/split manifests. Single seed and Imagenette auxiliary data differ from paper; Anomalib default unpadded PDN + map padding retained; paper Appendix E also reports results without PDN padding. Native-resolution masks/maps are used; previous resized-grid results are not strictly equivalent. Image score is the maximum at model output resolution. No test-dependent threshold or model selection.','', 'Timing: 30 fit-image warmups, batch 1, CUDA synchronization; preprocessing includes file decode/resize/host-to-device; postprocessing includes native resize/device-to-host. Model GPU events and synchronized wall latency are recorded separately. FPS is reciprocal mean latency (not a throughput benchmark). GPU memory is PyTorch peak allocated/reserved, not total device usage.','']
 paper=P/'paper_comparison.md'
 if paper.exists():lines.append(paper.read_text(encoding='utf-8'))
 for folder in ['SimpleNet_MVTec_results_20261020','SuperSimpleNet_MVTec_results_20261020','DRAEM_MVTec_all15_20261015','PatchSVDD_MVTec_all15_20261012']:
  f=ROOT/folder/'results.csv'
  if f.exists():
   prior=list(csv.DictReader(f.open(encoding='utf-8-sig')));prior=[r for r in prior if r.get('category') in CATS]
   if len(prior)==15:
    for size,avg in means.items():
     gaps=[]
     for key in ['image_auroc','pixel_auroc','aupro_030']:
      if all(r.get(key) for r in prior):gaps.append(f'{key}: {100*(avg[key]-np.mean([float(r[key]) for r in prior])):+.3f} percentage points')
     lines += [f'{size.upper()} minus {folder}: '+', '.join(gaps)+'. Resolution/timing protocols differ; these are descriptive comparisons.', '']
  if f.exists():lines+=['### Prior experiment: '+folder,'','```csv',f.read_text(encoding='utf-8-sig'),'```','']
 (REPORT/'COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
 try:
  if sys.argv[1]=='smoke':metric_smoke();report()
  elif sys.argv[1]=='report':report()
  else:evaluate(sys.argv[1],sys.argv[2])
 except BaseException:
  dump(P/'evaluation_failure.json',dict(command=sys.argv,traceback=traceback.format_exc()));raise
