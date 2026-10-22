import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
from pathlib import Path
import sys,json,time,csv,hashlib,random,subprocess,traceback,shutil
SETUP=Path(__file__).resolve().parent; ROOT=SETUP.parent
CATS=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
DATA=ROOT/'archive'; REPORT=ROOT/'reports/simplenet_supersimplenet_20261020'
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(p,stage,**kw):dump(p,dict(stage=stage,utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),**kw))
def paths(method):
 title={'simplenet':'SimpleNet','supersimplenet':'SuperSimpleNet'}[method]
 return ROOT/(title+'_MVTec_training_20261020'),ROOT/(title+'_MVTec_results_20261020')
def audit():
 import cv2,numpy as np
 rows=[];manifest=[]
 for cat in CATS:
  assert {p.name for p in (DATA/cat/'train').iterdir() if p.is_dir()}=={'good'}
  hashes={};counts={}
  for split in ['train','test']:
   files=sorted((DATA/cat/split).glob('*/*.png'));assert files;hashes[split]=set();counts[split]=len(files)
   for p in files:
    im=cv2.imread(str(p));assert im is not None;h=sha(p);hashes[split].add(h)
    row=dict(category=cat,split=split,path=str(p.relative_to(DATA)),sha256=h,shape=list(im.shape),label=int(p.parent.name!='good'))
    if split=='test' and row['label']:
     mask=DATA/cat/'ground_truth'/p.parent.name/(p.stem+'_mask.png');m=cv2.imread(str(mask),0)
     assert m is not None and m.shape==im.shape[:2] and set(np.unique(m)).issubset({0,255}) and m.any()
     row.update(mask=str(mask.relative_to(DATA)),mask_sha256=sha(mask))
    manifest.append(row)
   assert len(hashes[split])==len(files),'Duplicate images'
  assert not hashes['train']&hashes['test'],'Train/test overlap'
  rows.append(dict(category=cat,**counts));print('audit',rows[-1],flush=True)
 weights=Path.home()/'.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth';assert weights.exists() and sha(weights).startswith('95faca4d')
 dump(SETUP/'preflight/data_manifest.json',manifest);dump(SETUP/'preflight/data_inventory.json',rows)
 dump(SETUP/'preflight/pretrained.json',dict(path=weights,sha256=sha(weights),weights='IMAGENET1K_V1'))
 from evaluation_metrics import pro_metric,pro_reference
 gt=np.zeros((2,9,10),bool);gt[0,1:3,1:3]=1;gt[0,5:8,6:9]=1;gt[1,1,1]=gt[1,2,2]=1
 for sc in [np.random.default_rng(0).random(gt.shape),gt.astype(float),np.ones(gt.shape),1-gt.astype(float)]:assert np.isclose(pro_metric(sc,gt)[0],pro_reference(sc,gt),atol=1e-12)
 dump(SETUP/'preflight/audit.json',dict(passed=True,categories=15,train=sum(r['train'] for r in rows),test=sum(r['test'] for r in rows),hash_disjoint=True,masks_valid=True,aupro_reference_passed=True))

def worker(method,cat,smoke=False):
 sys.path.insert(0,str(SETUP/'runtime'/method))
 import numpy as np,torch,torchvision
 from torch.utils.data import DataLoader,Subset
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 seed=0 if method=='simplenet' else 42
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 trainroot,resultroot=paths(method);out=(SETUP/'preflight'/method if smoke else trainroot/cat);out.mkdir(parents=True,exist_ok=True)
 status=out/'status.json';state(status,'initializing',method=method,category=cat,smoke=smoke)
 loss_history=[]
 if method=='simplenet':
  import simplenet,backbones
  from datasets.mvtec import MVTecDataset,DatasetSplit
  config=dict(backbone='wideresnet50',layers=['layer2','layer3'],pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,meta_epochs=40,gan_epochs=4,embedding_size=256,noise_std=.015,dsc_hidden=1024,dsc_layers=2,dsc_margin=.5,pre_proj=1,batch=8,resize=329,imagesize=288,seed=0,lr=.001,dsc_lr=.0002,cos_lr=False,train_backbone=False,num_workers=2,checkpoint='fixed_final',train_val_split=1.0)
  model=simplenet.SimpleNet(torch.device('cuda'))
  backbone=backbones.load('wideresnet50');backbone.name='wideresnet50';backbone.seed=None
  model.load(backbone,config['layers'],torch.device('cuda'),(3,288,288),1536,1536,patchsize=3,meta_epochs=40,gan_epochs=4,embedding_size=256,noise_std=.015,dsc_hidden=1024,dsc_layers=2,dsc_margin=.5,pre_proj=1)
  model.set_model_dir(str(out),'model')
  original_scalar=model.logger.logger.add_scalar
  def scalar(tag,value,*a,**k):
   if tag=='loss':
    v=float(value.detach()) if torch.is_tensor(value) else float(value)
    if not np.isfinite(v):raise FloatingPointError('non-finite loss')
    loss_history.append(v)
   return original_scalar(tag,value,*a,**k)
  model.logger.logger.add_scalar=scalar
  ds=MVTecDataset(str(DATA),cat,resize=329,imagesize=288,split=DatasetSplit.TRAIN)
  assert all(r[1]=='good' and Path(r[2]).parent.name=='good' and Path(r[2]).parent.parent.name=='train' for r in ds.data_to_iterate)
  train_files=[r[2] for r in ds.data_to_iterate]
  loader=DataLoader(Subset(ds,range(8)) if smoke else ds,batch_size=8,shuffle=True,num_workers=0 if smoke else 2,pin_memory=True)
 else:
  import train as official
  from datamodules.mvtec import MVTec
  from model.supersimplenet import SuperSimpleNet
  captured={};official.main_mvtec=lambda device,config:captured.update(config);official.run_unsup('mvtec');config=captured
  config.update(datasets_folder=DATA.parent,checkpoint='fixed_final',ratio=1,dataset='mvtec',category=cat)
  model=SuperSimpleNet(config['image_size'],config).cuda()
  dm=MVTec(root=DATA,category=cat,image_size=config['image_size'],train_batch_size=32,eval_batch_size=1,num_workers=0 if smoke else 8,seed=42)
  dm.train_data.setup()
  frame,anom=dm.train_data.samples;assert anom.empty and (frame.label_index==0).all()
  train_files=frame.image_path.tolist();assert all(Path(p).parent.name=='good' and Path(p).parent.parent.name=='train' for p in train_files)
  def forbidden(*a,**k):raise RuntimeError('Test access forbidden during training')
  dm.test_dataloader=forbidden
  if smoke:dm.train_dataloader=lambda:DataLoader(Subset(dm.train_data,range(32)),batch_size=32,shuffle=True,num_workers=0)
 dump(out/'config.json',config);dump(out/'train_manifest.json',train_files)
 dump(out/'environment.json',dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,seed=seed,versions=json.loads((SETUP/'versions.json').read_text()),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32))
 if method=='simplenet':opts={'adaptor':model.proj_opt,'discriminator':model.dsc_opt}
 else:opts={'adamw':model.get_optimizers()[0]}
 dump(out/'optimizer_config.json',{name:[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups] for name,opt in opts.items()})
 dump(out/'model_parameters.json',{name:dict(shape=list(p.shape),requires_grad=p.requires_grad) for name,p in model.named_parameters()})
 del opts
 handles=[];grads={}
 if smoke:
  def hook(name):
   def check(g):
    assert torch.isfinite(g).all(),name;grads[name]=float(g.norm())
   return check
  for name,p in model.named_parameters():
   if p.requires_grad:handles.append(p.register_hook(hook(name)))
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 def checkpoint(epoch,loss,opt=None,sched=None):
  state(status,'training',method=method,category=cat,epoch=epoch,total_epochs=40 if method=='simplenet' else 300,loss=loss,elapsed_seconds=time.perf_counter()-start)
  with (out/'training.csv').open('a',newline='') as fp:
   w=csv.writer(fp)
   if fp.tell()==0:w.writerow(['epoch','loss','elapsed_seconds'])
   w.writerow([epoch,loss,time.perf_counter()-start])
  payload=dict(epoch=epoch,model=model.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
  if opt is not None:payload.update(optimizer=opt.state_dict(),scheduler=sched.state_dict())
  else:payload.update(proj_optimizer=model.proj_opt.state_dict(),discriminator_optimizer=model.dsc_opt.state_dict(),discriminator_scheduler=model.dsc_schl.state_dict())
  tmp=out/'recovery.tmp';torch.save(payload,tmp);tmp.replace(out/'recovery.pt')
 if method=='simplenet':
  if smoke:model.gan_epochs=1
  for epoch in range(1,2 if smoke else 41):
   previous=len(loss_history);model._train_discriminator(loader);checkpoint(epoch,float(np.mean(loss_history[previous:])))
  model.logger.logger.close()
 else:
  model.epoch_callback=checkpoint
  official.train(model,1 if smoke else 300,dm,'cuda',{}, {},clip_grad=config['clip_grad'],eval_step_size=config['eval_step_size'])
 torch.cuda.synchronize();train_seconds=time.perf_counter()-start
 memory=dict(train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),train_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 for h in handles:h.remove()
 torch.save(model.state_dict(),out/'final.pt')
 reloaded=torch.load(out/'final.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items());del reloaded
 if smoke:
  if method!='simplenet':model.eval()
  sample=(ds[0]['image'] if method=='simplenet' else dm.train_data[0]['image']).unsqueeze(0).cuda()
  with torch.no_grad():
   pred=model._predict(sample)[:2] if method=='simplenet' else model(sample)
   assert all(np.isfinite(np.asarray(v) if not torch.is_tensor(v) else v.detach().cpu().numpy()).all() for v in pred)
  required=['pre_projection','discriminator'] if method=='simplenet' else ['feature_adaptor','discriminator']
  assert all(any(k.startswith(prefix) and v>0 for k,v in grads.items()) for prefix in required),grads
  assert not any(('feature_extractor' in k or 'feature_aggregator' in k) for k in grads)
  dump(out/'smoke.json',dict(passed=True,finite_gradients=True,gradient_norms=grads,checkpoint_roundtrip=True,train_seconds=train_seconds,**memory));state(status,'smoke_passed');return
 dump(out/'training_summary.json',dict(train_seconds=train_seconds,**memory,final_checkpoint_sha256=sha(out/'final.pt')))
 state(status,'evaluating_final_checkpoint');model.load_state_dict(torch.load(out/'final.pt',map_location='cuda',weights_only=True))
 if method=='simplenet':
  del model.proj_opt,model.dsc_opt,model.dsc_schl
 else:model.eval()
 if method=='simplenet':testds=MVTecDataset(str(DATA),cat,resize=329,imagesize=288,split=DatasetSplit.TEST)
 else:dm.test_data.setup();testds=dm.test_data
 evaluate(method,cat,model,testds,train_seconds,memory,resultroot/cat)
 state(status,'complete')

def evaluate(method,cat,model,ds,train_seconds,memory,out):
 import torch,numpy as np
 from torch.utils.data import DataLoader
 from sklearn.metrics import roc_auc_score
 from evaluation_metrics import pro_metric
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out.mkdir(parents=True,exist_ok=True);maps=[];masks=[];scores=[];labels=[];times=[];records=[];example=None
 def predict(x):
  if method=='simplenet':
   sc,mp,_=model._predict(x);return np.asarray(sc).reshape(-1),np.asarray(mp)
  mp,sc=model(x);return sc.detach().cpu().sigmoid().numpy().reshape(-1),mp.detach().cpu().sigmoid().numpy().reshape(-1,*mp.shape[-2:])
 # Warmup uses train/good, never reruns a test image.
 if method=='simplenet':
  from datasets.mvtec import MVTecDataset,DatasetSplit
  warm=MVTecDataset(str(DATA),cat,resize=329,imagesize=288,split=DatasetSplit.TRAIN)[0]['image']
 else:
  import cv2
  from datamodules.mvtec import MVTec
  wdm=MVTec(root=DATA,category=cat,image_size=(256,256),num_workers=0);wdm.train_data.setup();warm=wdm.train_data[0]['image']
 with torch.no_grad():
  warm=warm.unsqueeze(0).cuda()
  for _ in range(20):predict(warm)
  del warm;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();eval_start=time.perf_counter()
  for batch in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
   x=batch['image'];torch.cuda.synchronize();t=time.perf_counter();sc,mp=predict(x.cuda());torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
   gt=batch['mask'].numpy().reshape(mp.shape)>0.5
   label=int(batch['is_anomaly'][0] if method=='simplenet' else batch['label'][0]);path=batch['image_path'][0]
   assert np.isfinite(sc).all() and np.isfinite(mp).all()
   maps.append(mp[0]);masks.append(gt[0]);scores.append(float(sc[0]));labels.append(label);records.append(dict(path=path,label=label,score=float(sc[0])))
   if label and (example is None or path<example[0]):example=(path,x[0].numpy(),gt[0],mp[0])
  evaluation_seconds=time.perf_counter()-eval_start
 memory.update(inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(),inference_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 maps=np.asarray(maps);masks=np.asarray(masks)
 result=dict(method=method,category=cat,image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(masks.ravel(),maps.ravel())),aupro_030=pro_metric(maps,masks)[0],train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,inference_batch1_mean_ms=float(np.mean(times)),inference_batch1_median_ms=float(np.median(times)),inference_batch1_p95_ms=float(np.percentile(times,95)),test_images=len(scores),**memory)
 dump(out/'metrics.json',result);dump(out/'image_scores.json',records);np.save(out/'inference_ms.npy',np.asarray(times));np.savez_compressed(out/'predictions.npz',maps=maps,masks=masks,scores=scores,labels=labels)
 path,img,gt,am=example;img=np.clip(img.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
 fig,ax=plt.subplots(1,4,figsize=(14,4));ax[0].imshow(img);ax[1].imshow(gt,cmap='gray',vmin=0,vmax=1);ax[2].imshow(am,cmap='jet');ax[3].imshow(img);ax[3].imshow(am,cmap='jet',alpha=.45)
 for a,title in zip(ax,['Input','Ground Truth','Anomaly Map','Overlay']):a.set_title(title);a.axis('off')
 fig.suptitle(method+' / '+cat);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
 dump(out/'visualization.json',dict(path=path,selection='lexicographically first anomalous test image; fixed independently of scores',mask_binarization='>0.5 after native preprocessing',map_color_scale='per-image min/max for display only'))
 aggregate()

def aggregate():
 REPORT.mkdir(parents=True,exist_ok=True);allrows=[]
 for method in ['simplenet','supersimplenet']:
  tr,rr=paths(method);rows=[json.loads(p.read_text()) for p in sorted(rr.glob('*/metrics.json'))];allrows+=rows
  if rows:
   rr.mkdir(exist_ok=True)
   with (rr/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 text=['# SimpleNet / SuperSimpleNet MVTec AD','', 'Status: results appear only after a category completes final-checkpoint evaluation. Missing categories are NOT zero scores.','', '## Protocol','', 'Only original train/good; no validation holdout; no test-time training or test-driven selection. SimpleNet: 40 x 4 training passes, fixed final checkpoint (authorized deviation from official test-AUROC selection). SuperSimpleNet: 300 epochs, official final checkpoint; intermediate tests removed. One seed per method (0 / 42), no tuning. Full official source commits and runtime patch in setup folder.','', 'Native input sizes differ: SimpleNet resize 329 / center crop 288; SuperSimpleNet resize 256x256. Metrics use each native output grid, masks binarized >0.5. Cross-method localization comparison therefore includes preprocessing differences.','', 'Image AUROC: all test image labels; SimpleNet native scores, SuperSimpleNet official sigmoid scores. Pixel AUROC: all normal and anomalous test pixels per category. AUPRO@0.30: 8-connected regions, equal region weight, global background FPR, exact unique-score sweep, linear interpolation at 0.30, area divided by 0.30. No decision threshold is fitted. Paper/official approximate AUPRO implementations may differ.','', 'Timing: batch=1, 20 train-image warmups, CUDA synchronization, includes host-to-device transfer and official postprocessing/output to CPU; excludes disk reading and input preprocessing. SimpleNet official CPU postprocessing is included. Training time includes recovery checkpoints and excludes evaluation. GPU memory is PyTorch peak allocated/reserved, not whole-device nvidia-smi memory.','', '## Results','', '| Method | Category | I-AUROC % | P-AUROC % | AUPRO % | Train s | Inference ms | Train peak MiB | Infer peak MiB |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in allrows:text.append(f"| {r['method']} | {r['category']} | {r['image_auroc']*100:.3f} | {r['pixel_auroc']*100:.3f} | {r['aupro_030']*100:.3f} | {r['train_seconds']:.1f} | {r['inference_batch1_mean_ms']:.3f} | {r['train_peak_allocated_bytes']/2**20:.1f} | {r['inference_peak_allocated_bytes']/2**20:.1f} |")
 for method in ['simplenet','supersimplenet']:
  rows=[r for r in allrows if r['method']==method]
  if len(rows)==15:
   means={k:sum(r[k] for r in rows)/15 for k in ['image_auroc','pixel_auroc','aupro_030','train_seconds','inference_batch1_mean_ms','train_peak_allocated_bytes','inference_peak_allocated_bytes']};dump(REPORT/(method+'_mean.json'),means);text+=['',method+' macro mean (15 categories): '+json.dumps(means)]
 if all(len([r for r in allrows if r['method']==m])==15 for m in ['simplenet','supersimplenet']):
  a=json.loads((REPORT/'simplenet_mean.json').read_text());b=json.loads((REPORT/'supersimplenet_mean.json').read_text());text+=['','## Observed transition','',f"SuperSimpleNet minus SimpleNet: image AUROC {(b['image_auroc']-a['image_auroc'])*100:.3f} percentage points; pixel AUROC {(b['pixel_auroc']-a['pixel_auroc'])*100:.3f}; AUPRO {(b['aupro_030']-a['aupro_030'])*100:.3f}. Speed ratio SimpleNet/SuperSimpleNet: {a['inference_batch1_mean_ms']/b['inference_batch1_mean_ms']:.3f}. Training peak memory ratio SSN/SN: {b['train_peak_allocated_bytes']/a['train_peak_allocated_bytes']:.3f}; inference peak ratio: {b['inference_peak_allocated_bytes']/a['inference_peak_allocated_bytes']:.3f}. Single-seed measurements do not establish statistical significance."]
 text+=['','## Reproduce','', 'From workspace root (existing completed category outputs are skipped by controller):','```powershell',r'.\SimpleNet_SuperSimpleNet_setup_20261020\venv\Scripts\python.exe -u .\SimpleNet_SuperSimpleNet_setup_20261020\experiment.py controller','```','', 'Individual command: `experiment.py worker simplenet bottle` or `experiment.py worker supersimplenet bottle`. Use a fresh dated output root for a new independent experiment. Environment: `setup/pip_freeze.txt`; exact category settings: training directory `config.json`; splits and SHA256: setup `preflight/data_manifest.json`.']
 (REPORT/'report.md').write_text('\n'.join(text),encoding='utf-8')
 from paper_report import build
 build(REPORT,SETUP,allrows)
 if allrows:
  with (REPORT/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)

def controller():
 state(SETUP/'status.json','preflight')
 if not (SETUP/'preflight/audit.json').exists():audit()
 (SETUP/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 for method in ['simplenet','supersimplenet']:
  with (SETUP/(method+'_smoke.log')).open('w') as log:subprocess.run([sys.executable,'-u',__file__,'smoke',method,'bottle'],stdout=log,stderr=subprocess.STDOUT,check=True)
  assert json.loads((SETUP/'preflight'/method/'smoke.json').read_text())['passed']
 aggregate()
 for method in ['simplenet','supersimplenet']:
  tr,rr=paths(method);tr.mkdir(exist_ok=True)
  for cat in CATS:
   if (rr/cat/'metrics.json').exists():continue
   state(SETUP/'status.json','training',method=method,category=cat)
   with (tr/(cat+'.log')).open('w') as log:subprocess.run([sys.executable,'-u',__file__,'worker',method,cat],stdout=log,stderr=subprocess.STDOUT,check=True)
 aggregate();state(SETUP/'status.json','complete')
if __name__=='__main__':
 try:
  if sys.argv[1]=='controller':controller()
  elif sys.argv[1]=='audit':audit()
  elif sys.argv[1]=='report':aggregate()
  else:worker(sys.argv[2],sys.argv[3],sys.argv[1]=='smoke')
 except Exception:
  state(SETUP/'status.json','failed',command=sys.argv,error=traceback.format_exc());raise
