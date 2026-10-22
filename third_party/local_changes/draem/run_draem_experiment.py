import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
from pathlib import Path
import sys,json,csv,time,random,hashlib,shutil,traceback,subprocess,re,contextlib,io
ROOT=Path(os.environ.get('DRAEM_WORKSPACE',str(Path(__file__).resolve().parent)));OUT=ROOT/'DRAEM_MVTec_all15_20261015';SETUP=ROOT/'DRAEM_setup';SNAP=OUT/'source_snapshot'
sys.path[:0]=[str(SETUP/'dependencies'),str(SETUP),str(SNAP if SNAP.exists() else ROOT/'DRAEM-main')]
import numpy as np
# imgaug0.4 compatibility with NumPy1.26: alias only, no algorithm changes.
if 'bool' not in np.__dict__:np.bool=np.bool_
import torch,cv2,imgaug
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score
from data_loader import MVTecDRAEMTrainDataset,MVTecDRAEMTestDataset
from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork
from loss import SSIM,FocalLoss
from metrics import pro_metric,pro_reference
CATS=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'];SEED=1024
PAPER_I=[99.2,91.8,98.5,97,99.9,100,100,98.7,98.9,93.9,99.6,100,93.1,99.1,100]
PAPER_P=[99.1,94.7,94.3,95.5,99.7,99.7,98.6,99.5,97.6,97.6,99.2,98.1,90.9,96.4,98.8]
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def dump(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
 for i in range(40):
  try:tmp.replace(p);return
  except PermissionError:
   if i==39:raise
   time.sleep(.1)
def status(folder,stage,**kw):dump(folder/'status.json',dict(stage=stage,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),**kw))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);imgaug.seed(SEED)
def sync():torch.cuda.synchronize()
def models():return ReconstructiveSubNetwork(3,3).cuda(),DiscriminativeSubNetwork(6,2).cuda()
def init(m):
 name=m.__class__.__name__
 if 'Conv' in name:m.weight.data.normal_(0.,.02)
 elif 'BatchNorm' in name:m.weight.data.normal_(1.,.02);m.bias.data.fill_(0)
def training_objects():
 a,b=models();a.apply(init);b.apply(init);opt=torch.optim.Adam([{'params':a.parameters(),'lr':1e-4},{'params':b.parameters(),'lr':1e-4}]);sched=torch.optim.lr_scheduler.MultiStepLR(opt,[560.,630.],gamma=.2,last_epoch=-1)
 return a,b,opt,sched
def dataset(cat):return MVTecDRAEMTrainDataset(str(ROOT/'archive'/cat/'train/good')+'/',str(SETUP/'datasets/dtd/images')+'/',resize_shape=[256,256])
def loader(cat):return DataLoader(dataset(cat),batch_size=8,shuffle=True,num_workers=0)
def train_step(a,b,opt,batch,ssim,focal):
 clean=batch['image'].cuda();x=batch['augmented_image'].cuda();gt=batch['anomaly_mask'].cuda();rec=a(x);p=torch.softmax(b(torch.cat((rec,x),1)),1);terms=[torch.nn.functional.mse_loss(rec,clean),ssim(rec,clean),focal(p,gt)];loss=sum(terms);assert torch.isfinite(loss)
 opt.zero_grad();loss.backward()
 norms=[]
 for mod in [a,b]:
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters() if p.requires_grad)
  norms.append(float(torch.sqrt(sum(p.grad.detach().square().sum() for p in mod.parameters() if p.grad is not None))))
 assert min(norms)>0;opt.step();return [float(loss.detach())]+[float(t.detach()) for t in terms],norms

def check_data():
 f=OUT/'preflight';f.mkdir(exist_ok=True);status(f,'data_audit');rows=[];manifest=[]
 for cat in CATS:
  tr=sorted((ROOT/'archive'/cat/'train/good').glob('*.png'));te=sorted((ROOT/'archive'/cat/'test').glob('*/*.png'));assert tr and te
  assert {p.name for p in (ROOT/'archive'/cat/'train').iterdir() if p.is_dir()}=={'good'}
  hashes={};nmask=0
  for split,ps in [('train',tr),('test',te)]:
   hashes[split]=set()
   for p in ps:
    im=cv2.imread(str(p));assert im is not None and im.ndim==3 and im.shape[2]==3;h=sha(p);hashes[split].add(h);r=dict(path=str(p.relative_to(ROOT)),sha256=h,split=split,category=cat,label=int(split=='test' and p.parent.name!='good'),shape=list(im.shape))
    if r['label']:
     mp=ROOT/'archive'/cat/'ground_truth'/p.parent.name/(p.stem+'_mask.png');mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE);assert mask is not None and mask.shape==im.shape[:2] and set(np.unique(mask)).issubset({0,255}) and mask.any();r.update(mask=str(mp.relative_to(ROOT)),mask_sha256=sha(mp));nmask+=1
    manifest.append(r)
  assert not hashes['train']&hashes['test'];assert len(list((ROOT/'archive'/cat/'ground_truth').glob('*/*.png')))==nmask
  rows.append(dict(category=cat,train_good=len(tr),test=len(te),normal_test=len(te)-nmask,anomaly_test=nmask));status(f,'data_audit',category=cat,completed=len(rows))
 dump(f/'data_manifest.json',manifest);dump(f/'data_inventory.json',rows)
 with (f/'data_inventory.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rng=np.random.default_rng(SEED);gt=np.zeros((2,9,10),bool);gt[0,1:3,1:3]=1;gt[0,5:8,6:9]=1;gt[1,1,1]=gt[1,2,2]=1
 for s in [rng.random(gt.shape),rng.integers(0,5,gt.shape).astype(float),gt.astype(float),np.ones(gt.shape),1-gt.astype(float)]:assert np.isclose(pro_metric(s,gt)[0],pro_reference(s,gt),atol=1e-12)
 # Verify the actual modern scheduler uses the user-approved epoch boundaries.
 p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.Adam([p],lr=1e-4);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[560.,630.],gamma=.2);lrs=[]
 for epoch in range(1,701):lrs.append(opt.param_groups[0]['lr']);opt.step();sch.step()
 assert lrs[559]==1e-4 and np.isclose(lrs[560],2e-5) and np.isclose(lrs[630],4e-6)
 a,b=models();params=sum(p.numel() for mod in [a,b] for p in mod.parameters());del a,b;torch.cuda.empty_cache();free=shutil.disk_usage(ROOT).free
 assert free>30*1024**3
 dump(f/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,imgaug=imgaug.__version__,opencv=cv2.__version__,python=sys.version,gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,disk_free_bytes=free,parameters=params,estimated_all15_weights_plus_recovery_bytes=params*4*15*4))
 dump(f/'data_checks.json',dict(passed=True,categories=15,train_images=sum(r['train_good'] for r in rows),test_images=sum(r['test'] for r in rows),mask_alignment_binary_and_hashes=True,train_test_hash_disjoint=True,aupro_synthetic_reference_passed=True,scheduler_verified=True))
 status(f,'data_checks_passed')

def check_synthetic():
 f=OUT/'preflight';ps=sorted((SETUP/'datasets/dtd/images').glob('*/*.jpg'));assert len(ps)==5640
 for p in ps:assert cv2.imread(str(p)) is not None,str(p)
 seed();ds=dataset('bottle');counts=[0,0];examples=[]
 for i in range(64):
  s=ds[i];mask=s['anomaly_mask'];x=s['image'];aug=s['augmented_image'];assert x.shape==aug.shape==(3,256,256) and mask.shape==(1,256,256)
  assert np.isfinite(x).all() and np.isfinite(aug).all() and set(np.unique(mask)).issubset({0,1}) and 0<=x.min() and x.max()<=1 and -1e-6<=aug.min() and aug.max()<=1+1e-6
  has=int(s['has_anomaly'][0]);assert bool(mask.any())==bool(has);assert np.allclose((aug-x)*(1-mask),0,atol=1e-7);counts[has]+=1
  if (has and sum(t[0] for t in examples)<3) or (not has and not any(t[0]==0 for t in examples)):examples.append((has,s))
 assert min(counts)>0
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(len(examples),3,figsize=(10,3*len(examples)),layout='constrained')
 for r,(has,s) in enumerate(examples):
  for c,k in enumerate(['image','augmented_image','anomaly_mask']):
   im=s[k].transpose(1,2,0);im=im[...,::-1] if c<2 else im[...,0];axs[r,c].imshow(im,cmap='gray' if c==2 else None);axs[r,c].axis('off');axs[r,c].set_title(k)
 fig.savefig(f/'synthetic_training_checks.png',dpi=130);plt.close(fig)
 dump(f/'synthetic_checks.json',dict(passed=True,dtd_images=len(ps),samples=64,normal=counts[0],synthetic_anomaly=counts[1],binary_masks=True,outside_mask_unchanged=True,source='official dataset code unchanged; np.bool compatibility alias'))

@torch.inference_mode()
def evaluate(cat,a,b,folder,panel=False):
 folder.mkdir(exist_ok=True);a.eval();b.eval();ds=MVTecDRAEMTestDataset(str(ROOT/'archive'/cat/'test')+'/',resize_shape=[256,256]);dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);maps=[];gts=[];scores=[];labels=[];soft_boundary=0
 bad=[i for i,p in enumerate(ds.images) if Path(p).parent.name!='good'];chosen=int(np.random.default_rng(SEED).choice(bad));vis=None
 if panel:dump(folder/'visual_selection.json',dict(seed=SEED,path=ds.images[chosen],policy='one anomalous image sampled before scores; no score-based choice'))
 sync();start=time.perf_counter()
 for i,s in enumerate(dl):
  x=s['image'].cuda();rec=a(x);p=torch.softmax(b(torch.cat((rec.detach(),x),1)),1);am=p[0,1].cpu().numpy();score=float(torch.nn.functional.avg_pool2d(p[:,1:],21,stride=1,padding=10).max().cpu());mask=s['mask'][0,0].numpy();soft_boundary+=int(((mask>0)&(mask<1)).sum());gt=mask.astype(np.uint8).astype(bool);label=int(s['has_anomaly'][0,0])
  assert label==int(Path(ds.images[i]).parent.name!='good');assert np.isfinite(am).all() and 0<=am.min() and am.max()<=1
  maps.append(am);gts.append(gt);scores.append(score);labels.append(label)
  if panel and i==chosen:vis=(s['image'][0].numpy().transpose(1,2,0)[...,::-1],gt,am)
  if i%20==0:status(folder,'evaluation',category=cat,images_done=i+1,total=len(ds))
 sync();secs=time.perf_counter()-start;maps=np.asarray(maps);gts=np.asarray(gts);labels=np.asarray(labels);scores=np.asarray(scores)
 au,n,_=pro_metric(maps,gts);m=dict(image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(gts.ravel(),maps.ravel())),aupro_at_030=au,image_ap=float(average_precision_score(labels,scores)),pixel_ap=float(average_precision_score(gts.ravel(),maps.ravel())),normal_images=int((labels==0).sum()),anomaly_images=int(labels.sum()),regions=n,inference_seconds=secs,inference_seconds_per_image=secs/len(ds),mask_partial_pixels_cast_to_zero=soft_boundary,inference_scope='load/preprocess+GPU forward+softmax+pool+CPU output; excludes metric/plot; first batch included')
 dump(folder/'metrics.json',m)
 with (folder/'per_image.csv').open('w',newline='') as fp:
  w=csv.writer(fp);w.writerow(['path','label','image_score']);w.writerows(zip(ds.images,labels.tolist(),scores.tolist()))
 if panel:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  x,gt,am=vis;fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained');axs[0].imshow(x);axs[1].imshow(gt,cmap='gray',vmin=0,vmax=1);axs[2].imshow(am,cmap='inferno',vmin=0,vmax=1);axs[3].imshow(x);axs[3].imshow(am,cmap='inferno',vmin=0,vmax=1,alpha=.45)
  for ax,title in zip(axs,['Original','Ground Truth','Anomaly probability','Overlay (no threshold)']):ax.axis('off');ax.set_title(title)
  fig.suptitle(cat+' / '+Path(ds.images[chosen]).parent.name+'/'+Path(ds.images[chosen]).name);fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
 return m

def pretrained_check():
 f=OUT/'pretrained_validation';f.mkdir(exist_ok=True);files=list((SETUP/'pretrained').rglob('*bottle_.pckl'));assert len(files)==1
 base=files[0].name[:-len('_bottle_.pckl')];directory=files[0].parent;allm={};hashes={}
 for cat in CATS:
  a,b=models()
  for mod,suffix in [(a,'.pckl'),(b,'_seg.pckl')]:
   p=directory/(base+'_'+cat+'_'+suffix);assert p.is_file();hashes[str(p.relative_to(ROOT))]=sha(p);mod.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
  allm[cat]=evaluate(cat,a,b,f/cat);del a,b;torch.cuda.empty_cache();dump(f/'our_metrics.json',allm)
 # Run author's complete evaluator unchanged, with only external compatibility/no-grad context.
 import test_DRAEM
 original=f/'original';original.mkdir(exist_ok=True);old=Path.cwd();os.chdir(original)
 try:
  with (original/'stdout.log').open('w') as stream,contextlib.redirect_stdout(stream),torch.no_grad():test_DRAEM.test(CATS,str(ROOT/'archive')+'/',str(directory)+'/',base)
 finally:os.chdir(old)
 log=(original/'stdout.log').read_text();checks={}
 for cat in CATS:
  match=re.search(r'^'+re.escape(cat)+r'\s*\nAUC Image:\s*([^\n]+)\nAP Image:\s*([^\n]+)\nAUC Pixel:\s*([^\n]+)\nAP Pixel:\s*([^\n]+)',log,re.M);assert match,cat
  ref=dict(zip(['image_auroc','image_ap','pixel_auroc','pixel_ap'],map(float,match.groups())));errs={k:abs(allm[cat][k]-v) for k,v in ref.items()};checks[cat]=dict(original=ref,max_abs_error=max(errs.values()));assert max(errs.values())<1e-6,(cat,errs)
 means={k:float(np.mean([r[k] for r in allm.values()])) for k in ['image_auroc','pixel_auroc','pixel_ap']}
 # Published rounded pretrained summary: broad predeclared sanity bound; not model/HP selection.
 assert abs(means['image_auroc']-.981)<.005 and abs(means['pixel_auroc']-.975)<.005 and abs(means['pixel_ap']-.689)<.01,means
 dump(f/'validation.json',dict(passed=True,base_model_name=base,metrics_match_original=checks,macro=means,checkpoint_sha256=hashes,no_hyperparameter_selection=True));status(f,'complete')

def smoke():
 f=OUT/'preflight';status(f,'smoke');check_synthetic();seed();a,b,opt,sched=training_objects();ssim=SSIM();focal=FocalLoss();before=[next(mod.parameters()).detach().clone() for mod in [a,b]];times=[];losses=[];torch.cuda.reset_peak_memory_stats();dl=loader('bottle')
 for i,s in enumerate(dl):
  sync();start=time.perf_counter();ls,gn=train_step(a,b,opt,s,ssim,focal);sync();times.append(time.perf_counter()-start);losses.append(ls)
  if i==7:break
 assert all(not torch.equal(v,next(mod.parameters())) for v,mod in zip(before,[a,b]));x=s['augmented_image'].cuda();a.eval();b.eval()
 with torch.no_grad():expected=b(torch.cat((a(x),x),1)).cpu()
 cp=f/'smoke_checkpoint.pt';torch.save(dict(reconstructive=a.state_dict(),discriminative=b.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),seed=SEED),cp)
 with torch.no_grad():next(a.parameters()).add_(1)
 ck=torch.load(cp,map_location='cpu',weights_only=False);a.load_state_dict(ck['reconstructive']);b.load_state_dict(ck['discriminative']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
 with torch.no_grad():actual=b(torch.cat((a(x),x),1)).cpu()
 assert torch.equal(expected,actual)
 peak=torch.cuda.max_memory_allocated();assert peak<torch.cuda.get_device_properties(0).total_memory
 dump(f/'smoke_checks.json',dict(passed=True,train_only=True,steps=8,batch_size=8,loss_columns=['total','MSE','SSIM','Focal'],losses=losses,last_gradient_norms=gn,both_networks_updated=True,save_reload_output_bitwise_equal=True,peak_gpu_allocated_bytes=peak,batch_seconds=times,bottle_estimated_training_hours=float(np.median(times[2:])*len(dl)*700/3600),smoke_checkpoint_sha256=sha(cp),smoke_weights_discarded=True))
 status(f,'complete')

def train(cat):
 f=OUT/cat;f.mkdir(exist_ok=False);seed();a,b,opt,sched=training_objects();ssim=SSIM();focal=FocalLoss();dl=loader(cat);dump(f/'protocol.json',dict(category=cat,epochs=700,batch=8,lr=1e-4,milestones=[560,630],gamma=.2,seed=SEED,source_commit='2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef',test_selection=False,train_normal_only=True,workers=0,AMP=False,TF32=False,preprocessing='official BGR/255 cv2linear256',dataset_sampling='official random replacement',train_images=len(dl.dataset)))
 sync();start=time.perf_counter()
 with (f/'epoch_metrics.csv').open('w',newline='') as fp:
  w=csv.writer(fp);w.writerow(['epoch','loss','mse','ssim','focal','lr_used','epoch_seconds','training_elapsed_seconds']);fp.flush()
  for epoch in range(1,701):
   a.train();b.train();sums=np.zeros(4);n=0;et=time.perf_counter();lr=opt.param_groups[0]['lr']
   for i,s in enumerate(dl):
    ls,gn=train_step(a,b,opt,s,ssim,focal);sums+=ls;n+=1
    if i%10==0:status(f,'training',category=cat,epoch=epoch,total_epochs=700,batch=i+1,total_batches=len(dl),loss=ls[0],elapsed_seconds=time.perf_counter()-start)
   sched.step();sync();elapsed=time.perf_counter()-start;w.writerow([epoch,*(sums/n),lr,time.perf_counter()-et,elapsed]);fp.flush()
   # Same latest-weight retention as official code; final names created only at700.
   for name,mod in [('reconstructive',a),('discriminative',b)]:
    tmp=f/(name+'_latest.tmp');torch.save(mod.state_dict(),tmp);tmp.replace(f/(name+'_latest.pt'))
   if epoch%25==0:
    tmp=f/'recovery.tmp';torch.save(dict(epoch=epoch,reconstructive=a.state_dict(),discriminative=b.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),imgaug_rng=imgaug.random.get_global_rng().state),tmp);tmp.replace(f/'recovery_latest.pt')
   print(cat,epoch,float(sums[0]/n),flush=True)
 seconds=time.perf_counter()-start
 for name in ['reconstructive','discriminative']:(f/(name+'_latest.pt')).rename(f/(name+'_final.pt'))
 dump(f/'training_summary.json',dict(epochs=700,training_seconds=seconds,seed=SEED,test_used=False,final_checkpoints={n:sha(f/(n+'_final.pt')) for n in ['reconstructive','discriminative']}))
 del opt,sched,ssim,focal;torch.cuda.empty_cache();m=evaluate(cat,a,b,f,panel=True);m['training_seconds']=seconds;dump(f/'metrics.json',m);status(f,'complete',category=cat,epochs_completed=700,metrics=m)

def aggregate():
 rows=[]
 for cat in CATS:
  f=OUT/cat
  if (f/'status.json').exists() and json.loads((f/'status.json').read_text())['stage']=='complete':
   m=json.loads((f/'metrics.json').read_text());i=CATS.index(cat);rows.append(dict(category=cat,**m,paper_image_auroc=PAPER_I[i]/100,paper_pixel_auroc=PAPER_P[i]/100))
 if not rows:return
 with (OUT/'results.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# DRAEM — resmi kod tabanlı MVTec AD deneyi','',f'Tamamlanan kategori: {len(rows)}/15. Tüm sayılar kayıtlı final700 checkpoint testinden; testten model/eşik seçilmedi.','', '| Kategori | Image AUROC % | Pixel AUROC % | AUPRO@0.30 % | Eğitim saat | sn/görüntü | Makale Image / Pixel % |','|---|---:|---:|---:|---:|---:|---|']
 for r in rows:lines.append(f"| {r['category']} | {100*r['image_auroc']:.2f} | {100*r['pixel_auroc']:.2f} | {100*r['aupro_at_030']:.2f} | {r['training_seconds']/3600:.3f} | {r['inference_seconds_per_image']:.4f} | {100*r['paper_image_auroc']:.1f} / {100*r['paper_pixel_auroc']:.1f} |")
 means={k:float(np.mean([r[k] for r in rows])) for k in ['image_auroc','pixel_auroc','aupro_at_030']};dump(OUT/'summary.json',dict(completed=len(rows),macro_completed_only=means));lines+=['','Tamamlananlar üzerinde eşit kategori ortalaması (%): '+str({k:round(100*v,3) for k,v in means.items()}),'','Protokol ve makale farkları: [protokol_TR.md](protokol_TR.md). AUPRO ek metriktir; makaledeki Pixel AP ile aynı değildir. Resmi mask linear resize/uint8 ve BGR korunur. Image skoru21×21 ortalama sonrası maksimum. Seed1024, tek tekrar. Makale: https://arxiv.org/html/2108.07610 ; resmi kod: https://github.com/VitjanZ/DRAEM','']
 for r in rows:lines += [f"![{r['category']} tek kusurlu örnek]({r['category']}/comparison.png)",'']
 (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
 mode=sys.argv[1]
 if mode=='all':
  for gate in ['data_checks.json','synthetic_checks.json','smoke_checks.json']:assert json.loads((OUT/'preflight'/gate).read_text())['passed']
  assert json.loads((OUT/'pretrained_validation/validation.json').read_text())['passed']
  for cat in CATS:
   assert not (OUT/cat).exists();env=os.environ.copy();env['DRAEM_WORKSPACE']=str(ROOT)
   with (OUT/(cat+'.stdout.log')).open('w') as so,(OUT/(cat+'.stderr.log')).open('w') as se:
    p=subprocess.Popen([sys.executable,'-u',str(SNAP/'run_draem_experiment.py'),'train',cat],cwd=ROOT,env=env,stdout=so,stderr=se)
    while p.poll() is None:
     ws={}
     try:ws=json.loads((OUT/cat/'status.json').read_text())
     except (OSError,ValueError):pass
     status(OUT,'running',category=cat,completed=CATS.index(cat),total=15,worker_pid=p.pid,worker_status=ws);time.sleep(10)
    assert p.returncode==0 and json.loads((OUT/cat/'status.json').read_text())['stage']=='complete',cat
   aggregate()
  status(OUT,'complete',completed=15,total=15)
 elif mode=='data':check_data()
 elif mode=='pretrained':pretrained_check()
 elif mode=='smoke':smoke()
 elif mode=='train':train(sys.argv[2])
if __name__=='__main__':
 try:main()
 except Exception:
  target=OUT/sys.argv[2] if len(sys.argv)>2 and sys.argv[1]=='train' else OUT
  if target.exists():status(target,'failed',mode=sys.argv[1],error=traceback.format_exc())
  raise
