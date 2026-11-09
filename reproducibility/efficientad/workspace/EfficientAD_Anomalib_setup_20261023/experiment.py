import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
import json,random,time,hashlib,sys,csv,traceback,subprocess
P=Path(__file__).resolve().parent; ROOT=P.parent; DATA=ROOT/'archive'
CATS='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()
REPORT=ROOT/'reports/efficientad_anomalib_20261023'
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False),encoding='utf-8');q.replace(p)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(p,stage,**kw):dump(p,dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw))
def paths(size):return ROOT/f'EfficientAD_{size.upper()}_Anomalib_training_20261023',ROOT/f'EfficientAD_{size.upper()}_Anomalib_results_20261023'
def audit():
 from PIL import Image
 import numpy as np
 rows=[]
 for cat in CATS:
  assert {q.name for q in (DATA/cat/'train').iterdir() if q.is_dir()}=={'good'}
  files=sorted((DATA/cat/'train/good').glob('*.png'));assert files
  indices=np.random.default_rng(42).permutation(len(files));n=int(.8*len(files));splits={}
  hashes=set()
  for role,ids in [('fit',indices[:n]),('validation',indices[n:])]:
   entries=[]
   for i in ids:
    f=files[i]
    with Image.open(f) as im:im.load();shape=im.size
    h=sha(f);assert h not in hashes;hashes.add(h)
    entries.append(dict(path=f.relative_to(DATA).as_posix(),sha256=h,size=shape))
   splits[role]=entries
  dump(P/'splits'/f'{cat}.json',dict(seed=42,algorithm='numpy.default_rng(42).permutation(sorted paths); floor(0.8*N)',**splits))
  rows.append(dict(category=cat,fit=n,validation=len(files)-n))
 dump(P/'inventory.json',rows)
 print('TRAIN_ONLY_AUDIT_OK',rows,flush=True)
def runtime():
 import torch,numpy as np
 torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available()
 return torch,np
def worker(size,cat,smoke=False):
 torch,np=runtime()
 from torchvision.transforms.v2 import Compose,Resize,ToTensor,RandomGrayscale,CenterCrop
 from torchvision.datasets import ImageFolder
 from PIL import Image
 from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
 from types import SimpleNamespace
 tr,res=paths(size);out=P/'smoke'/size if smoke else tr/cat;out.mkdir(parents=True,exist_ok=True)
 assert smoke or not (out/'final.pt').exists(),'Final exists: do not overwrite'
 st=out/'status.json';status(st,'initializing')
 split=json.loads((P/'splits'/f'{cat}.json').read_text());fit=[DATA/e['path'] for e in split['fit']];val=[DATA/e['path'] for e in split['validation']]
 assert not set(fit)&set(val)
 transform=Compose([ToTensor(),Resize((256,256),antialias=True)])
 def read(f):
  assert f.parent.name=='good' and f.parent.parent.name=='train'
  with Image.open(f) as im:return transform(im.convert('RGB')).unsqueeze(0).cuda()
 module=EfficientAd(model_size={'s':'small','m':'medium'}[size],padding=False,pad_maps=True,pre_processor=False,post_processor=False,evaluator=False,visualizer=False).cuda()
 model=module.model
 weight=next((P/'assets').rglob(f'pretrained_teacher_{"small" if size=="s" else "medium"}.pth'))
 model.teacher.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True),strict=True);model.teacher.requires_grad_(False)
 config=dict(size=size,category=cat,seed=42,steps=70000,batch_size=1,input_size=256,lr=1e-4,weight_decay=1e-5,adam_betas=[.9,.999],adam_eps=1e-8,lr_drop_step=66500,lr_gamma=.1,padding=False,pad_maps=True,precision='float32',tf32=False,checkpoint='fixed_final',validation='normal calibration only after training',split_manifest=str(P/'splits'/f'{cat}.json'),split_sha256=sha(P/'splits'/f'{cat}.json'),teacher=str(weight),teacher_sha256=sha(weight),anomalib=json.loads((P/'version.json').read_text()))
 dump(out/'config.json',config)
 dump(out/'data_flow.json',dict(teacher_statistics=[str(f) for f in (fit[:4] if smoke else fit)],training_candidates=[str(f) for f in fit],calibration=[str(f) for f in (val[:3] if smoke else val)],auxiliary_root=str(P/'assets/imagenette2/train'),test_loader_created=False))
 dump(out/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),python=sys.version))
 model.eval()
 stats=module.teacher_channel_mean_std(SimpleNamespace(image=read(f)) for f in (fit[:4] if smoke else fit))
 assert torch.isfinite(stats['std']).all() and (stats['std']>0).all();model.mean_std.update(stats);model.mean_std.requires_grad_(False)
 aux=ImageFolder(P/'assets/imagenette2/train',transform=Compose([Resize((512,512)),RandomGrayscale(.3),CenterCrop((256,256)),ToTensor()]))
 assert len(aux)==9469,len(aux)
 optimizer=torch.optim.Adam(list(model.student.parameters())+list(model.ae.parameters()),lr=1e-4,weight_decay=1e-5)
 scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=66500,gamma=.1)
 # Separate deterministic permutation streams; no test dataset exists in this worker.
 fg=torch.Generator().manual_seed(42);ag=torch.Generator().manual_seed(43);fi=[];ai=[]
 def nextindex(queue,n,g):
  if not queue:queue.extend(torch.randperm(n,generator=g).tolist())
  return queue.pop()
 steps=3 if smoke else 70000;history=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 model.train();model.teacher.eval()
 for step in range(1,steps+1):
  x=read(fit[nextindex(fi,len(fit),fg)]);a=aux[nextindex(ai,len(aux),ag)][0].unsqueeze(0).cuda()
  optimizer.zero_grad(set_to_none=True);losses=model(x,a);loss=sum(losses)
  assert torch.isfinite(loss),'Nonfinite loss'
  loss.backward()
  if smoke or step==1:
   grad={name:float(sum((p.grad.detach().square().sum() for p in net.parameters() if p.grad is not None),torch.tensor(0.,device='cuda')).sqrt()) for name,net in [('student',model.student),('ae',model.ae)]}
   assert all(np.isfinite(v) and v>0 for v in grad.values());assert all(p.grad is None for p in model.teacher.parameters())
  optimizer.step();scheduler.step()
  if step==1 or step%100==0 or step==steps:
   torch.cuda.synchronize();history.append(dict(step=step,loss=float(loss),components=[float(v) for v in losses],lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-start))
   status(st,'training',step=step,total=steps,loss=float(loss),seconds=time.perf_counter()-start);print(size,cat,history[-1],flush=True)
  if step%5000==0 or step==steps:
   payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,config=config,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),fit_queue=fi,aux_queue=ai,fit_generator=fg.get_state(),aux_generator=ag.get_state())
   torch.save(payload,out/'recovery.tmp');(out/'recovery.tmp').replace(out/'recovery.pt')
 torch.cuda.synchronize();train_seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();model.eval()
 status(st,'calibrating',step=steps)
 with torch.no_grad():
  maps=[[],[]]
  for f in (val[:3] if smoke else val):
   for store,m in zip(maps,model.get_maps(read(f),normalize=False)):store.append(m.cpu())
  q={}
  for name,ms in zip(['st','ae'],maps):
   flat=torch.cat(ms).flatten();qa,qb=torch.quantile(flat,torch.tensor([.9,.995]));assert qb>qa
   q['qa_'+name]=qa.cuda();q['qb_'+name]=qb.cuda()
  model.quantiles.update(q);model.quantiles.requires_grad_(False)
  pred=model(read(val[0])).anomaly_map.detach().clone()
  assert pred.shape==(1,1,256,256) and torch.isfinite(pred).all()
  torch.save(dict(model=model.state_dict(),step=steps,config=config),out/'final.pt')
  model.load_state_dict(torch.load(out/'final.pt',weights_only=False,map_location='cuda')['model']);assert torch.equal(pred,model(read(val[0])).anomaly_map)
 dump(out/'loss_history.json',history)
 metrics=dict(training_seconds=train_seconds,training_peak_allocated_mb=peak/2**20,training_peak_reserved_mb=reserved/2**20,steps=steps,final_checkpoint=str(out/'final.pt'),calibration={k:float(v) for k,v in q.items()},gradient=grad,checkpoint_reload_equal=True,fit_count=len(fit),validation_count=len(val),test_images_seen=0,imagenette_count=len(aux))
 dump(out/'training.json',metrics);status(st,'smoke_passed' if smoke else 'trained',**metrics)
 print('WORKER_DONE',size,cat,smoke,flush=True)

def controller():
 status(P/'status.json','preflight');audit()
 subprocess.run([sys.executable,str(P/'evaluate.py'),'smoke'],check=True)
 for size in ['s','m']:
  subprocess.run([sys.executable,__file__,'smoke',size,'bottle'],check=True)
 status(P/'status.json','smokes_passed')
 for size in ['s','m']:
  for cat in CATS:
   status(P/'status.json','training',model=size,category=cat)
   subprocess.run([sys.executable,__file__,'train',size,cat],check=True)
  # Evaluation begins only after the full model's 15 categories have trained.
  for cat in CATS:
   status(P/'status.json','evaluating',model=size,category=cat)
   subprocess.run([sys.executable,str(P/'evaluate.py'),size,cat],check=True)
 status(P/'status.json','complete')
if __name__=='__main__':
 try:
  if sys.argv[1]=='audit':audit()
  elif sys.argv[1]=='run':controller()
  else:worker(sys.argv[2],sys.argv[3],sys.argv[1]=='smoke')
 except BaseException:
  dump(P/'failure.json',dict(command=sys.argv,traceback=traceback.format_exc()));status(P/'status.json','failed',command=sys.argv);raise
