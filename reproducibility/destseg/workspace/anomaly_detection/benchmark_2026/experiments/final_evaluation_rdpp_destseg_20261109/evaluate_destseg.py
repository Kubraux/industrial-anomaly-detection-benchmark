"""Evaluation only; immutable trained DeSTSeg source and final weights."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time,argparse,traceback,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parents[1];ROOT=B.parent.parent
TRAIN=B/'experiments/destseg_auto_mvtec_seed111_20261103'
os.environ['TORCH_HOME']=str(B/'models/destseg/torch_cache')
sys.path.insert(0,str(TRAIN/'source'))
import numpy as np,torch,psutil
from PIL import Image
from torchvision import transforms
from model.destseg import DeSTSeg
from data.mvtec_dataset import MVTecDataset
from sklearn.metrics import roc_auc_score,average_precision_score
spec=importlib.util.spec_from_file_location('common',ROOT/'compare_dfr_rd4ad_common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,default=str),encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state_hash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def preprocess(p):
 with Image.open(p) as im:return transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
def score(m,x):
 raw=m(x)[0];amap=torch.nn.functional.interpolate(raw,(256,256),mode='bilinear',align_corners=False)
 return amap,amap.flatten(1).sort(descending=True).values[:,:100].mean(1)
def masks(image,mask):
 with Image.open(image) as im:size=im.size
 if mask is None:return np.zeros((256,256),bool),np.zeros((256,256),bool),size
 with Image.open(mask) as im:
  a=np.array(im);assert im.size==size and a.ndim==2 and set(np.unique(a))=={0,255}
  t=transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True)(transforms.ToTensor()(im))
  orig=(t.numpy()[0]>=.5);near=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0
 assert orig.any() and near.any()
 return orig,near,size
def metrics(labels,scores,maps,gt):
 maps=np.asarray(maps);gt=np.asarray(gt,bool);assert maps.shape==gt.shape and np.isfinite(maps).all()
 def binary(y,s):
  return {'status':'completed','auroc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s))} if len(np.unique(y))==2 else {'status':'not_applicable','auroc':None,'ap':None}
 region={'status':'not_applicable','aupro_0_30':None}
 if gt.any() and (~gt).any():
  v,n,_=common.aupro(maps,gt);region={'status':'completed','aupro_0_30':v,'regions':n}
 return {'image':binary(labels,scores),'pixel':binary(gt.ravel(),maps.ravel()),'region':region}
def run(a):
 out=a.output.resolve();assert out.is_relative_to(R);out.mkdir(parents=True,exist_ok=False)
 dump(out/'status.json',{'status':'running'});start=time.perf_counter()
 try:
  torch.set_num_threads(2);torch.manual_seed(111);np.random.seed(111);torch.cuda.manual_seed_all(111)
  torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
  entry=read(TRAIN/'completed.json')[a.category];cp=Path(entry['checkpoint']);assert sha(cp)==entry['sha256']
  cfg=read(cp.parent.parent/'config.json');assert cfg['steps']==5000 and cfg['student_steps']==1000 and cfg['selection']=='fixed_final' and not cfg['smoke'] and cfg['category']==a.category and cfg['seed']==111
  assert sha(cp.parent.parent/'config.json')==entry['config_sha256'] and entry['teacher_unchanged'] and entry['test_images_read']==0
  teacher=B/'models/destseg/torch_cache/hub/checkpoints/resnet18-5c106cde.pth';teacher_before=sha(teacher)
  assert teacher_before=='5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8'
  model=DeSTSeg(dest=True,ed=True).eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
  before=state_hash(model);validation={}
  if a.mode=='dry':
   p=Path(entry['sample']);assert sha(p)==entry['sample_sha256']
   with torch.no_grad():am,sc=score(model,preprocess(p)[None])
   with np.load(cp.parent.parent/'normal_export.npz') as z:
    assert np.array_equal(am.numpy(),z['anomaly_map']);assert np.allclose(sc.numpy(),z['image_score'],rtol=0,atol=1e-7)
   validation['saved_normal_map_bitwise_equal']=True
   # Non-square synthetic positive mask verifies resize orientation and exact GT policy.
   im=np.zeros((40,80,3),np.uint8);mask=np.zeros((40,80),np.uint8);mask[4:20,50:75]=255
   Image.fromarray(im).save(out/'fixture.png');Image.fromarray(mask).save(out/'fixture_mask.png')
   orig,near,_=masks(out/'fixture.png',out/'fixture_mask.png');assert near[:128,160:].any() and not near[:, :128].any()
   g=np.zeros((2,8,8),bool);g[1,2:5,3:7]=1
   assert metrics([0,1],[0,1],g.astype(float),g)['region']['aupro_0_30']==1
   assert abs(metrics([0,1],[1,0],1-g.astype(float),g)['region']['aupro_0_30'])<1e-12
   validation.update({'asymmetric_alignment':True,'metric_perfect_inverted':True})
  model.cuda().eval();assert torch.cuda.is_available()
  data=ROOT/'archive'/a.category
  images=sorted((data/('train/good' if a.mode=='dry' else 'test')).glob('*.png' if a.mode=='dry' else '*/*.png'))
  if a.mode=='dry':images=images[:2]
  assert images
  if a.mode=='benchmark':
   train_hash={sha(p) for p in (data/'train/good').glob('*.png')};assert train_hash and not train_hash.intersection(sha(p) for p in images)
   paired={data/'ground_truth'/p.parent.name/(p.stem+'_mask.png') for p in images if p.parent.name!='good'}
   assert paired==set((data/'ground_truth').glob('*/*.png'))
  rows=[];maps=[];ogts=[];ngts=[];labels=[];scores=[];peak_ram=0
  with torch.no_grad():
   x=preprocess(images[0])[None].cuda()
   for _ in range(3):score(model,x)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   for i,p in enumerate(images):
    input_hash=sha(p);torch.cuda.synchronize();t=time.perf_counter();x=preprocess(p)[None].cuda();torch.cuda.synchronize();tm=time.perf_counter()
    am,sc=score(model,x);torch.cuda.synchronize();ms=(time.perf_counter()-tm)*1000
    amap=am.cpu().numpy()[0,0];s=float(sc.item());assert amap.shape==(256,256) and np.isfinite(amap).all() and 0<=s<=1
    pred=out/f'{i:04d}_prediction.npz';np.savez_compressed(pred,anomaly_map=amap,image_score=s)
    pipeline=(time.perf_counter()-t)*1000
    mask=None if p.parent.name=='good' else data/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
    orig,near,size=masks(p,mask);mf=out/f'{i:04d}_masks.npz';np.savez_compressed(mf,original=orig,common_nearest=near)
    rows.append({'id':p.relative_to(ROOT/'archive').as_posix(),'image_path':str(p),'image_sha256':input_hash,'mask_path':str(mask) if mask else None,'mask_sha256':sha(mask) if mask else None,'native_size_wh':size,'image_label':int(mask is not None),'image_score':s,'prediction':pred.name,'prediction_sha256':sha(pred),'masks':mf.name,'masks_sha256':sha(mf),'model_ms':ms,'pipeline_ms':pipeline})
    maps.append(amap);ogts.append(orig);ngts.append(near);labels.append(int(mask is not None));scores.append(s)
    peak_ram=max(peak_ram,psutil.Process().memory_info().rss)
    if i%25==0:dump(out/'progress.json',{'images_done':i+1,'images_total':len(images),'phase':'inference'})
  dump(out/'per_image.json',rows);dump(out/'progress.json',{'images_done':len(images),'phase':'metrics'})
  result={'method':'DeSTSeg','dataset':'MVTec_AD','class':a.category,'seed':111,'track':'T1','mode':a.mode,'benchmark_result':a.mode=='benchmark','positive_images':sum(labels),'negative_images':len(labels)-sum(labels),'mask_count':sum(labels),'common256':metrics(labels,scores,maps,ngts),'original_mask':metrics(labels,scores,maps,ogts),'checkpoint_sha256':entry['sha256'],'definitions':{'image_score':'mean largest100 of bilinear256 sigmoid segmentation map','original_mask':'tensor bilinear antialias>=0.5','common_mask':'PIL nearest>0','aupro':'common exact8-connected equal-region normalized FPR0.30'}}
  assert state_hash(model)==before and sha(cp)==entry['sha256'] and sha(teacher)==teacher_before
  assert all(sha(row['image_path'])==row['image_sha256'] for row in rows)
  summarize=lambda k:{'n':len(rows),'median_ms':float(np.median([z[k] for z in rows])),'p95_ms':float(np.percentile([z[k] for z in rows],95))}
  resources={'model_only':summarize('model_ms'),'pipeline_decode_to_export':summarize('pipeline_ms'),'scope':'batch1 FP32;3warmups; synchronized model includes interpolation/top100; pipeline includes decode through compressed prediction export; excludes GT/metrics/hashes','gpu':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'sampled_peak_ram_bytes':max(peak_ram,psutil.Process().memory_info().rss),'wall_seconds':time.perf_counter()-start}
  dump(out/'metrics.json',result);dump(out/'resource_usage.json',resources);dump(out/'config.json',{'evaluation':vars(a),'training_config':cfg,'calibration':'none','optimizer_steps':0})
  sources=[Path(__file__),ROOT/'compare_dfr_rd4ad_common.py']+list((TRAIN/'source').rglob('*.py'))
  dump(out/'provenance.json',{'commit':cfg['commit'],'checkpoint':str(cp),'checkpoint_sha256_before':entry['sha256'],'checkpoint_sha256_after':sha(cp),'state_unchanged':True,'teacher_sha256':teacher_before,'sources':{str(p):sha(p) for p in sources},'manifest_sha256':sha(out/'per_image.json'),'test_images_read':len(rows) if a.mode=='benchmark' else 0,'validation':validation})
  dump(out/'status.json',{'status':'completed','mode':a.mode,'training_started':False});print('completed',a.category,a.mode,flush=True)
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');dump(out/'status.json',{'status':'failed'});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--category',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['dry','benchmark'],required=True);run(p.parse_args())
