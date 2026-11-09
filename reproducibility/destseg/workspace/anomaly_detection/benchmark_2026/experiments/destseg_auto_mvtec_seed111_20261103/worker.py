"""One isolated category attempt, no target test access."""
import os,sys,json,time,hashlib,traceback,argparse
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;B=R.parents[1];ROOT=B.parent.parent
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
os.environ['TORCH_HOME']=str(B/'models/destseg/torch_cache')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--category',required=True);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 out=a.out.resolve();assert out.is_relative_to(R)
 out.mkdir(parents=True,exist_ok=False)
 os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   s=os.fsdecode(args[0]).replace('\\','/').lower()
   if '/archive/' in s and ('/test/' in s or '/ground_truth/' in s):raise RuntimeError('Target held-out data access prohibited')
 sys.addaudithook(guard)
 started=time.perf_counter()
 try:
  import torch,numpy as np,imgaug,random,psutil
  from PIL import Image
  from torchvision import transforms
  sys.path.insert(0,str(R/'source'))
  import train
  from model.destseg import DeSTSeg
  assert torch.cuda.mem_get_info()[0]>=14*2**30,'Less than14GiB free VRAM'
  torch.set_num_threads(2);random.seed(111);np.random.seed(111);imgaug.seed(111);torch.manual_seed(111);torch.cuda.manual_seed_all(111)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  cfg=json.loads((R/'config.json').read_text());assert a.category in cfg['categories']
  steps=6 if a.smoke else 5000;student_steps=3 if a.smoke else 1000
  dump(out/'config.json',{**cfg,'category':a.category,'steps':steps,'student_steps':student_steps,'smoke':a.smoke,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32})
  args=SimpleNamespace(checkpoint_path=str(out/'checkpoints'),log_path=str(out/'logs'),run_name_head='smoke' if a.smoke else 'DeSTSeg',steps=steps,de_st_steps=student_steps,lr_res=.1,lr_seghead=.01,lr_de_st=.4,bs=32,num_workers=0,mvtec_path=str(ROOT/'archive')+'/',dtd_path=str(ROOT/'DRAEM_setup/datasets/dtd/images'),gamma=4,log_per_steps=1 if a.smoke else 50)
  rotate=a.category in ['bottle','grid','hazelnut','leather','tile','carpet','screw'];slight=a.category in ['wood','zipper','cable']
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  train.train(args,a.category,rotate_90=rotate,random_rotate=5 if rotate or slight else 0)
  torch.cuda.synchronize();training_seconds=time.perf_counter()-t
  usage={'training_seconds':training_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name(),'rss_at_training_end':psutil.Process().memory_info().rss,'torch':torch.__version__,'cuda':torch.version.cuda}
  torch.cuda.empty_cache()
  checkpoint=next((out/'checkpoints').glob('*.pckl'));before=sha(checkpoint)
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)
  assert all(torch.isfinite(v).all() for v in state.values()),'Nonfinite checkpoint'
  model=DeSTSeg(dest=True,ed=True).cpu().eval();model.load_state_dict(state,strict=True)
  assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items()),'Reload mismatch'
  image=sorted((ROOT/'archive'/a.category/'train/good').glob('*.png'))[0]
  with Image.open(image) as im:
   x=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))[None]
  with torch.no_grad():
   raw=model(x)[0];amap=torch.nn.functional.interpolate(raw,(256,256),mode='bilinear',align_corners=False);score=amap.flatten(1).topk(100).values.mean(1)
  assert amap.shape==(1,1,256,256) and torch.isfinite(amap).all() and 0<=float(score)<=1
  assert sha(checkpoint)==before,'Checkpoint modified by validation'
  np.savez_compressed(out/'normal_export.npz',anomaly_map=amap.numpy(),image_score=score.numpy())
  rows=[json.loads(line) for line in (out/'logs/loss_trace.jsonl').read_text().splitlines()]
  assert len(rows)==steps and rows[-1]['step']==steps and {v['phase'] for v in rows}=={'student','segmentation'}
  verify=json.loads((out/'logs/verification.json').read_text());assert verify['teacher_unchanged'] and verify['fixed_final_step']==steps
  usage.update({'peak_rss_sampled_bytes':max(v['rss_bytes'] for v in rows),'total_attempt_seconds':time.perf_counter()-started,'inference_latency':'Not recorded'})
  dump(out/'resource_usage.json',usage)
  result={'status':'completed','category':a.category,'smoke':a.smoke,'seed':111,'steps':steps,'checkpoint':str(checkpoint),'sha256':before,'training_state_sha256':sha(out/'checkpoints/training_state.pt'),'config_sha256':sha(out/'config.json'),'teacher_unchanged':True,'finite_gradients_checked':True,'reload_passed':True,'normal_export_passed':True,'test_images_read':0,'sample':str(image),'sample_sha256':sha(image),'package_bytes':sum(p.stat().st_size for p in (out/'checkpoints').iterdir()),'resources':usage}
  dump(out/'completion.json',result);print(json.dumps(result),flush=True)
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
