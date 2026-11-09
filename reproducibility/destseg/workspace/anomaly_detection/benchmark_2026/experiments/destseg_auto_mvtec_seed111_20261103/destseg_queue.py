"""Persistent RD++ -> DeSTSeg smoke -> fixed-final15 queue. No test evaluation."""
import os,sys,time,json,hashlib,subprocess,traceback,shutil,argparse,ctypes,msvcrt
from pathlib import Path
import psutil
R=Path(__file__).resolve().parent;B=R.parents[1];ROOT=B.parent.parent
RD=B/'experiments/rdpp_full_mvtec_seed111_20261030'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');os.replace(tmp,p)
def status(s,**kw):
 dump(R/'progress.json',{'status':s,'queue_pid':os.getpid(),'updated':time.time(),**kw});print(s+' '+json.dumps(kw),flush=True)
def decision(state,done,alive):
 if state.get('status')=='failed':return 'failed'
 if state.get('status')=='completed' and len(done)==15 and not alive:return 'ready'
 return 'waiting'
def rd_state():
 state=read(RD/'progress.json');done=read(RD/'completed.json') if (RD/'completed.json').exists() else {}
 pid=state.get('pid');alive=bool(pid and psutil.pid_exists(pid))
 return decision(state,done,alive),state,done
def validate_manifest():
 for item in read(R/'pinned_manifest.json'):
  assert sha(item['path'])==item['sha256'],'Pinned asset changed: '+item['path']
def validate_completion(out,smoke,cat):
 v=read(out/'completion.json');assert v['status']=='completed' and v['smoke']==smoke and v['category']==cat and v['seed']==111
 assert v['steps']==(6 if smoke else 5000) and v['reload_passed'] and v['teacher_unchanged'] and v['normal_export_passed']
 assert sha(v['checkpoint'])==v['sha256']
 assert sha(out/'checkpoints/training_state.pt')==v['training_state_sha256']
 assert sha(out/'config.json')==v['config_sha256']
 return v
def active_budget():
 return read(R/'budget.json').get('charged_seconds',0) if (R/'budget.json').exists() else 0
def run_attempt(cat,smoke):
 validate_manifest()
 assert rd_state()[0]=='ready','RD++ is no longer safely complete'
 parent=R/('smoke' if smoke else 'categories')/cat;parent.mkdir(parents=True,exist_ok=True)
 attempt=parent/('attempt_%03d'%(len([p for p in parent.glob('attempt_*') if p.is_dir()])+1))
 cmd=[sys.executable,'-B','-u',str(R/'worker.py'),'--category',cat,'--out',str(attempt)]
 if smoke:cmd+=['--smoke']
 budget=active_budget();limit=900 if smoke else read(R/'config.json')['max_seconds']-budget
 assert limit>0,'Full-run time budget exhausted'
 started=time.time();child=None
 with (parent/(attempt.name+'.stdout.log')).open('x',encoding='utf-8') as stdout,(parent/(attempt.name+'.stderr.log')).open('x',encoding='utf-8') as stderr:
  try:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
   dump(R/'active_child.json',{'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'attempt':str(attempt),'started':started})
   status('smoke_running' if smoke else 'training',category=cat,child_pid=child.pid,attempt=str(attempt))
   while child.poll() is None:
    if (R/'STOP').exists():raise RuntimeError('STOP requested')
    if time.time()-started>limit:raise RuntimeError('Attempt/run time budget exceeded')
    if shutil.disk_usage(R).free<2*2**30:raise RuntimeError('Less than2GiB disk free')
    time.sleep(5)
   assert child.returncode==0,'Worker failed: '+str(attempt)
  finally:
   if child and child.poll() is None:child.terminate();child.wait(timeout=30)
   dump(R/'budget.json',{'charged_seconds':budget+time.time()-started,'max_seconds':read(R/'config.json')['max_seconds']})
 return validate_completion(attempt,smoke,cat)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
 if args.dry_run:
  assert decision({'status':'training'}, {},True)=='waiting'
  assert decision({'status':'completed'},dict.fromkeys(range(15)),True)=='waiting'
  assert decision({'status':'completed'},dict.fromkeys(range(14)),False)=='waiting'
  assert decision({'status':'failed'}, {},False)=='failed'
  assert decision({'status':'completed'},dict.fromkeys(range(15)),False)=='ready'
  validate_manifest();d,s,done=rd_state()
  dump(R/'queue_validation.json',{'passed':True,'decision':d,'rdpp_completed':len(done),'gpu_allocated':False,'training_launched':False,'state_transition_cases':5})
  print('Queue dry-run passed; current decision='+d);return
 lock=(R/'queue.lock').open('a+b');lock.seek(0)
 if not lock.read(1):lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 # Do not spawn another worker after a crashed/restarted monitor while its child survives.
 if (R/'active_child.json').exists():
  old=read(R/'active_child.json')
  if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['created'])<1:raise RuntimeError('Prior child still running; refusing duplicate queue')
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 dump(R/'process.json',{'pid':os.getpid(),'started':time.time(),'python':sys.executable})
 try:
  validate_manifest();deadline=time.time()+604800
  while True:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   try:d,state,done=rd_state()
   except (json.JSONDecodeError,PermissionError):time.sleep(5);continue
   if d=='failed':raise RuntimeError('RD++ failed; no downstream launch')
   if d=='ready':break
   assert time.time()<deadline,'RD++ wait deadline exceeded'
   status('waiting_for_rdpp',rdpp_completed=len(done),rdpp_category=state.get('current_category'))
   time.sleep(60)
  for cat,record in done.items():assert sha(record['checkpoint'])==record['sha256'],'RD++ checkpoint checksum mismatch: '+cat
  assert shutil.disk_usage(R).free>=12*2**30,'Need12GiB disk before DeSTSeg'
  if (R/'smoke_success.json').exists():
   sm=read(R/'smoke_success.json');sm=validate_completion(Path(sm['checkpoint']).parents[1],True,'capsule')
  else:
   sm=run_attempt('capsule',True);dump(R/'smoke_success.json',sm)
  estimated=sm['resources']['training_seconds']/6*5000*15*2
  assert estimated<=read(R/'config.json')['max_seconds'],'Pilot extrapolation exceeds registered time budget'
  assert sm['package_bytes']*16+2*2**30<shutil.disk_usage(R).free,'Insufficient estimated package storage'
  dump(R/'gpu_readiness.json',{'passed':True,'smoke_checkpoint_sha256':sm['sha256'],'estimated_seconds_2x_margin':estimated,'measured_smoke':sm['resources'],'test_data_used':False})
  completed=read(R/'completed.json') if (R/'completed.json').exists() else {}
  for cat in read(R/'config.json')['categories']:
   if (R/'STOP').exists():raise RuntimeError('STOP requested')
   if cat in completed:
    validate_completion(Path(completed[cat]['checkpoint']).parents[1],False,cat);print('SKIP verified '+cat,flush=True);continue
   completed[cat]=run_attempt(cat,False);dump(R/'completed.json',completed)
  status('completed',completed_categories=list(completed),test_evaluation='not_run')
 except BaseException:
  failure=traceback.format_exc();(R/('failure_'+time.strftime('%Y%m%d_%H%M%S')+'.txt')).write_text(failure,encoding='utf-8');status('failed',error=failure);raise
 finally:
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.close()
if __name__=='__main__':main()
