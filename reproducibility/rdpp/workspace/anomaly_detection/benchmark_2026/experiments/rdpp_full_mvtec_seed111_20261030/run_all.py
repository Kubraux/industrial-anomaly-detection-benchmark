import os,sys,json,time,random,shutil,traceback,msvcrt,gc
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;B=R.parents[1];ROOT=B.parent.parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(R/'mpl_cache');os.environ['NUMBA_CACHE_DIR']=str(R/'numba_cache')
sys.path.insert(0,str(R/'source'))
import main as lib
import numpy as np
import torch
from PIL import Image
lib.SOURCE=B/'models/rdpp/source';lib.BENCH=B;lib.ROOT=ROOT
START=time.time()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8');tmp.replace(p)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])
def main():
    lock=(R/'process.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    os.environ['RDPP_START_TIME']=str(START)
    # Keep Windows awake during this explicitly requested persistent compute job.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        lib.seed_all()
        done=read(R/'completed.json') if (R/'completed.json').exists() else {}
        dump(R/'process.json',{'pid':os.getpid(),'started':START,'python':sys.executable,'status':'running'})
        teacher=lib.SOURCE.parent/'torch_cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth'
        assert lib.sha(teacher)==lib.TEACHER_SHA
        expected=read(B/'experiments/rdpp_training_smoke_20261030/training_inventory_and_estimate.json')['categories']
        inventory={}
        for info in expected:
            cat=info['category'];folder=ROOT/'archive'/cat/'train'
            assert {p.name for p in folder.iterdir() if p.is_dir()}=={'good'}
            files=list((folder/'good').glob('*.png'));assert len(files)==info['train_images']
            rows=[]
            for p in files:
                with Image.open(p) as im:im.verify()
                rows.append({'path':str(p),'sha256':lib.sha(p)})
            inventory[cat]=rows
        assert sum(map(len,inventory.values()))==3629
        if (R/'train_inventory.json').exists():assert read(R/'train_inventory.json')==inventory
        else:dump(R/'train_inventory.json',inventory)
        dump(R/'environment.json',{'python':sys.version,'prefix':sys.prefix,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'source_hashes':{str(p.relative_to(R)):lib.sha(p) for p in (R/'source').rglob('*.py')}})
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                s=os.fsdecode(args[0]).replace('\\','/').lower()
                if '/archive/' in s and ('/test/' in s or '/ground_truth/' in s):raise RuntimeError('held-out access forbidden')
        sys.addaudithook(guard)
        order=list(lib.EPOCHS)
        assert list(done)==order[:len(done)],'completed categories must form the original prefix'
        for cat in lib.EPOCHS:
            if cat in done:
                info=done[cat];checkpoint=Path(info['checkpoint'])
                assert read(checkpoint.parent/'status.json')['status']=='completed'
                assert lib.sha(checkpoint)==info['sha256']
                ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert ck['category']==cat and ck['selection']=='fixed_final' and ck['epoch']==lib.EPOCHS[cat] and ck['mode']=='train'
                del ck
                boundary=Path(info['rng_boundary']);assert lib.sha(boundary)==info['rng_sha256']
                restore(torch.load(boundary,map_location='cpu',weights_only=False))
                print('SKIP verified completed '+cat,flush=True);continue
            assert time.time()-START<48*3600,'wall budget exceeded'
            assert shutil.disk_usage(R).free>4*2**30,'insufficient disk headroom'
            parent=R/'attempts'/cat;parent.mkdir(parents=True,exist_ok=True)
            attempt=parent/f'attempt_{len(list(parent.glob("attempt_*")))+1:03d}';attempt.mkdir()
            torch.save(rng(),attempt/'entry_rng.pt')
            dump(R/'progress.json',{'status':'training','pid':os.getpid(),'current_category':cat,'epoch_limit':lib.EPOCHS[cat],'completed_categories':list(done),'total_categories':15,'active_directory':str(attempt/cat),'updated':time.time()})
            print('START '+cat+' '+str(attempt),flush=True)
            lib.train_category(SimpleNamespace(mode='train',output=attempt,data_root=ROOT/'archive'),cat)
            final=attempt/cat/'final.pt'
            assert read(attempt/cat/'status.json')['status']=='completed'
            boundary=attempt/'next_category_rng.pt';torch.save(rng(),boundary)
            done[cat]={'checkpoint':str(final),'sha256':lib.sha(final),'rng_boundary':str(boundary),'rng_sha256':lib.sha(boundary),'epoch':lib.EPOCHS[cat]}
            dump(R/'completed.json',done)
            print('COMPLETED '+cat,flush=True);gc.collect();torch.cuda.empty_cache()
        dump(R/'progress.json',{'status':'completed','pid':os.getpid(),'completed_categories':list(done),'total_categories':15,'updated':time.time()})
    except Exception:
        (R/f'failure_{int(time.time())}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        dump(R/'progress.json',{'status':'failed','pid':os.getpid(),'completed_categories':list(locals().get('done',{})),'traceback':traceback.format_exc()});raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()
if __name__=='__main__':main()
