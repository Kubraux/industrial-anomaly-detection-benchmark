"""Controlled discriminator reset audit. Only Adam state reset policy changes."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import torch
import run_mvtec_benchmark as bench
from run_capsule_ablations import DiagnosticGanomaly

class ResetAdamGanomaly(DiagnosticGanomaly):
    def reinit_d(self):
        super().reinit_d()
        # New random D weights must not inherit moments from the old weights.
        # This is an experimental policy, not an asserted upstream bug fix.
        self.optimizer_d.state.clear()
    def get_errors(self):
        values=super().get_errors()
        values['d_adam_state_entries']=len(self.optimizer_d.state)
        return values

def worker(args):
    original_options=bench.make_options
    def options(a,category):
        opt=original_options(a,category)
        opt.stable_bce=False
        return opt
    bench.make_options=options
    bench.Ganomaly=ResetAdamGanomaly
    bench.worker(args)

def controller(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__).resolve(),bench.BASE/'run_mvtec_benchmark.py',
              bench.BASE/'run_capsule_ablations.py',bench.BASE/'options.py',
              *sorted((bench.BASE/'lib').glob('*.py'))]:
        dst=out/'source_snapshot'/p.relative_to(bench.BASE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
    bench.save_json(out/'protocol.json',dict(
        categories=['capsule','carpet'],epochs=25,seed=0,image_size=256,nz=100,batch_size=8,
        lr=0.0002,change='clear optimizer_d.state after existing reinit_d',
        unchanged='same initial seed, data order policy, losses, BN behavior, D reset trigger (<1e-5), final epoch selection',
        hypothesis='Repeated random D weights plus stale Adam moments destabilize training',
        interpretation='exploratory paired comparison to existing seed=0 baselines; not independent validation',
        selection='final epoch only; no test-selected checkpoint',
        started_utc=bench.stamp()))
    baseline=Path(args.baseline).resolve()
    rows=[]
    for cat in ['capsule','carpet']:
        row=json.loads((baseline/cat/'metrics.json').read_text())
        row['variant']='baseline_keep_adam'
        rows.append(row)
    bench.save_csv(out/'results.csv',rows)
    for i,cat in enumerate(['capsule','carpet']):
        bench.save_json(out/'status.json',dict(status='running',category=cat,completed=i,total=2,pid=os.getpid(),updated_utc=bench.stamp()))
        cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker','--category',cat,
             '--data',str(Path(args.data).resolve()),'--output',str(out),'--baseline',str(baseline)]
        with (out/(cat+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=bench.BASE,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            bench.save_json(out/'status.json',dict(status='failed',category=cat,completed=i,exit_code=result.returncode))
            raise RuntimeError('Failed '+cat)
        row=json.loads((out/cat/'metrics.json').read_text())
        row['variant']='reset_adam_on_d_reinit'
        rows.append(row)
        bench.save_csv(out/'results.csv',rows)
    bench.save_json(out/'status.json',dict(status='complete',completed=2,total=2,updated_utc=bench.stamp()))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--category',choices=['capsule','carpet'])
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--baseline',required=True)
    args=p.parse_args()
    args.epochs=25
    worker(args) if args.worker else controller(args)

