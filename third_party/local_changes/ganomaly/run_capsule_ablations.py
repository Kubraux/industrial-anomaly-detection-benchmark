"""Exploratory Capsule ablations; fixed final epoch, no automatic winner selection."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import torch
import run_mvtec_benchmark as bench
from lib.model import Ganomaly

VARIANTS = {
    'size64_nz100': dict(image_size=64, nz=100, stable_bce=False),
    'size256_nz256': dict(image_size=256, nz=256, stable_bce=False),
    'size256_nz100_stable_bce': dict(image_size=256, nz=100, stable_bce=True),
}
class DiagnosticGanomaly(Ganomaly):
    def __init__(self, opt, loaders):
        super().__init__(opt, loaders)
        self.reinitializations = 0
        if opt.stable_bce:
            self.netd.classifier.Sigmoid = torch.nn.Identity()
            self.l_bce = torch.nn.BCEWithLogitsLoss()
    def reinit_d(self):
        self.reinitializations += 1
        super().reinit_d()
    def get_errors(self):
        values = super().get_errors()
        values['d_loss_real'] = self.err_d_real.item()
        values['d_loss_fake'] = self.err_d_fake.item()
        values['d_reinitializations_cumulative'] = self.reinitializations
        for name,raw in [('real',self.pred_real),('fake',self.pred_fake)]:
            p = raw.detach().sigmoid() if self.opt.stable_bce else raw.detach()
            values['d_'+name+'_prob_mean'] = p.mean().item()
            values['d_'+name+'_near_zero_fraction'] = (p<1e-6).float().mean().item()
            values['d_'+name+'_near_one_fraction'] = (p>1-1e-6).float().mean().item()
        return values

def worker(args):
    cfg = VARIANTS[args.variant]
    original_options = bench.make_options
    original_save = bench.save_json
    def options(a,category):
        opt = original_options(a,category)
        opt.name = args.variant
        opt.isize = cfg['image_size']
        opt.nz = cfg['nz']
        opt.stable_bce = cfg['stable_bce']
        return opt
    def save(path, obj):
        if Path(path).name == 'metrics.json':
            obj.update(variant=args.variant, **cfg)
        original_save(path,obj)
    bench.make_options = options
    bench.save_json = save
    bench.Ganomaly = DiagnosticGanomaly
    args.category = 'capsule'
    bench.worker(args)

def controller(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__).resolve(),bench.BASE/'run_mvtec_benchmark.py',
              bench.BASE/'options.py',*sorted((bench.BASE/'lib').glob('*.py'))]:
        dst=out/'source_snapshot'/p.relative_to(bench.BASE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
    baseline=Path(args.baseline).resolve()
    old=json.loads(baseline.read_text(encoding='utf-8-sig'))
    if old['epochs']!=25 or old['seed']!=0 or old['image_size']!=256:
        raise RuntimeError('Baseline settings mismatch')
    shutil.copy2(baseline,out/'baseline_metrics.json')
    bench.save_json(out/'protocol.json',dict(category='capsule',epochs=25,seed=0,batch=8,
        variants=VARIANTS,baseline=str(baseline),lr=0.0002,
        score='latent mean squared difference; final epoch only',
        evaluation='same existing Capsule test split; EXPLORATORY, not independent tuning validation',
        controls='one factor changed from baseline per variant; size also changes network capacity',
        stable_bce='remove classifier sigmoid, use BCEWithLogitsLoss; same adversarial objective',
        diagnostics='epoch sample-weighted means; cumulative reset count is averaged over batches',
        threshold='train normal score q99; test F1-max explicitly oracle',
        started_utc=bench.stamp()))
    old.update(variant='baseline_size256_nz100',nz=100,stable_bce=False)
    results=[old]
    bench.save_csv(out/'results.csv',results)
    for i,variant in enumerate(VARIANTS):
        bench.save_json(out/'status.json',dict(status='running',variant=variant,completed=i,
            total=3,pid=os.getpid(),updated_utc=bench.stamp()))
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',
            '--variant',variant,'--data',str(Path(args.data).resolve()),'--output',str(out),
            '--baseline',str(baseline)]
        with (out/(variant+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=bench.BASE,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            bench.save_json(out/'status.json',dict(status='failed',variant=variant,
                completed=i,exit_code=process.returncode,updated_utc=bench.stamp()))
            raise RuntimeError('Failed variant: '+variant)
        result=json.loads((out/variant/'metrics.json').read_text())
        # Preserve a consistent CSV column order shared with baseline.
        result={k:result[k] for k in results[0]}
        results.append(result)
        bench.save_csv(out/'results.csv',results)
    bench.save_json(out/'status.json',dict(status='complete',completed=3,total=3,updated_utc=bench.stamp()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--baseline',required=True)
    args=p.parse_args()
    args.epochs=25
    worker(args) if args.worker else controller(args)

