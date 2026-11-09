"""RD++ official modules with train-only, fixed-final orchestration."""
import os
os.environ["OMP_NUM_THREADS"]="4"
os.environ["MKL_NUM_THREADS"]="4"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
from pathlib import Path
import argparse,json,sys,time,hashlib,random,traceback,threading,subprocess
from datetime import datetime,timezone
from importlib import metadata
import numpy as np
import torch
import psutil
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_train import MultiProjectionLayer,Revisit_RDLoss,loss_fucntion
from dataset.dataset import MVTecDataset_train,get_data_transforms

SOURCE=Path(__file__).resolve().parent
BENCH=SOURCE.parents[2]
ROOT=BENCH.parent.parent
EPOCHS={"carpet":10,"grid":260,"leather":10,"tile":260,"wood":100,"bottle":200,"cable":240,"capsule":300,"hazelnut":160,"metal_nut":160,"pill":200,"screw":280,"toothbrush":280,"transistor":300,"zipper":300}
TEACHER_SHA="95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
def state_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def cpu_state(module):return {k:v.detach().cpu() for k,v in module.state_dict().items()}
def seed_all():
    random.seed(111);np.random.seed(111);torch.manual_seed(111);torch.cuda.manual_seed_all(111)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
class Monitor:
    def __init__(self):
        self.stop=threading.Event();self.rows=[];self.peak_ram=0
        self.thread=threading.Thread(target=self.poll,daemon=True)
    def poll(self):
        process=psutil.Process()
        while not self.stop.is_set():
            self.peak_ram=max(self.peak_ram,process.memory_info().rss)
            try:
                used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],creationflags=subprocess.CREATE_NO_WINDOW,text=True,timeout=4)
                self.rows.append({"time":time.time(),"device_used_mib":float(used.strip().splitlines()[0])})
            except Exception:pass
            self.stop.wait(0.5)
    def start(self):self.thread.start()
    def finish(self):self.stop.set();self.thread.join(timeout=5)
def payload(mods,opts,cat,epoch,steps,mode,teacher):
    return {**{name:cpu_state(mod) for name,mod in mods.items()},
        "optimizer_proj":opts[0].state_dict(),"optimizer_distill":opts[1].state_dict(),
        "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()},
        "gradients":{name:{k:p.grad.detach().cpu() for k,p in mod.named_parameters() if p.grad is not None} for name,mod in mods.items()},
        "category":cat,"epoch":epoch,"optimizer_steps":steps,"mode":mode,"seed":111,"epoch_schedule":EPOCHS,
        "teacher_file":str(teacher),"teacher_sha256":TEACHER_SHA,"selection":"fixed_final" if mode=="train" else "bounded_smoke_not_benchmark",
        "source_commit":"7f2ceb7c87e602617b8600e1a498f7ef7f5247d6"}
def save_atomic(path,data):
    tmp=path.with_suffix(".tmp");torch.save(data,tmp);tmp.replace(path)
def train_category(args,cat):
    out=args.output/cat
    out.mkdir(exist_ok=False)
    logf=out/"run.log"
    def log(s):
        s=datetime.now(timezone.utc).isoformat()+" "+s;print(s,flush=True)
        with logf.open("a",encoding="utf-8") as f:f.write(s+"\n")
    monitor=Monitor();start=time.perf_counter();trace=[];steps=0;seen=[];status="failed"
    teacher=SOURCE.parent/"torch_cache/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
    checkpoint=None
    try:
        assert torch.cuda.is_available(),"CUDA required"
        assert sha(teacher)==TEACHER_SHA
        torch.hub.set_dir(str(teacher.parent.parent))
        # Dataset reader remains official. Only train/good is constructed.
        transform,_=get_data_transforms(256,256)
        dataset=MVTecDataset_train(str(args.data_root/cat/"train"),transform)
        assert len(dataset)>0
        paths=[Path(p).resolve() for p in dataset.img_paths]
        assert len(paths)==len(set(paths))
        assert all(p.parent.name=="good" and p.parent.parent.name=="train" for p in paths)
        manifest=[{"path":str(p),"sha256":sha(p)} for p in paths]
        dump(out/"train_manifest.json",manifest)
        loader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True,num_workers=0,drop_last=False)
        config={"mode":args.mode,"category":cat,"seed":111,"epochs":EPOCHS[cat],"batch_size":16,"input_size":256,
            "accumulation_steps":2,"loss_division":False,"proj_lr":.001,"distill_lr":.005,"betas":[.5,.999],
            "weight_proj":.2,"teacher_autograd":"upstream_enabled_not_optimized","precision":"float32","amp":False,
            "smoke_minibatches":6 if args.mode=="smoke" else None,"data_root":str(args.data_root),
            "test_loader_created":False,"checkpoint_selection":"fixed_final",
            "cudnn_deterministic":True,"cudnn_benchmark":False,
            "matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_tf32":torch.backends.cudnn.allow_tf32}
        dump(out/"config.json",config)
        os.environ["MPLCONFIGDIR"]=str(out/"mpl_cache")
        os.environ["NUMBA_CACHE_DIR"]=str(out/"numba_cache")
        encoder,bn=wide_resnet50_2(pretrained=True)
        encoder=encoder.cuda();bn=bn.cuda();encoder.eval()
        decoder=de_wide_resnet50_2(pretrained=False).cuda()
        proj=MultiProjectionLayer(base=64).cuda();criterion=Revisit_RDLoss()
        mods={"proj":proj,"bn":bn,"decoder":decoder}
        opts=[torch.optim.Adam(proj.parameters(),lr=.001,betas=(.5,.999)),
              torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()),lr=.005,betas=(.5,.999))]
        initial={k:state_hash(v) for k,v in mods.items()}
        teacher_before=state_hash(encoder)
        torch.cuda.reset_peak_memory_stats();monitor.start()
        log("START "+cat+" "+args.mode+" batch16; no evaluation dataset constructed")
        batch_count=0
        for epoch in range(1,EPOCHS[cat]+1):
            for mod in mods.values():mod.train()
            for i,(img,img_noise,ids) in enumerate(loader):
                torch.cuda.synchronize();t=time.perf_counter()
                img=img.cuda();img_noise=img_noise.cuda()
                inputs=encoder(img);inputs_noise=encoder(img_noise)
                feature_noise,feature=proj(inputs,features_noise=inputs_noise)
                lp=criterion(inputs_noise,feature_noise,feature)
                outputs=decoder(bn(feature))
                ld=loss_fucntion(inputs,outputs)
                loss=ld+.2*lp
                assert torch.isfinite(loss) and torch.isfinite(lp) and torch.isfinite(ld),"nonfinite loss"
                loss.backward()
                norms={}
                for name,mod in mods.items():
                    grads=[p.grad for p in mod.parameters() if p.grad is not None]
                    assert grads and all(torch.isfinite(g).all() for g in grads),"nonfinite/missing grads"
                    norms[name]=float(sum(g.detach().float().norm() for g in grads).item())
                if (i+1)%2==0:
                    opts[0].step();opts[1].step();steps+=1
                    # torch1.10 default was set_to_none=False; retain original semantics.
                    opts[0].zero_grad(set_to_none=False);opts[1].zero_grad(set_to_none=False)
                torch.cuda.synchronize()
                batch_count+=1;seen.extend(str(p) for p in ids)
                row={"epoch":epoch,"batch":i+1,"minibatch_count":batch_count,"optimizer_steps":steps,
                     "batch_size":len(img),"loss":float(loss.item()),"projection_loss":float(lp.item()),
                     "distill_loss":float(ld.item()),"gradient_norms":norms,"step_seconds":time.perf_counter()-t,
                     "allocated_bytes":torch.cuda.memory_allocated(),"peak_allocated_bytes":torch.cuda.max_memory_allocated()}
                trace.append(row)
                with (out/"loss_trace.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
                log(f"batch {batch_count}, updates {steps}, loss {row['loss']:.6f}, peak {row['peak_allocated_bytes']/2**30:.3f} GiB")
                if args.mode=="smoke" and batch_count>=6:break
                if args.mode=="smoke" and time.perf_counter()-start>900:raise RuntimeError("smoke time budget exceeded")
            if args.mode=="smoke":break
            # Last unpaired gradients are retained, as in upstream; recovery carries them.
            save_atomic(out/"recovery.pt",payload(mods,opts,cat,epoch,steps,args.mode,teacher))
        train_peak_allocated=torch.cuda.max_memory_allocated()
        train_peak_reserved=torch.cuda.max_memory_reserved()
        after={k:state_hash(v) for k,v in mods.items()}
        assert all(after[k]!=initial[k] for k in mods),"parameters did not change"
        assert state_hash(encoder)==teacher_before,"teacher weights/buffers changed"
        if args.mode=="smoke":assert batch_count==6 and steps==3
        checkpoint=out/("smoke_final.pt" if args.mode=="smoke" else "final.pt")
        assert not checkpoint.exists()
        save_atomic(checkpoint,payload(mods,opts,cat,epoch,steps,args.mode,teacher))
        # No test images: verify reload on first training image, without synthetic noise.
        from utils.utils_test import cal_anomaly_map
        from scipy.ndimage import gaussian_filter
        import cv2
        normal=cv2.cvtColor(cv2.imread(str(paths[0])),cv2.COLOR_BGR2RGB)
        normal=transform(cv2.resize(normal/255.,(256,256))).unsqueeze(0).cuda()
        for mod in mods.values():mod.eval()
        def score():
            with torch.no_grad():
                f=encoder(normal);recon=decoder(bn(proj(f)))
                amap,_=cal_anomaly_map(f,recon,256,amap_mode="a")
                return gaussian_filter(amap,sigma=4)
        before_map=score()
        # Perturb a trained parameter to ensure the subsequent reload really restores it.
        with torch.no_grad():next(proj.parameters()).add_(1.0)
        ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
        for key,mod in mods.items():mod.load_state_dict(ck[key],strict=True)
        del ck
        after_map=score()
        assert before_map.shape==(256,256) and np.isfinite(before_map).all()
        assert np.array_equal(before_map,after_map),"checkpoint reload changed score/map"
        assert sha(teacher)==TEACHER_SHA
        np.savez_compressed(out/"reload_training_prediction.npz",map=after_map,image_score=np.max(after_map))
        dump(out/"verification.json",{"passed":True,"trained_modules_changed":{k:after[k]!=initial[k] for k in mods},
            "teacher_state_unchanged":True,"teacher_sha256":TEACHER_SHA,"checkpoint_sha256":sha(checkpoint),
            "reload_bitwise_map_equal":True,"sample":str(paths[0]),"test_images_read":0,"test_loader_created":False,
            "observed_training_samples":seen,"optimizer_steps":steps,"minibatches":batch_count})
        status="completed"
        log("COMPLETE bounded smoke" if args.mode=="smoke" else "COMPLETE fixed-final category")
    except Exception:
        dump(out/"failure.json",{"traceback":traceback.format_exc(),"optimizer_steps":steps})
        log("FAILED\n"+traceback.format_exc())
        raise
    finally:
        if monitor.thread.ident is not None:monitor.finish()
        resources={"status":status,"elapsed_seconds":time.perf_counter()-start,
             "training_peak_allocated_bytes":locals().get("train_peak_allocated",torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
             "training_peak_reserved_bytes":locals().get("train_peak_reserved",torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None),
             "process_ram_peak_sampled_bytes":monitor.peak_ram,"device_used_peak_sampled_mib":max((r["device_used_mib"] for r in monitor.rows),default=None),
             "device_scope":"total device usage, includes other processes; sampled0.5s","optimizer_steps":steps,"full_training":args.mode=="train"}
        dump(out/"resource_usage.json",resources);dump(out/"gpu_samples.json",monitor.rows)
        dump(out/"metrics.json",{"method":"RD++","dataset":"MVTec AD","class":cat,"track":"T1_normal_only_smoke" if args.mode=="smoke" else "T1",
             "seed":111,"status":status,"image":{"status":"not_run","ap":None,"auroc":None},
             "pixel":{"status":"not_run","ap":None,"auroc":None},"region":{"status":"not_run","aupro":None},
             "reason":"training-only execution; no test data or benchmark performance measured"})
        dump(out/"status.json",{"status":status,"mode":args.mode,"checkpoint":str(checkpoint) if checkpoint else None,"optimizer_steps":steps})
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["smoke","train"],default="smoke")
    p.add_argument("--classes",nargs="+",choices=list(EPOCHS),default=["carpet"])
    p.add_argument("--data-root",type=Path,default=ROOT/"archive")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allow-full-training",action="store_true")
    a=p.parse_args()
    if a.mode=="train" and not a.allow_full_training:p.error("Full training requires explicit --allow-full-training; current task authorizes smoke only.")
    if a.mode=="smoke" and a.classes!=["carpet"]:p.error("Registered smoke is carpet only")
    a.output=a.output.resolve();a.data_root=a.data_root.resolve()
    if not a.output.is_relative_to(BENCH/"experiments"):p.error("Outputs must stay under benchmark experiments")
    a.output.mkdir(parents=True,exist_ok=True)
    # Reject any attempted opening of test/ground_truth images in this process.
    def guard(event,arguments):
        if event=="open" and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(arguments[0]).replace("\\","/").lower()
            if str(a.data_root).replace("\\","/").lower() in path and ("/test/" in path or "/ground_truth/" in path):
                raise RuntimeError("Training process attempted held-out data access")
    sys.addaudithook(guard)
    seed_all()
    dump(a.output/"environment.json",{"python":sys.version,"prefix":sys.prefix,"base_prefix":sys.base_prefix,
         "versions":{n:metadata.version(n) for n in ("torch","torchvision","numpy","geomloss","numba","scipy","scikit-learn","Pillow","opencv-python")},
         "gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,"driver_command":"nvidia-smi",
         "script_sha256":sha(__file__)})
    for cat in a.classes:train_category(a,cat)
if __name__=="__main__":main()
