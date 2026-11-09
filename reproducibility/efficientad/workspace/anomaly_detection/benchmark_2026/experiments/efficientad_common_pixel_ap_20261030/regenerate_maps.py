"""Fixed-final EfficientAD inference. Only common shared_gt and AP functions called."""
import os
os.environ["OMP_NUM_THREADS"]="4"
os.environ["MKL_NUM_THREADS"]="4"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
from pathlib import Path
import sys, json, csv, time, hashlib, random, traceback, importlib.util, platform, gc, ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from importlib import metadata
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[3]
SETUP=ROOT/"EfficientAD_Anomalib_setup_20261023"
DATA=ROOT/"archive"
CFG=json.loads((RUN/"config.yaml").read_text())
assert RUN.parent.parent == ROOT/"anomaly_detection/benchmark_2026"
assert (RUN/"plan.md").is_file()
os.environ["MPLCONFIGDIR"]=str(RUN/"cache/matplotlib")
os.environ["TORCH_HOME"]=str(RUN/"cache/torch")
SOURCES={}; RESULTS=[]; FAILURES=[]
START=time.perf_counter()
def dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
def log(s):
    s=datetime.now(timezone.utc).isoformat()+" "+s
    print(s,flush=True)
    with (RUN/"run.log").open("a",encoding="utf-8") as f:f.write(s+"\n")
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def src(p):
    p=Path(p).resolve()
    if str(p) not in SOURCES:SOURCES[str(p)]={"path":str(p),"bytes":p.stat().st_size,"sha256_before":sha(p)}
    return p
def read(p):return json.loads(src(p).read_text(encoding="utf-8-sig"))
class PMC(ctypes.Structure):
    _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
def peak_ram():
    p=PMC();p.cb=ctypes.sizeof(p)
    k=ctypes.WinDLL("kernel32");k.GetCurrentProcess.restype=wintypes.HANDLE
    lib=ctypes.WinDLL("psapi")
    lib.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
    assert lib.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(p),p.cb)
    return int(p.PeakWorkingSetSize)
def state_hash(model):
    h=hashlib.sha256()
    for key,v in model.state_dict().items():
        h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def worker(size,cat):
    begin=time.perf_counter()
    out=RUN/size/cat
    assert not out.exists(),"Refuse overwrite category"
    out.mkdir(parents=True)
    tr=ROOT/f"EfficientAD_{size.upper()}_Anomalib_training_20261023"/cat
    hist=ROOT/f"EfficientAD_{size.upper()}_Anomalib_results_20261023"/cat
    config=read(tr/"config.json");training=read(tr/"training.json")
    assert config["size"]==size and config["category"]==cat and config["seed"]==42
    assert config["steps"]==70000 and config["input_size"]==256
    assert config["padding"] is False and config["pad_maps"] is True
    assert config["precision"]=="float32" and config["tf32"] is False
    assert config["anomalib"]=={"version":"2.2.0","commit":"592aafc373091dda8206510a0853eb11c8e4099d"}
    split=read(SETUP/"splits"/(cat+".json"))
    assert sha(SETUP/"splits"/(cat+".json"))==config["split_sha256"]
    assert sha(src(config["teacher"]))==config["teacher_sha256"]
    records=read(hist/"test_manifest.json")
    assert [r["path"] for r in records]==sorted(r["path"] for r in records)
    paths=[DATA/r["path"] for r in records]
    assert paths==sorted((DATA/cat/"test").glob("*/*.png"))
    assert len(paths)==len(set(paths))
    for role in ("fit","validation"):
        assert all(Path(r["path"]).parent.as_posix()==cat+"/train/good" for r in split[role])
    fit_hash,val_hash,test_hash=({r["sha256"] for r in group} for group in (split["fit"],split["validation"],records))
    assert not (fit_hash&val_hash or fit_hash&test_hash or val_hash&test_hash)
    checkpoint=src(tr/"final.pt")
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert ck["step"]==70000 and ck["config"]==config
    model=EfficientAdModel(model_size=EfficientAdModelSize.S if size=="s" else EfficientAdModelSize.M,padding=False,pad_maps=True)
    model.load_state_dict(ck["model"],strict=True)
    del ck
    model=model.cuda().eval();model.requires_grad_(False)
    quantiles={k:float(v.detach().cpu()) for k,v in model.quantiles.items()}
    assert quantiles==training["calibration"]
    assert quantiles["qb_st"]>quantiles["qa_st"] and quantiles["qb_ae"]>quantiles["qa_ae"]
    assert torch.isfinite(model.mean_std["mean"]).all()
    assert torch.isfinite(model.mean_std["std"]).all() and (model.mean_std["std"]>0).all()
    initial=state_hash(model)
    names=["/".join(r["path"].split("/")[-2:]) for r in records]
    gt,mask_sources=common.shared_gt(cat,names)
    legacy_common=read(ROOT/"DFR_RD4AD_common_evaluation/RD4AD"/cat/"inputs.json")
    assert names==legacy_common["ordered_test_images"]
    gt_hash=hashlib.sha256(gt.tobytes()).hexdigest()
    assert gt_hash==legacy_common["gt_sha256"],"Common mask identity mismatch"
    for r in mask_sources:src(r["path"])
    warm=src(DATA/split["fit"][0]["path"])
    assert sha(warm)==split["fit"][0]["sha256"]
    with Image.open(warm) as im:x=tf(im.convert("RGB")).unsqueeze(0).cuda()
    with torch.inference_mode():
        for _ in range(CFG["warmup"]):model(x)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    maps=np.empty(gt.shape,np.float32);scores=[];labels=[];manifest=[];timings=[];deltas=[]
    with torch.inference_mode():
        for i,(f,old) in enumerate(zip(paths,records)):
            src(f)
            assert sha(f)==old["sha256"],"Image bytes changed"
            t=time.perf_counter()
            with Image.open(f) as im:
                native_shape=[im.height,im.width]
                rgb=im.convert("RGB")
                x=tf(rgb).unsqueeze(0).cuda()
            torch.cuda.synchronize();pre=time.perf_counter()-t
            assert native_shape==old["shape"]
            label=int(f.parent.name!="good")
            mask_path=None;native_positive=0
            if label:
                mask_path=DATA/cat/"ground_truth"/f.parent.name/(f.stem+"_mask.png")
                assert Path(old["mask"]).resolve()==mask_path.resolve()
                assert sha(src(mask_path))==old["mask_sha256"]
                with Image.open(mask_path) as im:
                    native_mask=np.array(im.convert("L"))
                assert list(native_mask.shape)==native_shape and set(np.unique(native_mask))<= {0,255}
                native_positive=int(np.count_nonzero(native_mask))
                assert native_positive>0 and gt[i].any()
            else:
                assert not old["mask"] and not gt[i].any()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();a.record();prediction=model(x);b.record();torch.cuda.synchronize()
            wall=time.perf_counter()-t;gpu=a.elapsed_time(b)/1000
            t=time.perf_counter()
            amap=prediction.anomaly_map[0,0].cpu().numpy()
            score=float(prediction.pred_score.item())
            assert amap.shape==(256,256) and amap.dtype==np.float32 and np.isfinite(amap).all()
            assert np.isfinite(score) and abs(score-float(amap.max()))<1e-7
            assert np.isclose(score,old["score"],atol=CFG["score_atol"],rtol=CFG["score_rtol"]),"Historical score mismatch"
            maps[i]=amap;post=time.perf_counter()-t
            scores.append(score);labels.append(label);deltas.append(abs(score-old["score"]))
            timings.append([pre*1000,gpu*1000,wall*1000,post*1000])
            manifest.append({"path":old["path"],"sha256":old["sha256"],"label":label,
                 "mask":str(mask_path) if mask_path else None,"mask_sha256":old["mask_sha256"],
                 "native_shape":native_shape,"map_shape":[256,256],"native_positive_pixels":native_positive,
                 "common_positive_pixels":int(gt[i].sum()),"score":score,"historical_score":old["score"],
                 "map_array_index":i})
            if (i+1)%50==0:log(f"{size}/{cat}: inferred {i+1}/{len(paths)}")
    final=state_hash(model);assert initial==final,"Model parameters or calibration changed"
    assert sha(checkpoint)==SOURCES[str(checkpoint)]["sha256_before"]
    gpu_peak=torch.cuda.max_memory_allocated();gpu_reserved=torch.cuda.max_memory_reserved()
    assert gpu_peak<=CFG["gpu_budget_bytes"]
    del model,x,prediction;torch.cuda.empty_cache()
    np.savez_compressed(out/"predictions.npz",maps=maps,masks=gt,scores=np.array(scores),labels=np.array(labels),names=np.array(names))
    dump(out/"test_manifest.json",manifest)
    dump(out/"config.json",{"original_training_config":config,"evaluation_config":CFG,
          "checkpoint":str(checkpoint),"checkpoint_sha256":SOURCES[str(checkpoint)]["sha256_before"],
          "calibration_reused":quantiles})
    dump(out/"verification.json",{"passed":True,"model_state_before":initial,"model_state_after":final,
         "calibration_unchanged":True,"checkpoint_unchanged":True,"common_gt_sha256":gt_hash,
         "matches_existing_common_evaluator_masks":True,"all_native_image_mask_shapes_match":True,
         "input_hashes_match_historical_manifests":True,"split_hashes_disjoint":True,
         "max_historical_image_score_absolute_difference":max(deltas),"score_atol":CFG["score_atol"],"score_rtol":CFG["score_rtol"]})
    ap_start=time.perf_counter()
    pixel_ap=float(common.average_precision_score(gt.ravel(),maps.ravel()))
    ap_seconds=time.perf_counter()-ap_start
    assert 0<=pixel_ap<=1
    tm=np.asarray(timings)
    dump(out/"timings.json",{"columns":["preprocess_ms","model_gpu_ms","model_wall_ms","postprocess_ms"],
         "rows":timings,"warmup":30,"batch":1,"pipeline_scope":"decode/resize/H2D + synchronized forward + D2H; excludes validation hashes, mask reads, serialization/AP"})
    row={"method":"EfficientAD-"+size.upper(),"variant":size,"category":cat,"dataset":"MVTec AD",
         "track":CFG["track"],"seed":42,"status":"completed","test_images":len(paths),
         "positive_images":sum(labels),"negative_images":len(labels)-sum(labels),
         "mask_count":len(paths),"positive_pixels":int(gt.sum()),"negative_pixels":int(gt.size-gt.sum()),
         "pixel_ap":pixel_ap,"image":{"status":"not_run","reason":"AP already available; regenerated scores used for fidelity checks"},
         "pixel":{"status":"completed","ap":pixel_ap,"grid":[256,256],"definition":CFG["pixel_metric"],"mask_policy":CFG["mask_policy"]},
         "region":{"status":"not_run","reason":"AP-specific extension"},
         "max_historical_image_score_difference":max(deltas),
         "calibration_unchanged":True,"common_mask_alignment_passed":True,
         "gpu_peak_allocated_bytes":gpu_peak,"gpu_peak_reserved_bytes":gpu_reserved,
         "process_peak_ram_bytes":peak_ram(),"pixel_ap_seconds":ap_seconds,"elapsed_seconds":time.perf_counter()-begin,
         "model_gpu_median_ms":float(np.median(tm[:,1])),"model_gpu_p95_ms":float(np.percentile(tm[:,1],95)),
         "pipeline_median_ms":float(np.median(tm[:,0]+tm[:,2]+tm[:,3])),
         "checkpoint":str(checkpoint),"checkpoint_sha256":SOURCES[str(checkpoint)]["sha256_before"],
         "prediction_sha256":sha(out/"predictions.npz")}
    assert row["process_peak_ram_bytes"]<=CFG["memory_budget_bytes"]
    dump(out/"metrics.json",row)
    log(f"COMPLETE {size}/{cat}: Pixel AP={pixel_ap:.10f}; max_score_delta={max(deltas):.3g}")
    return row
if (RUN/"metrics.json").exists():raise SystemExit("Refusing overwrite")
log("START fixed-final inference/common256 AP; no training, recalibration, threshold fitting")
try:
    import numpy as np, torch
    from PIL import Image
    from torchvision.transforms.v2 import Compose,ToTensor,Resize
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel,EfficientAdModelSize
    import anomalib.models.image.efficient_ad.torch_model as model_module
    assert torch.cuda.is_available()
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    integrity=read(SETUP/"implementation_integrity.json")
    assert sha(src(model_module.__file__))==integrity["files"]["torch_model.py"]
    spec=importlib.util.spec_from_file_location("historical_common_evaluator",src(ROOT/"compare_dfr_rd4ad_common.py"))
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    assert common.ROOT==ROOT
    tf=Compose([ToTensor(),Resize((256,256),antialias=True)])
    env={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
        "versions":{p:metadata.version(p) for p in ("torch","torchvision","anomalib","numpy","scipy","scikit-learn","Pillow")},
        "gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,"script_sha256":sha(__file__),
        "common_evaluator_sha256":sha(ROOT/"compare_dfr_rd4ad_common.py"),"transform":str(tf)}
    (RUN/"environment.txt").write_text(json.dumps(env,indent=2),encoding="utf-8")
    for y,s,expected in [([1,0,1],[.9,.8,.1],5/6),([1,0],[.5,.5],.5),([0,1],[.1,.9],1),([1,0,0,1],[.9,.8,.7,.6],.75)]:
        assert abs(common.average_precision_score(y,s)-expected)<1e-12
    dump(RUN/"metric_validation.json",{"status":"passed","checks":["noninterpolated AP5/6","ties1/2","perfect1","pooled2x2 mask3/4"]})
    for p in [ROOT/"INDUSTRIAL_ANOMALY_BENCHMARK_PROTOCOL.md",ROOT/"BENCHMARK_CHANGELOG.md",
              ROOT.parent/"benchmark/benchmark_2026/EXISTING_EXPERIMENT_AUDIT.md",
              SETUP/"PROTOCOL.md",SETUP/"evaluate.py",SETUP/"experiment.py",SETUP/"version.json",ROOT/"DFR_RD4AD_common_evaluation/protocol.json"]:
        src(p)
    # Guard every historical result file, not just the consumed manifests.
    for size in CFG["variants"]:
        for p in (ROOT/f"EfficientAD_{size.upper()}_Anomalib_results_20261023").rglob("*"):
            if p.is_file():src(p)
    for size in CFG["variants"]:
        for cat in CFG["categories"]:
            try:
                assert time.perf_counter()-START < CFG["seconds_budget"],"Time budget exceeded"
                RESULTS.append(worker(size,cat))
            except Exception:
                error={"variant":size,"category":cat,"traceback":traceback.format_exc()}
                FAILURES.append(error);dump(RUN/size/cat/"failure.json",error)
                RESULTS.append({"method":"EfficientAD-"+size.upper(),"variant":size,"category":cat,"status":"failed","pixel_ap":None})
                log("FAILED "+size+"/"+cat+"\n"+error["traceback"])
                torch.cuda.empty_cache()
            gc.collect()
            dump(RUN/"metrics.partial.json",RESULTS)
except Exception:
    FAILURES.append({"stage":"setup","traceback":traceback.format_exc()})
    log("SETUP FAILED\n"+traceback.format_exc())
finally:
    log("Checking historical input/checkpoint/source integrity")
    for entry in SOURCES.values():
        entry["sha256_after"]=sha(entry["path"])
        entry["unchanged"]=entry["sha256_before"]==entry["sha256_after"]
        if not entry["unchanged"]:FAILURES.append({"stage":"integrity","path":entry["path"]})
    summary=[]
    for size in CFG["variants"]:
        rows=[r for r in RESULTS if r["variant"]==size and r["status"]=="completed"]
        summary.append({"method":"EfficientAD-"+size.upper(),"categories":len(rows),
              "macro_pixel_ap":sum(r["pixel_ap"] for r in rows)/15 if len(rows)==15 else None,"grid":[256,256]})
    dump(RUN/"metrics.json",{"experiment_id":CFG["experiment_id"],"config":CFG,"results":RESULTS,"status":"failed" if FAILURES else "completed"})
    dump(RUN/"summary.json",summary);dump(RUN/"failures.json",FAILURES)
    dump(RUN/"source_manifest.json",list(SOURCES.values()))
    dump(RUN/"resource_usage.json",{"scope":"new inference/common-grid evaluation only","elapsed_seconds":time.perf_counter()-START,
          "peak_process_ram_bytes":peak_ram(),"gpu_peak_allocated_bytes":max((r.get("gpu_peak_allocated_bytes",0) for r in RESULTS),default=0),
          "training_time":None,"calibration_time":None,"warmup_per_category":30,"gpu":"NVIDIA RTX A4000"})
    fields=["method","category","status","test_images","positive_images","negative_images","pixel_ap","max_historical_image_score_difference"]
    with (RUN/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fields,extrasaction="ignore");w.writeheader();w.writerows(RESULTS)
    log("FINISH "+json.dumps(summary)+" failures="+str(len(FAILURES)))
sys.exit(1 if FAILURES else 0)
