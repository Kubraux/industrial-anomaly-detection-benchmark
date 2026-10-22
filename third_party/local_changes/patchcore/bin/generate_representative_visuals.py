"""Saved PatchCore inference only: four representative panels per class; shared nonlinear display."""
import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch
from torch.utils.data import DataLoader
import patchcore.common
import patchcore.patchcore
from patchcore.datasets.mvtec import MVTecDataset,DatasetSplit

CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
MODEL_ROOT=ROOT/"results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_ALL15/models"
DATA_ROOT=ROOT.parent/"archive"
OUTPUT=ROOT/"visual_results_powernorm_staging"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def choose(records):
    groups={}
    for i,r in enumerate(records):groups.setdefault(r[1],[]).append(i)
    for indices in groups.values():indices.sort(key=lambda i:records[i][2])
    if not groups.get("good"):raise ValueError("No good test sample")
    anomalous=sorted(k for k in groups if k!="good")
    selected=[groups["good"][0]]+[groups[k][0] for k in anomalous[:3]]
    remaining=sorted((i for k in anomalous for i in groups[k] if i not in selected),key=lambda i:(records[i][1],records[i][2]))
    selected+=remaining[:4-len(selected)]
    if len(selected)!=4 or len(set(selected))!=4:raise ValueError("Need one good and three anomalous samples")
    return selected

def prohibit_fit(*args,**kwargs):
    raise RuntimeError("Training/coreset fitting is forbidden in this visualization run")

def main():
    torch.set_num_threads(4)
    torch.manual_seed(0);np.random.seed(0)
    # Hard guards: loading existing indices and inference are the only model operations.
    patchcore.patchcore.PatchCore.fit=prohibit_fit
    patchcore.common.FaissNN.fit=prohibit_fit
    OUTPUT.mkdir(exist_ok=True)
    jobs=[]
    for name in CLASSES:
        modeldir=MODEL_ROOT/("mvtec_"+name)
        params=modeldir/"patchcore_params.pkl"
        index=modeldir/"nnscorer_search_index.faiss"
        if not params.is_file() or not index.is_file():raise FileNotFoundError(modeldir)
        with params.open("rb") as f: info=pickle.load(f)
        if tuple(info["input_shape"])!=(3,224,224):raise ValueError(f"Unexpected saved input shape: {info}")
        target=OUTPUT/name
        if target.exists() and any(target.iterdir()):raise FileExistsError(f"Refusing to change existing files: {target}")
        dataset=MVTecDataset(str(DATA_ROOT),classname=name,resize=256,imagesize=224,split=DatasetSplit.TEST)
        selected=choose(dataset.data_to_iterate)
        # Validate mask pairing by file stem rather than relying only on positional ordering.
        for _,kind,img,mask in dataset.data_to_iterate:
            if kind!="good" and (mask is None or Path(mask).stem!=Path(img).stem+"_mask"):
                raise ValueError(f"Mask pairing mismatch: {img}, {mask}")
        jobs.append((name,modeldir,dataset,selected,{p.name:sha(p) for p in (params,index)}))
    audit=[]
    for name,modeldir,dataset,selected,modelhashes in jobs:
        print(f"PROCESSING {name}: full test set ({len(dataset)} images)",flush=True)
        model=patchcore.patchcore.PatchCore(torch.device("cpu"))
        model.load_from_path(str(modeldir),torch.device("cpu"),patchcore.common.FaissNN(False,4))
        model.eval()
        loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
        with torch.inference_mode():
            _,maps,_,_=model.predict(loader)
        maps=np.asarray(maps,dtype=np.float32)
        if maps.shape!=(len(dataset),224,224) or not np.isfinite(maps).all():
            raise ValueError(f"Invalid maps: {name} {maps.shape}")
        vmin,vmax=float(maps.min()),float(maps.max())
        # A single shared class scale, computed BEFORE selecting four visualizations.
        display_max=vmax if vmax>vmin else vmin+max(1e-6,abs(vmin)*1e-6)
        normalization=PowerNorm(gamma=0.5,vmin=vmin,vmax=display_max)
        target=OUTPUT/name;target.mkdir(exist_ok=False)
        paths=[]
        for i in selected:
            record=dataset.data_to_iterate[i]
            _,kind,img_path,mask_path=record
            sample=dataset[i]
            rgb=sample["image"].numpy().transpose(1,2,0)
            rgb=np.clip(rgb*np.asarray(dataset.transform_std)+np.asarray(dataset.transform_mean),0,1)
            mask=sample["mask"].numpy().squeeze()
            fig,axes=plt.subplots(1,3,figsize=(10.4,3.5),layout="constrained")
            axes[0].imshow(rgb);axes[0].set_title("Test image")
            axes[1].imshow(mask,cmap="gray",vmin=0,vmax=1,interpolation="nearest");axes[1].set_title("Ground-truth mask")
            heat=axes[2].imshow(maps[i],cmap="inferno",norm=normalization,interpolation="nearest")
            axes[2].set_title("PatchCore anomaly map\nShared class scale; PowerNorm gamma=0.5",fontsize=10)
            for ax in axes:ax.set_axis_off()
            fig.colorbar(heat,ax=axes[2],fraction=.046,pad=.04)
            fig.suptitle(f"{name} / {kind} / {Path(img_path).name}",fontsize=11)
            savename=f"{kind}_{Path(img_path).stem}.png"
            fig.savefig(target/savename,dpi=180);plt.close(fig)
            paths.append(dict(input=str(Path(img_path).resolve()),mask=mask_path,output=str(target/savename),anomaly_type=kind))
        assert len(list(target.iterdir()))==4
        assert all(sha(modeldir/p)==h for p,h in modelhashes.items())
        entry=dict(classname=name,test_count=len(dataset),model_folder=str(modeldir),output_folder=str(target),
                   vmin=vmin,vmax=vmax,display_vmax=display_max,selected=paths,model_hashes=modelhashes,
                   preprocessing="existing evaluation: resize256, center crop224, ImageNet normalization; left panel denormalized model input",
                   scale="raw anomaly maps; extrema over ALL class test maps; no per-image normalization",
                   display=dict(normalization="PowerNorm",gamma=0.5,cmap="inferno",threshold=None),
                   training=False)
        audit.append(entry)
        (OUTPUT/"representative_visuals_manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
        print(f"CLASS: {name}\nMODEL: {modeldir}\nOUTPUT: {target}\nSCALE: [{vmin}, {vmax}]",flush=True)
        for item in paths:print("SELECTED: "+item["input"],flush=True)
        del model,maps,loader;gc.collect()
    result=dict(classes=len(audit),saved_pngs=sum(len(list((OUTPUT/c).glob("*.png"))) for c in CLASSES),
                training=False,display="PowerNorm gamma=0.5; inferno; shared all-test class extrema")
    (OUTPUT/"representative_visuals_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DONE: "+json.dumps(result),flush=True)

if __name__=="__main__":main()
