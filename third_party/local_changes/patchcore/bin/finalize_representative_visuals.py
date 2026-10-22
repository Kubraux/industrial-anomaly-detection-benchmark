"""Validate staged plots, publish exactly four PNGs/class, retain metric files."""
import json
import hashlib
import os
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
STAGE=(ROOT/"visual_results_powernorm_staging").resolve()
OUT=(ROOT/"visual_results").resolve()
CLASSES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper".split()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    audit=json.loads((STAGE/"representative_visuals_manifest.json").read_text())
    assert [a["classname"] for a in audit]==CLASSES
    preserved={str(p.resolve()):sha(p) for p in OUT.rglob("*") if p.is_file() and p.suffix.lower()!=".png" and p.parent!=OUT}
    old=json.loads((OUT/"representative_visuals_manifest.json").read_text())
    previous={a["classname"]:a for a in old}
    checks=[]
    for a in audit:
        c=a["classname"]
        target=(OUT/c).resolve()
        assert target.parent==OUT
        items=a["selected"]
        assert len(items)==4 and sum(x["anomaly_type"]=="good" for x in items)==1
        assert a["display"]==dict(normalization="PowerNorm",gamma=0.5,cmap="inferno",threshold=None)
        names={Path(x["output"]).name for x in items}
        assert {p.name for p in (STAGE/c).glob("*.png")}==names
        for name in names:
            with Image.open(STAGE/c/name) as im:
                im.verify()
        for name,h in a["model_hashes"].items():
            assert sha(Path(a["model_folder"])/name)==h
        if c in previous:
            before=previous[c]
            assert [x["input"] for x in items]==[x["input"] for x in before["selected"]]
            checks.append(dict(classname=c,vmin_delta=a["vmin"]-before["vmin"],vmax_delta=a["vmax"]-before["vmax"]))
    history=ROOT/"results"/"representative_visuals_previous_manifest.json"
    history.write_text(json.dumps(old,indent=2),encoding="utf-8")
    removed=[]
    for a in audit:
        c=a["classname"]; target=(OUT/c).resolve();target.mkdir(exist_ok=True)
        names={Path(x["output"]).name for x in a["selected"]}
        for name in names:
            destination=(target/name).resolve()
            assert destination.parent==target and target.parent==OUT
            os.replace(STAGE/c/name,destination)
        for p in target.glob("*.png"):
            assert p.resolve().parent==target
            if p.name not in names:
                removed.append(str(p));p.unlink()
        assert len(list(target.glob("*.png")))==4
        a["output_folder"]=str(target)
        for item in a["selected"]: item["output"]=str(target/Path(item["output"]).name)
        print("CLASS:",c,"\nMODEL:",a["model_folder"],"\nOUTPUT:",target)
        for item in a["selected"]: print("SELECTED:",item["input"])
    assert all(sha(Path(p))==h for p,h in preserved.items())
    (OUT/"representative_visuals_manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    result=dict(classes=15,saved_pngs=60,training=False,model_hashes_unchanged=True,
                preserved_non_image_files=list(preserved),removed_old_pngs=removed,
                previous_scale_comparison=checks,display="PowerNorm gamma=0.5; inferno; shared all-test class extrema")
    (OUT/"representative_visuals_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("DONE: 15 classes; 60 PNGs; no training; model files unchanged")
if __name__=="__main__": main()
