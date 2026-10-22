"""One diagnostic panel; existing PaDiM Gaussian reused, no fitting or test-threshold tuning."""
import sys,json,pickle,time,csv,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from PIL import Image
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'PaDiM_bottle_WR50_Rd550'
sys.path.insert(0,str(RUN/'source_snapshot'))
from mvtec_controlled import MVTecDataset
from run_bottle import embedding_concat
OUT=RUN/'broken_small_diagnostic'
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def main():
 OUT.mkdir(exist_ok=False)
 dump('status.json',dict(stage='feature_extraction'))
 start=time.perf_counter();torch.set_num_threads(4)
 net=wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).cuda().eval()
 net.requires_grad_(False)
 idx=torch.tensor(json.loads((RUN/'selected_channels.json').read_text()))
 train=MVTecDataset(str(ROOT/'archive'),'bottle',True)
 test=MVTecDataset(str(ROOT/'archive'),'bottle',False)
 selected=next(i for i,p in enumerate(test.x) if Path(p).parent.name=='broken_small' and Path(p).stem=='001')
 feats=[]
 handles=[layer[-1].register_forward_hook(lambda mod,inp,out:feats.append(out.detach().cpu()))
          for layer in [net.layer1,net.layer2,net.layer3]]
 def extract(ds):
  batches=[]
  for x,_,_ in DataLoader(ds,batch_size=32,num_workers=0):
   feats.clear()
   with torch.no_grad():net(x.cuda())
   e=embedding_concat(embedding_concat(feats[0],feats[1]),feats[2])
   batches.append(e.index_select(1,idx))
  return torch.cat(batches).flatten(2).numpy()
 tr=extract(train);te=extract(Subset(test,[selected]))
 for h in handles:h.remove()
 del net;torch.cuda.empty_cache()
 with (RUN/'temp_wide_resnet50_2/train_bottle.pkl').open('rb') as f:mean,cov=pickle.load(f)
 assert tr.shape==(209,550,3136)
 train_dist=np.empty((209,3136));test_dist=np.empty((1,3136))
 for i in range(3136):
  if i%128==0:dump('status.json',dict(stage='normal_train_calibration_scoring',position=i,total=3136))
  inv=np.linalg.inv(cov[:,:,i])
  for e,d in [(tr,train_dist),(te,test_dist)]:
   delta=e[:,:,i]-mean[:,i]
   quadratic=np.einsum('bi,ij,bj->b',delta,inv,delta,optimize=True)
   assert np.isfinite(quadratic).all() and quadratic.min()>-1e-4
   d[:,i]=np.sqrt(np.maximum(quadratic,0))
  if i==0:
   reference=mahalanobis(te[0,:,i],mean[:,i],inv)
   assert np.isclose(reference,test_dist[0,i],rtol=1e-4,atol=1e-4)
 def maps(d):
  a=F.interpolate(torch.from_numpy(d.reshape(-1,56,56))[:,None],size=224,
                  mode='bicubic',align_corners=False)[:,0].numpy()
  return np.stack([gaussian_filter(s,sigma=4) for s in a])
 normal=maps(train_dist)
 threshold=float(np.quantile(normal,.995,method='higher'))
 # Threshold is fixed from normal training scores BEFORE reading selected GT.
 amap=maps(test_dist)[0]
 with (RUN/'per_image.csv').open() as f:prior=next(r for r in csv.DictReader(f) if Path(r['path'])==Path(test.x[selected]))
 assert np.isclose(amap.max(),float(prior['image_score']),rtol=1e-3,atol=1e-3),(amap.max(),prior)
 raw=amap>threshold
 pred=morphology.opening(raw,morphology.disk(4))
 with Image.open(test.x[selected]) as im:
  rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC).crop((16,16,240,240)))
 with Image.open(test.mask[selected]) as im:
  gt=np.asarray(im.convert('L').resize((256,256),Image.Resampling.NEAREST).crop((16,16,240,240)))>0
 fig,axs=plt.subplots(1,5,figsize=(20,4.7),constrained_layout=True)
 axs[0].imshow(rgb);axs[1].imshow(gt,cmap='gray',vmin=0,vmax=1)
 axs[2].imshow(rgb)
 heat=axs[2].imshow(amap,cmap='jet',alpha=.5,vmin=float(amap.min()),vmax=float(amap.max()))
 axs[3].imshow(pred,cmap='gray',vmin=0,vmax=1)
 axs[4].imshow(mark_boundaries(rgb,pred.astype(np.uint8),color=(1,0,0),mode='thick'))
 for ax,title in zip(axs,['Input','Ground truth','Anomaly heatmap','Predicted mask (opening r=4)','Predicted boundary (red)']):
  ax.set_title(title);ax.axis('off')
 fig.colorbar(heat,ax=axs[2],shrink=.7,label='Mahalanobis score')
 fig.suptitle(f'PaDiM | broken_small/001 | normal-train q99.5 threshold={threshold:.3f}\nDiagnostic only: in-sample training calibration; no test-F1 selection',fontsize=12)
 fig.savefig(OUT/'broken_small_comparison.png',dpi=160);plt.close(fig)
 dump('calibration.json',dict(threshold=threshold,source='all209 normal fitting images; in-sample calibration',
  quantile=.995,quantile_method='higher',rule='raw score > threshold',
  achieved_train_pixel_fpr=float((normal>threshold).mean()),test_masks_used_for_threshold=False,
  note='Diagnostic threshold only; no independent held-out calibration. Existing threshold-free metrics unchanged.',
  postprocessing='skimage morphology.opening disk radius4 as repo visualization',
  selected='broken_small/001.png; original seed1024 selection retained',
  predicted_pixels_before_opening=int(raw.sum()),predicted_pixels_after_opening=int(pred.sum()),
  saved_image_score=float(prior['image_score']),recomputed_image_score=float(amap.max()),
  seconds=time.perf_counter()-start,model_refitted=False))
 dump('status.json',dict(stage='complete'))
 print('Panel complete',threshold,flush=True)
if __name__=='__main__':main()
