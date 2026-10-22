from pathlib import Path
import json,csv

def build(report,setup,observed):
 refs=[]
 # SimpleNet CVPR table 1; I/P AUROC in percent. AUPRO not reported there.
 sn={'bottle':(100,98.0),'cable':(99.9,97.6),'capsule':(97.7,98.9),'carpet':(99.7,98.2),'grid':(99.7,98.8),'hazelnut':(100,97.9),'leather':(100,99.2),'metal_nut':(100,98.8),'pill':(99,98.6),'screw':(98.2,99.3),'tile':(99.8,97),'toothbrush':(99.7,98.5),'transistor':(100,97.6),'wood':(100,94.5),'zipper':(99.9,98.9)}
 for cat,(i,p) in sn.items():refs.append(dict(reference='SimpleNet paper Table 1',method='simplenet',category=cat,image_auroc=i/100,pixel_auroc=p/100,aupro_030=None,source='https://arxiv.org/pdf/2303.15140'))
 for ver in ['icpr','jims']:
  d=json.loads((setup/'sources/supersimplenet/paper_results'/ver/'results_avg_std.json').read_text())
  vals=d['avg']['mvtec'] if ver=='icpr' else d['unsup']['1']['avg']['mvtec']
  for cat in sn:refs.append(dict(reference='SuperSimpleNet '+ver+' official 5-run mean',method='supersimplenet',category=cat,image_auroc=vals['I-AUROC'][cat],pixel_auroc=vals['P-AUROC'][cat],aupro_030=vals['AUPRO'][cat],source='https://github.com/blaz-r/SuperSimpleNet/tree/'+json.loads((setup/'versions.json').read_text())['supersimplenet']['commit']+'/paper_results/'+ver))
 with (report/'paper_references.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=refs[0]);w.writeheader();w.writerows(refs)
 compare=[]
 for ref in refs:
  r=next((x for x in observed if x['method']==ref['method'] and x['category']==ref['category']),None)
  for metric in ['image_auroc','pixel_auroc','aupro_030']:
   compare.append(dict(reference=ref['reference'],method=ref['method'],category=ref['category'],metric=metric,paper=ref[metric],measured=r[metric] if r else None,delta_percentage_points=100*(r[metric]-ref[metric]) if r and ref[metric] is not None else None))
 with (report/'paper_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=compare[0]);w.writeheader();w.writerows(compare)
 text=['# Makale ve resmi kod protokol incelemesi','','Sonuclar tamamlanana kadar paper_comparison.csv olcum hucreleri bostur. Bos degerler sifir degildir.','', '| Bilesen | SimpleNet resmi kod | SuperSimpleNet guncel unsupervised MVTec |','|---|---|---|', '| Backbone | ImageNet V1 WideResNet50-2; layer2+layer3; dondurulmus | Ayni backbone/katmanlar; dondurulmus |', '| Ozellikler | 3x3 patch; 1536 boyut; tek Linear adaptor; MLP discriminator | 2x ozellik buyutme + 3x3 ortalama; 1x1 adaptor; ayri segmentation/classification basliklari |', '| Girdi | Resize 329, center crop 288; ImageNet normalize | 256x256 resize; ImageNet normalize |', '| Sentetik anomali | Gaussian std 0.015 | Gaussian std 0.015; Perlin threshold 0.2 |', '| Loss | Normal ve sentetik icin truncated hinge, margin 0.5 | Segmentation truncated L1 + focal; classification focal (gamma 4) |', '| Optimizer | Adaptor AdamW lr 1e-4 wd 0.01; discriminator Adam lr 2e-4 wd 1e-5 | AdamW adaptor lr 1e-4 wd 0.01; segmentation/classification lr 2e-4 wd 1e-5 |', '| Takvim | 40 meta-epoch x 4 GAN epoch; batch 8; cosine kapali | 300 epoch; batch 32; 240 ve 270 sonunda lr x0.4 |', '| Diger | Seed 0 | Seed 42; stop_grad=True; clip_grad=False; adapt_cls_feat=False; dt/dilate yok |','', '## Checkpoint ve veri izolasyonu','', 'SimpleNet resmi train() test I-AUROC ile secim, esitlikte test P-AUROC ile secim yapiyor. Kullanici onayi ile resmi _train_discriminator() degistirilmeden 40 kez cagrilir, son agirlik kaydedilir. Ara test yok. SuperSimpleNet resmi son epoch agirligini kaydediyor; her 4 epoch test islemi kaldirildi. Mimari, loss, optimizer adimlari ve scheduler korunur. Recovery checkpointleri sadece kurtarma icindir; skorla secilmez. Smoke test normal train verisinden bir batch kullanir, tam egitim modeli sifirdan kurulup seed yeniden ayarlanir.','', '## Makale-kod farklari ve karsilastirma sinirlari','', 'SimpleNet makalesi 256 resize / 224 crop, batch 4, bias olmadan adaptor ve Adam tarif eder. Resmi run.sh 329/288 ve batch 8; resmi Projection bias icerir ve adaptor AdamW varsayilan wd=0.01 kullanir. Bu deney kullanicinin onayladigi resmi kodu korur; makalenin birebir yeniden uretimi olarak sunulmaz. Makale Tablo 1 ortalamalari I-AUROC %99.6 ve P-AUROC %98.1; AUPRO bu tabloda yoktur.','', 'SuperSimpleNet guncel kod JIMS uzantisidir: adapt_cls_feat=False. ICPR modeli adapt edilmis classification ozellikleri kullanir; guncel deney ICPR agirliklarini veya supervised/mixed veri rejimini kullanmaz. Ana referans JIMS unsup/1; ICPR sadece ayri tarihsel referanstir. Resmi paper_results bes kosu ortalamasidir, deney her yontem icin tek seed kullanir.','', 'Yerel ortam Python 3.11; SuperSimpleNet requirements torch 2.1.0+cu118 / torchvision 0.16.0+cu118 / numpy 1.26.0 / anomalib 0.7 korunur. SimpleNet README Python 3.8 / torch 1.12.1 / torchvision 0.13.1 / numpy 1.22.4 bildirir ve diger surumlerin de calisabilecegini belirtir; bu deney ortak izole torch 2.1 ortamindadir. Ortam farki kayit altindadir; onceki deney ortamlari degistirilmez.','', 'Metrikler her yontemin kendi on-isleme cikti boyutunda hesaplanir. SimpleNet crop kenarlari atar; SuperSimpleNet tam goruntuyu kucultur. Bu nedenle lokalizasyon farki yalnizca mimarinin etkisi degildir. SimpleNet maskeleri resmi transform sonrasinda >0.5 ile ikililestirilir; SuperSimpleNet nearest mask resize kullanir.','', 'AUPRO exact global-FPR sweep ve 8-connected bilesenlerle hesaplanir; resmi SimpleNet compute_pro 5x5 prediction dilation, threshold-grid ve gozlenen maksimum FPR ile normalizasyon kullaniyor. Bunlar standart AUPRO@0.30 ile ayni degildir ve bu deneyde kullanilmaz. Anomalib region ROC interpolasyonuyla da sayisal farklar olabilir. SuperSimpleNet resmi sigmoid islemi korunur; sadece global min/max normalizasyonu atlanir (siralamayi degistirmez). SimpleNet native skor kullanir. Esik secimi veya test tabanli ayar yoktur.','', 'Inference CUDA senkronizasyonuyla batch=1, H2D ve resmi postprocessing dahil olculur. SimpleNet resmi kod gereksiz buyutulmus ozellik ciktilari da urettiginden bunlarin suresi dahildir; optimizasyon yapilmamistir. SuperSimpleNet makalesi de bu maliyete dikkat ceker. GPU A4000; makalelerdeki 3080Ti veya V100S ile dogrudan hiz/bellek esdegerligi varsayilmaz. PyTorch peak allocated/reserved tum GPU kullanimiyla ayni degildir.','', '## Kaynaklar','', '- https://github.com/DonaldRR/SimpleNet','- https://arxiv.org/pdf/2303.15140','- https://github.com/blaz-r/SuperSimpleNet','- https://arxiv.org/abs/2408.03143','- https://arxiv.org/abs/2508.19060','', '## Referans tablosu','', '| Referans | Kategori | I-AUROC % | P-AUROC % | AUPRO % |','|---|---|---:|---:|---:|']
 for r in refs:
  pro='N/A' if r['aupro_030'] is None else f"{100*r['aupro_030']:.3f}"
  text.append(f"| {r['reference']} | {r['category']} | {100*r['image_auroc']:.3f} | {100*r['pixel_auroc']:.3f} | {pro} |")
 (report/'protocol_and_papers.md').write_text('\n'.join(text),encoding='utf-8')


_original_build = build
def build(report, setup, observed):
 _original_build(report, setup, observed)
 means=[]
 keys=['image_auroc','pixel_auroc','aupro_030','train_seconds','evaluation_seconds','inference_batch1_mean_ms','train_peak_allocated_bytes','inference_peak_allocated_bytes']
 for method in ['simplenet','supersimplenet']:
  rows=[r for r in observed if r['method']==method]
  if len(rows)==15:means.append(dict(method=method,categories=15,**{k:sum(r[k] for r in rows)/15 for k in keys}))
 if means:
  with (report/'means.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=means[0]);w.writeheader();w.writerows(means)
 refs=list(csv.DictReader((report/'paper_comparison.csv').open()))
 lines=['# Olcum ve makale karsilastirmasi','','Farklar yuzde puan cinsindendir. Henuz tamamlanmayan olcumler bekliyor olarak gosterilir. Ana SSN referansi JIMS unsupervised; ICPR tarihsel referanstir. Ayrintili protokol farklari protocol_and_papers.md dosyasindadir.','','| Referans | Kategori | Metrik | Makale % | Olcum % | Fark pp |','|---|---|---|---:|---:|---:|']
 for r in refs:
  vals=[('N/A' if not r[k] else f"{float(r[k])*100:.3f}") for k in ['paper','measured']]
  if not r['measured']:vals[1]='bekliyor'
  delta=f"{float(r['delta_percentage_points']):+.3f}" if r['delta_percentage_points'] else 'N/A'
  lines.append(f"| {r['reference']} | {r['category']} | {r['metric']} | {vals[0]} | {vals[1]} | {delta} |")
 (report/'paper_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
