# Magnetic Tile kontrollü AE–DPAE raporu — güncel revizyon

**Ana eğitim başlamadı; kullanıcı onayı bekleniyor.**
[Güncellenmiş tam pretraining tablosu](pretraining_verification.md)

## A. Tamamlanan temizlik

Eski archive/ dataset klasörü (6644 dosya), runs/ae/bottle/ (13 dosya),
summary/ (7 dosya) ve eski loader/train/eval/prepare cache dosyaları (6) silindi.
Toplam 6670 dosya. Dökümler: cleanup_inventory.csv, cleanup_result.json, cleanup_cache.csv.
Model kaynakları, requirements, environment, ortak utility'ler ve Magnetic Tile korunmuştur.
Aktif veri hattı Magnetic Tile manifestlerine taşınmıştır.

## B–D. Dataset ve kesin split — değişmedi

Kök: ../Magnetic-tile-defect-datasets.-master/Magnetic-tile-defect-datasets.-master/
Her MT_*/Imgs klasöründe JPG kaynak ve eş isimli PNG maske bulunur.
1344 JPG (grayscale L) + 1344 PNG maske doğrulandı. Bozuk dosya ve boyut eşleşmezliği yok.
Genişlik 103–632; yükseklik 222–403. Kökteki tanıtım görselleri dahil değildir.

| Sınıf | Görüntü | Train | Test |
|---|---:|---:|---:|
| MT_Free | 952 | 761 | 191 |
| MT_Blowhole | 115 | 0 | 115 |
| MT_Break | 85 | 0 | 85 |
| MT_Crack | 57 | 0 | 57 |
| MT_Fray | 32 | 0 | 32 |
| MT_Uneven | 103 | 0 | 103 |
| Toplam | 1344 | 761 | 583 |

Seed42, sıralı normal yollar Random(42) ile karıştırıldı, floor(%80) train.
Paper split 330/142/422; exact dosya listesi bilinmiyor. Paper'ın 1344 toplam
tanımı ile 472+422=894 deney sayısı tutarsız. Dataset/split birebir reproduction değildir.
Deney adı **same-dataset controlled AE–DPAE comparison**.

## E. Manifest / leakage

train_manifest.csv ve test_manifest.csv değiştirilmedi. Hash kontrolü korunuyor.
Train'de anomaly yok, path/pixel-hash overlap yok; tüm görüntüler bir kez kapsanıyor.
Aynı fiziksel parçanın farklı çekimleri için grup kimliği yok; group-level bağımsızlık iddiası yok.
AE_L2_controlled/DPAE_controlled aynı manifest/seed/preprocessing/evaluation koduna yönlendirilir.
Henüz iki trained run olmadığı için gerçek run hash eşitliği iddia edilmez.

## F. Düzeltilen yöntem ve isimler

repo_AE_baseline = değişmeyen eski AEModel; SSIM+MSE / eski Adam / max-pixel error.
AE_L2_controlled = Fig3/Table1 DPAE mimarisinin L2 ablation'ı; Bergmann AE(L2) değildir.
DPAE_controlled = Fig3/Table1 yapı + Eq4–9 loss + Eq10 latent score.

Önceki 3→32→64→128→128 controlled encoder ve eski decoder kaldırıldı.
Yeni encoder 3→128→256→d→d→d, 224→112→56→28→14→7, Conv+BN+LeakyReLU,
FC+BN latent d. Decoder Linear/BN ve beş 3×3/stride2 deconv+BN+LeakyReLU;
kanallar d,d,256,128,3; spatial 7→14→28→56→112→224.
Son decoder bloğunda da BN+LeakyReLU vardır; sigmoid/clamp yok.

Paper'da verilmeyen kernel/stride/padding/output-padding, slope, BN layout/ayarları,
default initialization, preprocessing ve prototype ayrıntıları config'te LOCAL ASSUMPTION'dır.
d=32 açık yerel seçim; Table4 .91 sonucunun d=32 olduğu varsayılmadı.
paper notation inconsistency nedeniyle gamma=1 olarak yorumlandı; lambda=1.

Proposed LOCAL prototype init N(0,1), seed42; v←v−αΔv, α=1.
Bu “literal Eq.7 local interpretation”; paper update iddiası değildir.
Prototip güncellemesi optimizer sonrası aynı forward'dan detach latentlerle;
prototipler buffer, SGD/momentum/weight decay kapsamı dışında.
α=.1/.5/1 matematiği unit testlerde; sensitivity training yapılmadı.

STRICT_REPRODUCTION eksik kaynak bilgileri nedeniyle kapalı.
CONTROLLED_IMPLEMENTATION hazır, fakat training_approved=false nedeniyle kullanıcı
onayı olmadan ana eğitim başlamaz. Tam tercih tablosu pretraining_verification.md'de.

## G–I. Sonuçlar ve ayrı paper referansları

| Controlled model | Our ROC-AUC | Durum |
|---|---|---|
| AE_L2_controlled | — | Eğitim başlamadı |
| DPAE_controlled | — | Eğitim başlamadı |

| Paper model (yalnız referans) | MT AUC |
|---|---:|
| AE(L2), Bergmann VISAPP 2019 referanslı | .68 |
| DPAE, Table4 | .91 |

Paper AE(L2), bizim L2 ablation'ımız değildir. Otomatik fark/reproduction kontrolü yok.
Yeni sonuçlar yalnız aynı manifest altında birbirleriyle karşılaştırılır.
ROC/score histogramları, raw/normalized per-image scores ve image-level AUC kodu hazır;
gerçek eğri/metrik/checkpoint üretilmedi. Threshold/F1 hesaplanmadı.

## J. Bilimsel yorum sınırları

Dataset/kurasyon, exact split, preprocessing, mimari, latent d, SUM loss ölçeği,
prototype init/update/step, optimizer, seed ve Python/PyTorch sürümü olası fark kaynaklarıdır.
Mevcut ortam Python3.10.11/PyTorch2.14.0+cpu; paper Python3.7/PyTorch1.2.0.
Paper .68/.91'in “yeniden üretildiği” iddia edilmez.

Son test ve korunmuş dosya hash kanıtları validation.json'a kaydedilir.
Kaynak: kullanıcının Fig3/Table1/Eq4–11 tam metin aktarımı.
Resmî GitHub önceki kontrolde 404; erişim kanıtı github_access.json.

## Final validation ? 2026-09-16

14 tests passed; source syntax passed; dataset revalidated. Original AE and both
manifest SHA256 values are unchanged. Split: 761 / 191 / 392.
All experiment configs retain training_approved=false; no checkpoint exists.
See validation.json for evidence. Main training was not started.

## Smoke-test update (2026-09-16)

User authorized only a two-epoch smoke test. Both models stopped at epoch 1, batch 5 due to Inf in final decoder BN running_var. No epoch completed; epoch 2 was not run. Main training remains unapproved. See [smoke report](smoke_2epoch_report.md). Smoke config alone has scoped approval.
