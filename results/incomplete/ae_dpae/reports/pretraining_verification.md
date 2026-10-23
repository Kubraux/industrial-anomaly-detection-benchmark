# Eğitim öncesi doğrulama — düzeltilmiş controlled implementation

**Ana eğitim başlamadı. Kullanıcı onayı bekleniyor; training_approved=false.**
Deney: **same-dataset controlled AE–DPAE comparison**.

Etiketler: **PAPER** = kullanıcının makale tam metninden aktardığı Fig. 3, Table 1 ve
Eq. (4)–(11) bilgisi. **LOCAL ASSUMPTION** = bu kontrollü deney için seçilen/önerilen
ayrıntı; paper fact değildir. **LOCAL/REPO** = korunan eski repo modeli; paper baseline değildir.

Yazar deposu ve API için önceki doğrudan erişim kontrolü HTTP 404 döndü
(github_access.json); bu revizyonda yeniden kontrol edilmedi. Yazar koduyla doğrulama yok.
Makale mimariyi hiç vermiyor demek yanlıştı: kanal/uzamsal ilerleme ve bloklar biliniyor;
yalnız belirtilmeyen düşük seviyeli ayrıntılar yerel olarak tamamlandı.

## Karşılaştırma tablosu

| Özellik | repo_AE_baseline | AE_L2_controlled | DPAE_controlled | Paper DPAE |
|---|---|---|---|---|
| Architecture | LOCAL/REPO: eski ConvAE E→D | LOCAL ASSUMPTION: DPAE mimarisinin L2 ablation'ı | PAPER: E1→D→E2; aynı mimari, bağımsız E1/E2 parametreleri | PAPER: E1→D→E2 |
| Loss | LOCAL/REPO: 0.5(1−SSIM)+0.5 MSE | LOCAL ASSUMPTION: Σ‖x−xhat‖² ile controlled ablation | PAPER Eq.9: L_rec+λ/2 Σ‖z−v‖²+γ/2 Σ‖zhat−vhat‖² | PAPER Eq.4–9 |
| Optimizer | LOCAL/REPO: eski eğitim reçetesi Adam; yeni koşu yok | PAPER ayarı devralındı: SGD | PAPER: SGD | PAPER: SGD |
| LR | LOCAL/REPO: eski varsayılan .001; yeni koşu yok | PAPER: .001 | PAPER: .001 | PAPER: .001 |
| Weight decay | LOCAL/REPO: eski Adam çağrısında 0 | PAPER: .0005 | PAPER: .0005 | PAPER: .0005 |
| Momentum | LOCAL/REPO: SGD momentum uygulanmaz | PAPER: .9 | PAPER: .9 | PAPER: .9 |
| Epoch | LOCAL/REPO: yeni MT koşusu planlanmadı | PAPER ayarı: 50 planlandı; 0 çalıştı | PAPER: 50 planlandı; 0 çalıştı | PAPER: 50 |
| Batch | LOCAL/REPO: yeni MT koşusu yok | PAPER nominal 8; LOCAL ASSUMPTION son 5+4 | PAPER nominal 8; LOCAL ASSUMPTION son 5+4 | PAPER: 8 |
| Input | LOCAL/REPO: eski config 256² RGB | PAPER boyutu: 224² RGB; LOCAL preprocessing | PAPER boyutu: 224² RGB; LOCAL preprocessing | PAPER: 224×224×3 |
| Train normal | LOCAL/REPO: MT eğitimi yok | LOCAL ASSUMPTION: 761, ortak manifest | LOCAL ASSUMPTION: 761, ortak manifest | PAPER: 330 |
| Test normal | LOCAL/REPO: MT testi yok | LOCAL ASSUMPTION: 191 | LOCAL ASSUMPTION: 191 | PAPER: 142 |
| Test defect | LOCAL/REPO: MT testi yok | LOCAL ASSUMPTION: 392 | LOCAL ASSUMPTION: 392 | PAPER: 422 |
| Anomaly score | LOCAL/REPO: kanal ortalama mutlak hatasının spatial maksimumu | LOCAL ASSUMPTION: reconstruction squared L2 | PAPER Eq.10: ‖z−zhat‖² | PAPER Eq.10: latent squared L2 |
| Latent d | LOCAL/REPO: spatial tensor; FC latent d yok | LOCAL ASSUMPTION: d=32 | LOCAL ASSUMPTION: d=32 | PAPER: d∈{16,32,64,128,256,512,1024}; Table4 d belirsiz |
| Encoder spatial/channel progression | LOCAL/REPO: 3→32→64→128→128 + bottleneck (default base_ch=32) | PAPER Fig.3 akışı; aşağıda E | PAPER Fig.3 akışı; aşağıda E | PAPER: E |
| Decoder stages | LOCAL/REPO: 4 transpose-conv, ReLU, son sigmoid | PAPER Table1 akışı; aşağıda D | PAPER Table1 akışı; aşağıda D | PAPER: D |
| Prototype initialization | Uygulanmaz | Uygulanmaz | LOCAL ASSUMPTION önerisi: bağımsız N(0,1), seed42 | PAPER: rastgele d-vektörü; dağılım belirtilmiyor |
| Prototype update rule | Uygulanmaz | Uygulanmaz | LOCAL ASSUMPTION: v←v−α mean(v−z), α=1; literal Eq.7 local interpretation | PAPER Eq.7 yalnız Δv=mean(v−z); operatör/adım belirtilmiyor |
| Paper-supported vs local assumption | LOCAL/REPO; kaynak değişmedi | PAPER DPAE yapı taşları + LOCAL L2 ablation; Bergmann AE değil | PAPER yapı/kayıp/skor + açık LOCAL tercihler | PAPER ile yerel seçimler birbirinden ayrıldı |

### E — Fig. 3 encoder

224×224×3 → 112×112×128 → 56×56×256 → 28×28×d → 14×14×d →
7×7×d → FC → BN → latent d.

- PAPER: beş Conv + BN + LeakyReLU bloğu; FC+BN latent; E1/E2 parametreleri farklı.
- LOCAL ASSUMPTION: encoder k=3, stride=2, padding=1, bias=true.
  Uzamsal yarılanma stride=2 ile uyumludur; stride'ın kendisi paper fact değildir.
- LOCAL ASSUMPTION: LeakyReLU slope=.2; BN eps=1e−5, momentum=.1,
  affine=true, track_running_stats=true; PyTorch varsayılan ağırlık başlangıcı, seed42.

### D — Table 1 decoder

latent d → Linear/BN → 7×7×d → 14×14×d → 28×28×d →
56×56×256 → 112×112×128 → 224×224×3.

- PAPER: Linear+BN; beş deconv k=3/stride=2 + BN + LeakyReLU;
  çıkış kanalları d,d,256,128,3.
- LOCAL ASSUMPTION: Linear(d,49d) → BN1d(49d) → reshape(d,7,7);
  decoder padding=1, output_padding=1, bias=true. Böylece her aşama tam iki kat büyür.
- Son blok da BN+LeakyReLU'dur. Ek sigmoid/clamp yok; reconstruction [0,1] ile sınırlı değildir.
  Giriş [0,1] ölçeğindedir; L2 ham reconstruction ile hesaplanır.

## Prototiplerin önerilen yerel ayarları — henüz onaylanmadı

prototype_init: normal(mean=0,std=1,seed=42)
prototype_update_rule: v_minus_step_delta
prototype_step: 1.0
prototype_update_timing: after_optimizer_using_pre_step_latents

Başlangıç için ayrı CPU RNG kullanılır; global augmentation RNG akışını tüketmez.
Prototipler buffer'dır; optimizer momentum/weight decay prototiplere uygulanmaz.
Loss mevcut prototiplerle hesaplanır, backward/SGD adımı tamamlanır, sonra aynı
forward'dan detach edilmiş z/zhat ile prototype update yapılır. Bu zamanlama LOCAL ASSUMPTION'dır.
α=1 için yeni prototip tam o batch'in latent ortalamasıdır. α=.1/.5/1 davranışı
unit testlerde doğrulanır; sensitivity training başlatılmadı.

## İki çalışma modu ve onay

- STRICT_REPRODUCTION: prototip dağılımı/update step, encoder ayrıntıları ve exact
  dataset file listesi doğrulanmadığından eğitim engellenir. Yerel değerler girilmiş
  olsa veya training_approved=true yapılsa bile bu mod şu anda açılmaz.
- CONTROLLED_IMPLEMENTATION: yukarıdaki bütün yerel seçimler config/metadata'da saklanır.
  Model oluşturma, iki loss/score yolu ve minibatch prototype update hazırdır.
  **training_approved=false** olduğu için ana eğitim yine başlamaz.
- Kullanıcının açık talebi: “Ben onay vermeden ana training'i başlatma.”
  Onay bu rapordaki somut ayarlara ilişkin olmalıdır; bu revizyonda onay verilmiş sayılmadı.

## Bilimsel adlandırma ve karşılaştırma

Makaledeki AE(L2)/AE(SSIM), Bergmann, Löwe, Fauser, Sattlegger, Steger,
“Improving Unsupervised Defect Segmentation by Applying Structural Similarity to
Autoencoders”, VISAPP 2019 çalışmasına referanstır (kullanıcının tam metin aktarımı).
Bizim AE_L2_controlled, DPAE mimarisinin L2 ablation'ıdır; Table4 AE(L2)'nin
implementasyonu değildir. repo_AE_baseline eski AEModel için ayrı, korunmuş isimdir.

λ=1; **paper notation inconsistency nedeniyle gamma=1 olarak yorumlandı**.
d=32'nin Table4 .91 sonucunu ürettiği varsayılmadı.
Paper .68/.91 AUC yalnız ayrı reference tablosunda; controlled sonuçlara otomatik
eşdeğerlik veya fark/reproduction kontrolü uygulanmaz. ROC eğrisi yalnız AUC'den türetilemez.

Ortak data/preprocessing/seed/evaluation: 761/191/392, seed42, değişmeyen manifestler;
RGB conversion, bilinear square resize, [0,1], train-only yatay/dikey bağımsız flip p=.5.
BatchNorm için 94×8+5+4; örnek atma/tekrarlama yok.
Bu preprocessing ve batching ayrıntıları LOCAL ASSUMPTION'dır.
Loss literal SUM; sessiz mean, gradient clipping veya LR değişikliği yok.

## Final validation ? 2026-09-16

14 tests passed; source syntax passed; dataset revalidated. Original AE and both
manifest SHA256 values are unchanged. Split: 761 / 191 / 392.
All experiment configs retain training_approved=false; no checkpoint exists.
See validation.json for evidence. Main training was not started.

## Smoke-test update (2026-09-16)

User authorized only a two-epoch smoke test. Both models stopped at epoch 1, batch 5 due to Inf in final decoder BN running_var. No epoch completed; epoch 2 was not run. Main training remains unapproved. See [smoke report](smoke_2epoch_report.md). Smoke config alone has scoped approval.
