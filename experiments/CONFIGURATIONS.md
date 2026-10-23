# Experimental configurations

Evidence snapshot: 23 October 2026. These are recorded settings, not runnable launch recipes. For architectures and source provenance see the [method guides](../methods/README.md) and [source catalog](../third_party/SOURCES.csv).

## Main benchmarks

- **patchcore** (patchcore_all15): Frozen WR50 layer2/3; resize256/center224; nearest-normal feature bank. No optimizer epochs should be invented for fitting. Local AUPRO uses 200 thresholds. Hardware for fitting is not inferred from the CPU AUPRO helper.
- **dfr** (dfr_all15): VGG19 first12 ReLU features;256px;baseline bilinear alignment;4x4 aggregation/stride4;PCA90%;Adam1e-4;batch4;150 epochs;seed0. Nearest changes alignment only using saved baseline latent dimensions; native128 uses kernel/stride2 and recomputed PCA. Environment is recorded per category.
- **rd4ad** (rd4ad_all15): Frozen pretrained WR50 teacher with bottleneck/decoder;256px;200 epochs;batch16;Adam0.005 betas(0.5,0.999);seed111. Final checkpoint; native disagreement maps and per-image PRO. Holdout run uses an80/20 normal split and separate A/B/C model/threshold conditions.
- **padim** (padim_all15): Frozen WR50 layers1/2/3;random550;seed1024;covariance+0.01I;resize bicubic256/crop224;map sigma4. Spatial Mahalanobis maps; fitting rather than gradient epochs. Bottle holdout167/42 and Screw256/64; q0.995 primary,q0.999 sensitivity,imageq0.95, higher quantile and strict >.
- **anogan** (anogan_all15): MVTec2019 adaptation: latent64,50 epochs,seed0,300 latent-search steps; objects128/textures512. Raw L2 map evaluation and normal-holdout area threshold; later max-map image AUROC is separate. Category protocols take precedence over generic labels.
- **ganomaly** (ganomaly_all15): Encoder-decoder-encoder; main official-flow25 epochs,seed0,RGB256,latent100,batch8,Adam0.0002. Image score is latent discrepancy. Earlier stopped4/15 run, Capsule one-factor variants, Adam reset and BN diagnostics have separate records and selection rules.
- **skip_ganomaly** (skip_ganomaly_all15): Main run20 epochs,seed0,RGB256,batch8,CPU; final evaluation once. Combined reconstruction/latent-feature scoring; no validated native pixel benchmark. Earlier one-epoch audit checks graph separation and BN/logging side effects.
- **patchsvdd** (patchsvdd): Main run299 passes,D64,lambda0.001,seed1024,batch64;full-bank exact float32 L2 and nearest256GT; no best-test selection. Carpet lambda0.1/0.01 runs remain separate. Random/pretrained/numerical probes do not establish new training.
- **simplenet** (simplenet_all15_20261020): Old CPU run stopped during Screw at meta-epoch8 with0 completed categories. New CUDA run:ImageNet V1 WR50 layers2/3;resize329/crop288;batch8;seed0;40 discriminator calls x4 GAN epochs;fixed final. Native scores and local exact AUPRO. Config snapshots retain optimizer and environment values.
- **supersimplenet** (supersimplenet_all15_20261020): JIMS unsupervised extension,adapt_cls_feat=False;ImageNet V1 WR50 layer2/3;256x256;batch32;seed42;300 epochs;AdamW adaptor1e-4,heads2e-4;scheduler240/270 gamma0.4. Gaussian std0.015/Perlin0.2;separate classification/segmentation logits with sigmoid;final checkpoint. RTX A4000 in recorded environment.
- **draem** (draem_all15_20261015): Normal MVTec images plus DTD synthetic anomalies;BGR256/255;MSE+SSIM+Focal;700 epochs,batch8,seed1024,Adam1e-4,560/630 gamma0.2;workers0;AMP/TF32off. Softmax anomaly channel;image=max21x21 average-pool map. Final700; downloaded-pretrained evaluation is separate. Recorded RTX A4000.

## Exact recorded settings

[configurations.json](configurations.json) consolidates the saved configuration, optimizer, protocol and environment snapshots for retained studies. Within each experiment and filename, common values apply to every listed source; overlay each source's overrides to recover its publication dictionary. Machine-specific workspace prefixes use the WORKSPACE_ROOT placeholder (in angle brackets); experiment parameters and category-specific settings remain unchanged. Source references and SHA-256 identify the unmodified archived originals, not the sanitized dictionaries. This removes repetition without selecting one category as representative of all categories.

All source references are relative to the repository root (locally, github_release). WORKSPACE_ROOT denotes the reader's research workspace; original absolute paths remain in the research archive. See the [portability notes](README.md#configuration-portability). Some protocol snapshots include stale progress, intended budgets or upstream defaults; the registry and evaluation protocols specify the verified execution and selection rules. An absent configuration is unknown, not an invitation to substitute a method default.

PatchCore and Skip-GANomaly have no embedded configuration snapshots in the detailed catalog; the verified summaries above and linked method guides document their available settings. The consolidated settings do not establish complete reproducibility.

## Additional studies

The registry retains DFR alignment/resolution ablations and spatial/threshold investigations; DFR/RD4AD common evaluation; RD4AD and PaDiM normal-holdout calibration; GANomaly Capsule variants, Adam-reset and BN investigations; PatchSVDD lambda ablations and feature/numerical failure analyses; Skip-GANomaly BN/graph audits; AnoGAN's later image-scoring variant; and DRAEM pretrained evaluation. Grouped studies reference their original result directories or reports; individual probes are not presented as independent benchmarks.

- DFR nearest uses saved baseline PCA dimensions; native128 recomputes PCA. Keep these changes distinct.
- GANomaly Capsule variants include size64_nz100, size256_nz256 and size256_nz100_stable_bce. Their settings remain in the consolidated protocol/config sources.
- Controlled AE and DPAE are represented together: both literal-sum smoke attempts failed in epoch 1, batch 5; each normalized-loss probe completed 10 batches. Only one additional AE epoch survives in the longer diagnostic. No final benchmark is established. Shared configurations include both diagnostic plans; plans are not completion evidence.
- The older SimpleNet CPU run stopped with 0/15 categories completed. The earlier GANomaly run stopped after four categories. These are distinct from their completed benchmarks.
- Some PatchSVDD diagnostic and DRAEM pretrained status files retain old progress counters despite final study results. Use the result/report references alongside those statuses.

## Dataset and evaluation interpretation

Main benchmarks use MVTec AD; controlled AE/DPAE use Magnetic Tile (761 normal training images; 191 normal and 392 defective test images). DRAEM uses DTD for synthetic anomalies. Per-category splits, preprocessing, optimizer parameters, seeds and recorded hardware are retained where the source snapshots provide them. Missing values remain unknown. A CPU evaluation helper does not establish training hardware.

[PROTOCOLS.md](PROTOCOLS.md) specifies model selection, threshold calibration, mask/score conventions and metric limitations. Result tables are referenced by the registry, never copied here.
