# EfficientAD-S and EfficientAD-M
Both variants completed MVTec AD T1, 15 categories, seed 42. Implementation: Anomalib 2.2.0, commit 592aafc373091dda8206510a0853eb11c8e4099d (https://github.com/open-edge-platform/anomalib). This is not the authors' official implementation.
Batch 1, Adam lr 1e-4 / weight decay 1e-5, 70,000 updates and lr drop at 66,500; 256 input. Fixed-final checkpoints, normal-only fit/calibration split; preserved calibration, no test selection. Imagenette replaces full ImageNet auxiliary data, and exact teacher distillation lineage is unverified: this is not an exact paper reproduction.
[Native S metrics](../../results/benchmarks/efficientad_s_anomalib/results.csv), [native M metrics](../../results/benchmarks/efficientad_m_anomalib/results.csv), [common256 Pixel AP](../../results/benchmarks/efficientad/common256_pixel_ap.csv), [Image AP](../../results/benchmarks/efficientad/image_ap.csv).
Common256 macro Pixel AP: S 61.335189%, M 60.482297%. Nearest-mask256 AP must not be conflated with historical native-resolution Pixel AUROC/AU-PRO. All 3450 regenerated image scores matched historical scores exactly; checkpoints and calibration unchanged. Native-grid Pixel AP is not computed. Single seed; uncertainty Not recorded.

## Reproducibility and limitations

[Recorded workspace and publication manifest](../../reproducibility/efficientad/README.md). Code/configuration/results are evidence snapshots, not a packaged one-command application. Weights, datasets, prediction archives, caches and virtual environments are not published. Retained checkpoint hashes identify local artifacts. No training or inference was run for this publication.
