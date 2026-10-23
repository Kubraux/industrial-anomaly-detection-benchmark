# Result protocol identifiers

All primary tables cover the same fifteen named MVTec AD categories but have different preprocessing, fitting, scoring and evaluators. No cross-protocol ranking is asserted. Fractions are retained in CSV; README values are percentages. Macro = unweighted mean of fifteen category values, excluding original aggregate rows. Blank means unavailable, never zero.

| Identifier | Selection and scoring | Localization / limit |
|---|---|---|
| patchcore_200_thresholds | Normal feature bank; image nearest-feature anomaly score | Full pixel AUROC; local 200-threshold AUPRO@0.30; anomaly-only pixel AUROC is a separate column |
| dfr_native_pro | Final150; reconstruction error | Native PRO implementation, not the common exact AUPRO |
| rd4ad_per_image_pro | Final200; feature disagreement | Native per-image PRO; Bottle had periodic test access |
| padim_exact_030 | Frozen WR50 spatial Gaussian fitting; max raw map | Resize256/crop224; exact local AUPRO@0.30 |
| anogan_native_threshold | Final50 and latent search; normal-holdout area threshold | Native pixel AUROC and normal/anomaly correct fractions; no native image AUROC or AUPRO |
| ganomaly_train_mode_final | Final25 latent discrepancy; separate best-test column | Train-mode test BN persists; final scores are also contaminated; no validated native pixel benchmark |
| skip_final_eval | Fixed final20, one final test; combined score | No validated native pixel benchmark |
| patchsvdd_exact_030 | Final299; full-bank exact L2 map product | Nearest256 GT; local exact AUPRO@0.30 |
| simplenet_fixed_final_exact_030 | New CUDA fixed final; bypasses upstream test-selecting loop | Resize329/crop288; native scores and local exact AUPRO@0.30 |
| supersimplenet_fixed_final_exact_030 | Final300 JIMS extension; separate classification head | 256 GT; local exact AUPRO@0.30; no operating threshold |
| draem_final700_exact_030 | Final700; image=max21x21 average-pooled segmentation map | BGR256, linearly resized uint8 masks; local AUPRO is not pixel AP |

The common DFR/RD4AD comparison has its own [matched evaluator](common_evaluation/report.md). Its GT, FPR aggregation and PRO integration cannot be retroactively assigned to the native tables. AnoGAN's later max-raw-map image AUROC is a separately labeled evaluation.

Test-F1-max/oracle columns use test labels. GANomaly best-test checkpoints are test-selected. Historical SimpleNet CPU selection is likewise test-based. Holdout studies used normal calibration data but followed prior test inspection; no strictly unseen confirmation is claimed.

Recorded timing scopes and hardware differ; no timing leaderboard is supplied. SuperSimpleNet recorded GPU counters sometimes exceed physical VRAM; they are not physical memory requirements. Single-seed category macro averages are not confidence intervals. Paper results embedded in source tables are reference values, not local measurements.

See [experiment protocols](../experiments/PROTOCOLS.md) and [method guides](../methods/README.md) for implementation details.
