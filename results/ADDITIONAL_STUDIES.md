# Significant additional results

Evidence snapshot: 23 October 2026. Settings and selection rules are in [experiments](../experiments/README.md). Tables here contain recorded outcomes, without full logs or per-image arrays.

| Study | Retained results | Interpretation |
|---|---|---|
| DFR feature alignment | [comparison](ablations/dfr_nearest/comparison.csv) | Bilinear versus nearest, three categories; lower false-positive area can coexist with worse AUROC/PRO |
| DFR Native64/128 | [comparison](ablations/dfr_native128/comparison.csv) | Kernel/stride and PCA capacity change together; not a pure output-resize intervention |
| DFR/RD4AD common evaluator | [summary](common_evaluation/report.md), [categories](common_evaluation/comparison.csv) | Matched evaluator reveals threshold-transfer failures despite high AUROC |
| RD4AD holdout | [A/B/C categories](calibration/rd4ad/comparison.csv) | B-C changes calibration only; A-C also changes fitting data; prior test inspection |
| PaDiM holdout | [Bottle](calibration/padim_bottle/metrics.csv), [Screw](calibration/padim_screw/metrics.csv) | q99.5 primary, q99.9 sensitivity; Dice/IoU/precision/recall/FPR retained; prior test inspection |
| GANomaly Capsule / Adam | [variants](diagnostics/ganomaly_archive/capsule_variants.csv), [reset Adam](diagnostics/ganomaly_archive/reset_adam.csv) | Train-normal q99 and test-oracle F1 remain separate; same inspected test set |
| PatchSVDD lambda | [Carpet](diagnostics/patchsvdd.csv) | Completed runs with weak/collapsed representations are retained as negative results |
| PatchSVDD root cause | [feature-scale probes](diagnostics/patchsvdd_probes.csv) | Existing trained/random/raw/pretrained evaluations, not new training; low Screw image AUROC remains |
| DRAEM downloaded weights | [categories](benchmarks/draem/pretrained_evaluation.csv) | Original-evaluator agreement is distinct from locally trained final700 performance |
| AnoGAN later scoring | [max-map image AUROC](benchmarks/anogan/image_max_results.csv) | Later saved-map aggregation; native thresholded accuracy remains in the primary category table |

The archived GANomaly reconstruction/BN and Skip-GANomaly graph/BN investigations are documented by their existing method guides. The GANomaly figure illustrates the archived reconstruction failure, not the official latent score itself. No new deployable threshold is inferred.

Controlled AE/DPAE remain failed literal-sum smoke runs and short normalized diagnostics, with no final benchmark. The older SimpleNet CPU run stopped before a completed category; its historical status is retained under incomplete/. The old PatchSVDD partial table is also historical and must not replace the completed benchmark. Historical AE cleanup records cannot recover deleted metrics. Source-only AnomalyNCD is excluded from completed results.

For PaDiM Bottle, q99.5 to q99.9 raises pooled Dice from 0.64413 to 0.71421 while recall falls from 0.97987 to 0.92430. This is sensitivity analysis on a previously inspected test set, not a selected deployment threshold.

All original outputs remain in the external research archive; [SOURCES.csv](SOURCES.csv) identifies the recorded inputs.
