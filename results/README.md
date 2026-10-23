# Results

Verified 23 October 2026. Eleven methods completed category-wise MVTec AD evaluation (15/15 categories each). These are local single-seed results, not successful paper-reproduction claims. Rows follow the project method order; preprocessing, selection and PRO/AUPRO definitions differ, so this is not a directly comparable leaderboard.

## Completed benchmarks

Equal-category macro averages in percent; CSV retains original fractions and full precision. PRO/AUPRO entries use the row's recorded evaluator, not a common metric implementation.

| Method | Image AUROC | Pixel AUROC | Recorded PRO/AUPRO | Protocol |
|---|---:|---:|---:|---|
| [PatchCore](benchmarks/patchcore/patchcore_final_results.csv) | 99.180 | 98.111 | 93.004 | patchcore_200_thresholds |
| [DFR](benchmarks/dfr/dfr_all_results.csv) | 93.854 | 95.251 | 89.943 | dfr_native_pro |
| [RD4AD](benchmarks/rd4ad/all15_results.csv) | 98.357 | 97.789 | 93.941 | rd4ad_per_image_pro |
| [PaDiM](benchmarks/padim/results.csv) | 95.402 | 97.310 | 92.246 | padim_exact_030 |
| [AnoGAN adaptation](benchmarks/anogan/results.csv) | ? | 76.354 | ? | anogan_native_threshold |
| [GANomaly official-flow](benchmarks/ganomaly/results.csv) | 57.621 | ? | ? | ganomaly_train_mode_final |
| [Skip-GANomaly](benchmarks/skip_ganomaly/skipganomaly_mvtec15_results.csv) | 65.516 | ? | ? | skip_final_eval |
| [PatchSVDD](benchmarks/patchsvdd/results.csv) | 80.336 | 91.160 | 68.606 | patchsvdd_exact_030 |
| [SimpleNet](benchmarks/simplenet/results.csv) | 98.429 | 96.046 | 89.256 | simplenet_fixed_final_exact_030 |
| [SuperSimpleNet (JIMS)](benchmarks/supersimplenet/results.csv) | 98.305 | 97.521 | 91.193 | supersimplenet_fixed_final_exact_030 |
| [DRAEM](benchmarks/draem/results.csv) | 97.597 | 97.134 | 92.506 | draem_final700_exact_030 |

[Machine-readable summary](summary.csv) ? [Protocol definitions](PROTOCOLS.md) ? [Additional studies](ADDITIONAL_STUDIES.md) ? [Input provenance](SOURCES.csv)

Primary category CSVs retain additional recorded metrics, including AP, anomaly-only pixel AUROC, thresholded correctness, component-score AUROC and oracle F1 where available. These columns keep their original names. Aggregate rows and absolute workstation prefixes were removed from publication tables; original measurements are unchanged and archived. Paper-reference columns are not local results.

**GANomaly:** the image AUROC above is final25; its best-test macro is 69.171%, separately preserved in summary.csv and the category table. Both inherit train-mode test BatchNorm contamination. **AnoGAN:** native image AUROC is unavailable; the later max-map evaluation gives 67.542% and is [reported separately](benchmarks/anogan/image_max_results.csv).

## Selected scientific figures

Figures are illustrative selections from existing outputs, not representative random samples. Each canvas is 1200x675. GT/binary masks use nearest-neighbor resizing; image proportions are preserved. Source plots, crop rectangles, hashes and scale notes are recorded in [figure provenance](../assets/figures/SOURCES.csv).

**Common evaluation ? ranking versus threshold transfer.** Same Leather/color/000 input, GT and two recorded models. Maps are independently scaled; cyan denotes GT. The RD4AD mask illustrates why high AUROC alone does not establish a useful operating threshold. [Quantitative comparison](common_evaluation/report.md).

![DFR and RD4AD common evaluation with maps and binary masks](../assets/figures/common_leather.png)

**DFR ablation.** Native64 and Native128 map pair retains its shared original display scale. Kernel/stride and PCA capacity change together.

![DFR spatial ablation with shared-scale maps and binary masks](../assets/figures/dfr_native64_native128.png)

**SuperSimpleNet final300.** Input, GT, anomaly map and overlay from the completed JIMS-extension run. Per-image heatmap normalization supports spatial inspection only; the image-level score comes from a separate head.

![SuperSimpleNet Bottle input GT map and overlay](../assets/figures/supersimplenet_bottle.png)

**PaDiM calibration.** Normal good/010 and defective broken_large/005. Masks use normal-holdout q99.5 and q99.9 thresholds; heatmaps are independently scaled. The sensitivity comparison followed prior test inspection.

![PaDiM holdout calibration normal and defective masks](../assets/figures/padim_calibration.png)

**GANomaly failure example.** Archived Carpet reconstruction/error examples show nearly uniform reconstructions on both a color defect and a normal image. This is a reconstruction diagnostic; it is not the official latent anomaly score.

![GANomaly Carpet archived reconstruction failure](../assets/figures/ganomaly_carpet_diagnostic.png)

**PatchCore localization.** Bottle/broken_small/000, with class-shared PowerNorm gamma0.5 display. No binary operating threshold is implied.

![PatchCore Bottle localization](../assets/figures/patchcore_bottle.png)

**PaDiM qualitative contrast.** A normal and defective Bottle under the original shared Mahalanobis-score overlay scale. Overlays are not binary masks.

![PaDiM Bottle normal and defective overlays](../assets/figures/padim_bottle.png)

**PatchSVDD qualitative contrast.** The same two sample identities under the original shared map-product display scale. Good Bottle localization does not generalize to all categories: weak/collapsed category outcomes remain in the tables.

![PatchSVDD Bottle normal and defective score maps](../assets/figures/patchsvdd_bottle.png)

## Evidence gaps and publication limits

- No native pixel benchmark/AUPRO is verified for GANomaly or Skip-GANomaly; AnoGAN has no native image AUROC/AUPRO. Blank cells are retained.
- Native DFR/RD4AD PRO and local exact/approximate AUPRO differ. Category macros cannot repair mismatched protocols.
- Full reproduction remains unverified: some historical source/configuration details and PatchCore coreset settings are missing; the deleted historical AE metrics cannot be recovered.
- The experiment registry and method evidence still include external workspace paths. The latest benchmark category tables are bundled here, but those external references were not rewritten outside this task. Some older experiment configuration links in method guides were already unavailable after experiment simplification.
- Historical incomplete/ files survive solely for existing method references. They do not override current benchmark results. The root README/release manifests may still describe older snapshots.
- All former results/assets files are preserved outside GitHub under publication_archive/results_assets_20261023; original training/evaluation records elsewhere are unchanged. No experiments were rerun.
