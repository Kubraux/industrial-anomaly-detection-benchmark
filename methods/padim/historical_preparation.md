# PaDiM Bottle: paper/repository audit and pending experiment

Status: PREPARED ONLY. User approval required before any model loading, feature extraction or Gaussian fitting. No Bottle performance result exists for this preparation.

Sources:
- Paper: https://arxiv.org/pdf/2011.08785 (arXiv v1, sections III, IV-B and V-B).
- Repository: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
- Read main.py and datasets/mvtec.py, local originals preserved; hashes in original_source_hashes.json.

## Paper vs repository
| Component | Finding / prepared action |
|---|---|
| Backbone | Paper includes ImageNet-pretrained ResNet18, Wide-ResNet50-2 and EfficientNet-B5. Repo implements first two; Bottle plan uses WR50-Rd550. Explicit IMAGENET1K_V1 preserves legacy pretrained=True weights. Network frozen/eval, no gradient training. |
| Feature layers | First three residual groups: layer1[-1], layer2[-1], layer3[-1]. Repo combines aligned features to 56x56 and 1792 channels for WR50. Preserved. Synthetic check confirms spatial replication alignment. |
| Dimensionality reduction | Paper random channel selection (not PCA); WR50 550, R18 100. Repo matches. Seed1024 and batch32 are repo settings, not claimed paper-mandated values. Indices saved once for both splits. |
| Gaussian | Per-location mean and full sample covariance (N-1 denominator) + epsilon I, epsilon=.01. Repo matches paper, including cross-layer covariance. Mahalanobis score preserved. |
| Image preprocessing | Paper256x256 then central224x224, bicubic. Repo Image.ANTIALIAS corresponds to Lanczos, not bicubic and is removed in current Pillow. Controlled loader uses explicit bicubic and ImageNet normalization. |
| Map preprocessing | Paper bicubic and sigma4 Gaussian; repo bilinear and sigma4. Controlled route changes interpolation to bicubic and keeps sigma4. align_corners=False and scipy reflect boundary defaults explicitly retained; exact historic library details unspecified. |
| Masks | Nearest-neighbor256 then central224, binary; cannot interpolate class labels bicubically. Evaluation stays at224 crop resolution. |
| Image score | Max of final upsampled/smoothed anomaly map, matching repo convention and paper max-map rule. No score rule search. |
| Threshold | Repo maximizes F1 on test masks for visualization. Completely removed from controlled entry point. Requested metrics are threshold-independent; no binary masks, F1 or fixed-threshold specificity reported. No validation holdout needed solely for these metrics. If a deployable threshold is later requested, agree a separate normal-only calibration protocol first. |
| Normalization | Repo uses shared per-category test min/max. Removed: metrics use raw maps. A nondegenerate shared affine normalization preserves ranking and exact AUPRO but is unnecessary/test-dependent. No per-image normalization. |
| AUPRO | Missing in repo. Added exact score sweep, ties grouped, 8-connected equal-weight regions, all test background incl normal images, FPR0-.30 with interpolated endpoint and integral/.30. Definition follows paper; connectivity and numerical integration are explicit local choices. |
| Caches | Fresh output required. Existing Gaussian caches cannot be silently reused. Original source files remain unchanged. |

## Proposed Bottle run after approval
1. Keep all209 train/good images for Gaussian fitting. Test20 good +63 anomalies. Normal=0, anomaly=1. Hash-separated splits.
2. WR50 ImageNet1K_V1; layer1/2/3; random550 of1792 features; seed1024; batch32; workers0. No augmentation.
3. Small train-only feature/numerical smoke after approval, then final full fitting with the same fixed configuration. Do not use test scores for hyperparameters/backbone/seed selection.
4. Evaluate83 images once using fixed preprocessing. Keep maps and masks in memory for metrics only. Store selected channels, distribution parameters, source/data hashes, per-image scalar scores, metrics and one fixed-seed comparison panel; no full map archive.
5. Report Image AUROC, Pixel AUROC, AUPRO@.30. No15-category launch.
6. Times and floating-point differences are implementation dependent; one seed is not a multi-seed paper reproduction guarantee. Other existing baselines use different resolutions/protocols; do not claim matched conditions solely because metric names match.

## Changes
- Compatibility: modern Pillow interpolation enum; explicit Torchvision weight enum; preserve image dimension with squeeze(1); Windows workers0.
- Paper alignment: bicubic image/map resizing; other PaDiM mathematics retained.
- Independent evaluation: remove test-selected threshold and test-derived score scaling; add exact AUPRO, fresh output, audit manifests and saved metrics.
- Operational: guarded --execute flag. Invoking without it prints prepared-only and exits before model initialization.

## Completed checks (no experiment executed)
- Python syntax.
- Perfect synthetic maps yield Image/Pixel AUROC and AUPRO1.
- Constant maps yield Image/Pixel AUROC.5.
- Exact AUPRO matches exhaustive threshold sweep with tied scores.
- Feature concatenation matches aligned nearest spatial replication on synthetic tensors.
- Real Bottle file audit:209 train,83 test;20 good/63 defect; no train/test byte-hash overlap.
- Dataset transform outputs image3x224x224 and binary mask1x224x224.
- Guarded invocation exits without model creation or fitting.

Entry point: controlled_bottle/run_bottle.py
Protocol: controlled_bottle/protocol.json
Only AFTER explicit user approval:
.venv_dfr/Scripts/python.exe -B PaDiM-Anomaly-Detection-Localization-master-main/controlled_bottle/run_bottle.py --execute

## Visualization update
Seed1024 chooses 2 normal and 1 image per defect folder BEFORE model predictions. One 5x3 panel: input / ground truth mask / heatmap overlay, matching repo comparison style. No test-optimal F1, no thresholded prediction boundaries. Raw maps are not written individually or in bulk. All83 test images remain in metric computation. Panel and actual performance results will be generated only after experiment approval.
