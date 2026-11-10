# MSFlow — final MVTec AD evaluation (10 November 2026)

Locally measured fixed-final T1 results, all15 categories/1725 test images. Seed9826; no retraining, recalibration, test-based selection or historical result replacement. Percentages; equal-category macro. Native means model-output512x512 (transistor256), not original camera resolution.

| Evaluation grid | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AU-PRO0.30 |
|---|---:|---:|---:|---:|---:|
| native | 99.2499 | 99.7223 | 98.5187 | 61.8705 | 95.1120 |
| common256 | 99.2499 | 99.7223 | 98.5188 | 61.8864 | 95.0075 |

## Per-category native (%)

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AU-PRO0.30 |
|---|---:|---:|---:|---:|---:|
| bottle | 100.0000 | 100.0000 | 98.4167 | 75.0529 | 93.8127 |
| cable | 98.6882 | 99.1992 | 98.3810 | 69.9976 | 93.7669 |
| capsule | 98.9629 | 99.7539 | 99.1746 | 51.5948 | 97.6747 |
| carpet | 98.4751 | 99.5429 | 98.6726 | 59.2556 | 95.2261 |
| grid | 99.4152 | 99.8081 | 99.4079 | 54.1461 | 97.2787 |
| hazelnut | 100.0000 | 100.0000 | 98.6998 | 55.0361 | 96.9308 |
| leather | 99.9660 | 99.9883 | 99.3801 | 49.6419 | 98.2358 |
| metal_nut | 100.0000 | 100.0000 | 99.2528 | 92.5135 | 94.3707 |
| pill | 98.8816 | 99.7964 | 98.7338 | 75.7285 | 95.5400 |
| screw | 95.7983 | 98.2986 | 99.3351 | 36.8998 | 96.5873 |
| tile | 99.7835 | 99.9134 | 98.1272 | 78.1029 | 94.3914 |
| toothbrush | 100.0000 | 100.0000 | 98.7313 | 40.8480 | 90.5051 |
| transistor | 99.9167 | 99.8810 | 96.1712 | 60.1828 | 91.2204 |
| wood | 99.1228 | 99.7197 | 96.2018 | 67.8233 | 94.5291 |
| zipper | 99.7374 | 99.9325 | 99.0943 | 61.2329 | 96.6102 |

## Per-category common256 (%)

| Category | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AU-PRO0.30 |
|---|---:|---:|---:|---:|---:|
| bottle | 100.0000 | 100.0000 | 98.4174 | 75.0651 | 93.8219 |
| cable | 98.6882 | 99.1992 | 98.3829 | 70.0293 | 93.7606 |
| capsule | 98.9629 | 99.7539 | 99.1747 | 51.5470 | 96.5759 |
| carpet | 98.4751 | 99.5429 | 98.6722 | 59.2421 | 95.3096 |
| grid | 99.4152 | 99.8081 | 99.4083 | 54.2312 | 97.2894 |
| hazelnut | 100.0000 | 100.0000 | 98.6995 | 55.0066 | 97.0466 |
| leather | 99.9660 | 99.9883 | 99.3796 | 49.6595 | 98.2414 |
| metal_nut | 100.0000 | 100.0000 | 99.2532 | 92.5270 | 94.3605 |
| pill | 98.8816 | 99.7964 | 98.7337 | 75.7156 | 95.4267 |
| screw | 95.7983 | 98.2986 | 99.3360 | 36.9921 | 96.7610 |
| tile | 99.7835 | 99.9134 | 98.1300 | 78.1313 | 94.2833 |
| toothbrush | 100.0000 | 100.0000 | 98.7329 | 40.8824 | 89.9188 |
| transistor | 99.9167 | 99.8810 | 96.1712 | 60.1828 | 91.2204 |
| wood | 99.1228 | 99.7197 | 96.1960 | 67.8083 | 94.4922 |
| zipper | 99.7374 | 99.9325 | 99.0944 | 61.2752 | 96.6043 |

## Evaluation and compatibility

Official source cool-xuan/msflow at e23975598bc7eb6f74604739d8f04d1519d704da. Reflect-padding WRN50 ImageNet V1; original architecture/losses and100epochs retained. Batch8 is the effective release CLI default; paper batch16 is not reproduced. Seed reset per category, Windows workers0, current isolated torch/FrEIA environment are documented implementation differences. Fixed-final replaces upstream per-epoch test/best selection. One seed; uncertainty Not recorded.
Original LANCZOS input preprocessing and grayscale replication retained. Image score=mean top3% of multiplicative map. Additive map used for all primary pixel AUROC/AP/AU-PRO. Common256 resizes maps with bilinear align_corners=False and original masks with PIL nearest>0; native masks follow official nearest preprocessing. Common grid does not make training input resolutions equal.
Exact common AU-PRO uses8-connected equally weighted regions, all background pixels including normal images, grouped ties and interpolated normalized FPR0.30 integration. Supplementary native multiplicative exact AU-PRO=90.9864%. This is a separate map choice fixed before testing, not a claim of reproducing the upstream approximate PRO implementation. No test-selected map or threshold.
All checkpoint/source/protocol hashes and model states unchanged. All1725 image/mask hashes match previous DeSTSeg evaluation; common masks are bitwise identical. Dry validation used only normal training images and synthetic fixtures. Published paper performance is not mixed into these tables.

## Resources

Training queue wall time=9.979h; sum category training time=9.921h. Evaluation wall=1473.48s. GPU RTX A4000; FP32/batch1,3 normal-train warmups. Median forward latency range=37.59–71.98ms across categories. Peak allocated inference VRAM=0.635GiB.
Forward timing excludes preprocessing/postprocessing. Pipeline includes decode/transfer/postprocessing/GT reads/compressed export; excludes hashing and metric computation. RSS is sampled at end, not claimed peak. No cross-method throughput ranking from different timing scopes.

[CSV](results.csv) | [Metrics](metrics.json) | [Resource records](resource_usage.json) | [Training summary](training_summary.csv) | [Integrity validation](validation_final.json) | [Prospective plan](plan.md)

Implementation code, weights, data and prediction archives remain local. Only documentation/configuration/results/provenance are selected for GitHub publication.
