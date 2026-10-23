# SuperSimpleNet

[Project overview](../../README.md) | [Method index](../README.md)

**Status: completed MVTec AD benchmark; 15/15 categories trained and evaluated.** Evidence verified on 23 October 2026 (Europe/Istanbul). The controller records completion at 2026-10-23T04:14:44+03:00.

## Principle and architecture

SuperSimpleNet learns to distinguish normal features from synthetic feature anomalies. Our recorded runtime uses a frozen ImageNet V1 WideResNet50-2 backbone, taking layer2 and layer3 features. Both feature maps are bilinearly resized to twice the larger feature grid, concatenated, and aggregated with a 3x3 average filter. A trainable 1x1 feature adaptor and separate segmentation and image-classification heads follow.

Gaussian feature perturbations (standard deviation 0.015) and Perlin masks (threshold 0.2) generate synthetic anomalies and training targets. The actual run uses only normal training images; real defective images and pixel ground truth are used for evaluation. The broader method supports other supervision regimes, but they were not tested here.

The selected configuration is the later JIMS extension: with adapt_cls_feat=False, segmentation uses adapted features while classification uses the original extracted features. This differs from the original ICPR configuration, which feeds adapted features to both heads. stop_grad=True detaches segmentation-map inputs to the classification branch.

### Anomaly scoring, detection and localization

The segmentation head emits spatial anomaly logits. For localization, the model upsamples these to 256x256 and applies Gaussian smoothing (sigma 4); the local evaluator then applies sigmoid. Higher map values indicate stronger anomaly evidence at each pixel.

The classification head combines features with the segmentation map, pools convolutional features and map values using maximum and average pooling, and produces a separate image logit. Its sigmoid is the image-level anomaly score; it is not simply the maximum of the displayed pixel map. Image AUROC measures ranking of normal versus defective images; pixel AUROC and AUPRO measure localization ranking.

No operating threshold is fitted. These scores do not establish calibrated defect probabilities, production pass/fail decisions, or thresholded segmentation accuracy. Per-image min/max scaling is used only for visualization.

## Recorded setup

| Item | Local evidence |
|---|---|
| Dataset | MVTec AD, separate category-wise models; original train/good only; no validation holdout |
| Input | 256x256 resize, ImageNet normalization; nearest-neighbor mask resize |
| Backbone | Frozen wide_resnet50_2, ImageNet V1 weights, layer2 + layer3 |
| Training | 300 epochs, batch 32, seed 42; CUDA; 8 data-loader workers |
| Optimizer | AdamW; adaptor LR 1e-4 / weight decay 0.01; both heads LR 2e-4 / weight decay 1e-5 |
| Schedule | LR multiplied by 0.4 at epochs 240 and 270 |
| Loss | Segmentation truncated L1 plus focal loss; image-classification focal loss |
| Other settings | No flips; no gradient clipping; synthetic noise and Perlin masks enabled |
| Checkpoint | Fixed final checkpoint; intermediate test evaluation removed |
| Recorded environment | Python 3.11.9, torch 2.1.0+cu118, torchvision 0.16.0+cu118, NumPy 1.26.0, NVIDIA RTX A4000 |

Training requires the upstream runtime, ImageNet pretrained weights, normal training images and the recorded dependencies. Our runner requires CUDA; this is not a claim that the method intrinsically requires that specific GPU. Pixel masks and defective test examples are needed for the reported evaluation, not normal-only fitting. The preflight audit records 15 categories, 3,629 training and 1,725 test images, disjoint image hashes and valid masks. The train-only smoke test passed before the full run. Completion is independently supported by all 15 category statuses and 300-epoch training histories.

[Source catalog](../../third_party/SOURCES.csv), [recorded versions](../../third_party/records/simplenet_supersimplenet_20261020/versions.json), [local runner](../../third_party/local_changes/simplenet_supersimplenet_20261020/experiment.py), [dependency record](../../third_party/local_changes/simplenet_supersimplenet_20261020/pip_freeze.txt), and [historical reproduction notes](../../third_party/records/simplenet_supersimplenet_20261020/REPRODUCE.md).

## Outcomes and limitations

The following are completed category evaluations, in percent, copied from the local results.csv and checked against each category's metrics.json. All 15 have a complete status, a training summary recording a final-checkpoint hash, and an existing final.pt. Each training.csv contains epochs 1 through 300. Checkpoint hashes and predictions were not recomputed during this documentation task.

| Category | Image AUROC (%) | Pixel AUROC (%) | AUPRO@0.30 (%) |
|---|---:|---:|---:|
| bottle | 100.000 | 97.635 | 89.710 |
| cable | 97.395 | 97.140 | 87.602 |
| capsule | 98.644 | 98.711 | 92.503 |
| carpet | 97.432 | 98.235 | 92.430 |
| grid | 99.916 | 97.967 | 93.095 |
| hazelnut | 100.000 | 98.039 | 94.598 |
| leather | 100.000 | 98.855 | 96.997 |
| metal_nut | 99.413 | 98.483 | 91.009 |
| pill | 98.391 | 98.571 | 94.844 |
| screw | 91.679 | 99.040 | 95.030 |
| tile | 98.954 | 95.325 | 84.276 |
| toothbrush | 94.167 | 98.457 | 86.119 |
| transistor | 100.000 | 95.551 | 91.414 |
| wood | 99.211 | 92.637 | 85.330 |
| zipper | 99.370 | 98.172 | 92.941 |
| **Macro average (15 categories)** | **98.305** | **97.521** | **91.193** |

The macro average gives equal weight to all 15 categories and matches the recorded supersimplenet_mean.json. Completed local training/evaluation does not establish successful paper reproduction.

Evaluation uses the fixed final checkpoint without intermediate test-driven selection. The evaluator computes image AUROC across test image labels, pixel AUROC across normal and anomalous test pixels, and AUPRO@0.30 using 8-connected components, equal region weighting, a global background-FPR sweep over unique scores, interpolation at 0.30, and area normalization by 0.30. The [local metric implementation](../../third_party/local_changes/simplenet_supersimplenet_20261020/evaluation_metrics.py) can differ numerically from upstream approximate AUPRO. Sigmoid is retained; global min/max score normalization is omitted.

This is one seed covering all 15 categories. It does not establish uncertainty, statistical superiority, or equivalence to the upstream multi-run paper results. Preprocessing and scoring differ from other method guides; those historical results are not a matched-protocol comparison. For example, Screw has much higher pixel AUROC than image AUROC, illustrating that localization ranking does not guarantee reliable image-level detection.

Some raw resource counters are inconsistent with a straightforward physical-GPU interpretation: Bottle records peak allocated memory above the environment's 16 GiB GPU capacity. These measurements are not used here to claim hardware requirements or benchmark memory/speed superiority.

### Evidence locations and snapshot scope

The following are original workspace-relative records outside github_release; they are not copied into this method guide or promised as published GitHub links:

- SuperSimpleNet_MVTec_results_20261020/results.csv and each completed category's metrics.json.
- SuperSimpleNet_MVTec_training_20261020/<category>/config.json, environment.json, optimizer_config.json, status.json and training_summary.json.
- SimpleNet_SuperSimpleNet_setup_20261020/preflight/audit.json and preflight/supersimplenet/smoke.json.
- reports/simplenet_supersimplenet_20261020/protocol_and_papers.md.
- SimpleNet_SuperSimpleNet_setup_20261020/runtime/supersimplenet/model/feature_extractor.py and model/supersimplenet.py, inspected for the architecture and scoring description.

The final table's source CSV SHA-256 is 082052ceff29253258fee436edc18c1919501113571e3471ce7f96a9d9bce2bf. The representative Bottle configuration SHA-256 is 10d718079292b3cebc5f608a9adbe09148bf726523e7247c4e44206c8b6696fc. These identify the inspected records, not an independent re-evaluation. The final result CSV, category completion statuses and full training histories were checked together. The recorded macro summary is reports/simplenet_supersimplenet_20261020/supersimplenet_mean.json; its SHA-256 is 06fa599815bfad22082c63fa6ad70f7e540e8a5e841fd754070d6819ecc6a25d. Root result tables remain unchanged.

## Sources and local work

Original paper: [SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/abs/2408.03143) (ICPR 2024). The implemented extension is described in [No Label Left Behind: A Unified Surface Defect Detection Model for all Supervision Regimes](https://arxiv.org/abs/2508.19060). Titles and identifiers are supported by locally retained paper texts and the local protocol report.

[Original GitHub repository](https://github.com/blaz-r/SuperSimpleNet) | [Recorded provenance](../../third_party/records/supersimplenet/upstream.json) | [MIT license](../../third_party/licenses/supersimplenet/LICENSE).

Recorded commit: 98ab4d5fbdcdef528fafbc42e4b5ee15f08f5a7d. It identifies the recorded upstream revision, not a claim that all local files are pristine upstream source.

The [preserved training patch](../../third_party/patches/supersimplenet/supersimplenet_training.patch) initializes the result container, rejects non-finite loss, removes intermediate test evaluation/logging and adds an epoch callback. The local runner guards against test-loader access during training, saves recovery checkpoints, saves/reloads the final model, and performs final evaluation. Recovery checkpoints are not selected by test scores. Architecture, losses, optimizer steps and scheduler are retained by this patch.

[Source status](../../third_party/records/supersimplenet/SOURCE_STATUS.md) and [reconstruction limits](../../third_party/RECONSTRUCTION_LIMITS.md) distinguish preserved evidence from unverified reconstruction.

No new experiment was run for this documentation. Relocated execution is unverified.
