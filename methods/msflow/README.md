# MSFlow

Official implementation: [cool-xuan/msflow](https://github.com/cool-xuan/msflow/tree/e23975598bc7eb6f74604739d8f04d1519d704da). [Paper](https://arxiv.org/abs/2308.15300). Code belongs to its original authors and is not redistributed here.

Completed15/15 MVTec AD categories,1725 test images, T1,seed9826. ImageNet V1 reflect-padding WideResNet50 teacher; parallel flow blocks2/5/8 plus fusion.100epochs,Adam1e-4,batch8,FP32;512 input except transistor256. Fixed-final only with no periodic test evaluation or test-selected checkpoint. Training queue wall9.979h on RTX A4000.

Batch8 follows effective official CLI but differs from paper16. Per-category seed reset,Windows workers0 and isolated torch2.5.1/FrEIA0.2 are recorded differences; exact paper reproduction is not claimed. Strict checkpoint/state/hash checks passed. No retraining or calibration for evaluation.

[Complete report and per-category tables](../../results/benchmarks/msflow/RESULTS.md) | [CSV](../../results/benchmarks/msflow/results.csv) | [Configuration/provenance](../../reproducibility/msflow/README.md).

Native primary image AUROC/AP: 99.2499/99.7223%; additive pixel AUROC/AP: 98.5187/61.8705%; additive exact AU-PRO0.30: 95.1120%. Native means model-output512/256, not original camera dimensions. Common256 evaluation is separately labeled. Continuous image score is mean top3% multiplicative map. Supplementary multiplicative exact AU-PRO is not mixed into primary results or claimed equivalent to upstream approximate PRO.

Single seed; uncertainty Not recorded. Different training inputs and timing scopes preclude an unqualified controlled leaderboard. Raw weights, maps, datasets, source code and environments remain local.
