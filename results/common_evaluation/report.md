# DFR / RD4AD common evaluation

Fifteen categories each, same test images and nearest256 ground truth. Exact tied-score AUPRO@0.30, 8-connected equal-weight regions and global background FPR. Threshold = training-normal pixel q99.5, method=higher, strict >, no normalization.

| Model | Pixel AUROC (%) | Exact AUPRO (%) | Pixel F1 (%) | Normal-test pixel FPR (%) |
|---|---:|---:|---:|---:|
| DFR | 95.247 | 89.640 | 41.558 | 1.494 |
| RD4AD | 97.784 | 93.680 | 15.038 | 52.560 |

[Category results](comparison.csv). Values are equal-category means, not pooled all-category confusion metrics. Higher ranking AUROC does not imply that a training-normal threshold transfers to test normals. The earlier native PRO implementations are different metrics; see [protocols](../../experiments/PROTOCOLS.md).
