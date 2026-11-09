# RD++
Official reference: https://github.com/tientrandinh/Revisiting-Reverse-Distillation (commit 7f2ceb7c87e602617b8600e1a498f7ef7f5247d6). Preserved MIT license accompanies source.
MVTec AD T1, 15 independently trained category models, batch 16, seed 111, WideResNet50 teacher, original projection/bottleneck/decoder and losses. Official category epochs: carpet/leather 10; grid/tile 260; wood 100; bottle/pill 200; cable 240; capsule/transistor/zipper 300; hazelnut/metal_nut 160; screw/toothbrush 280.
Fixed-final selection replaces upstream test-selected checkpoints; no per-epoch test evaluation. Original two-minibatch unscaled accumulation retained. Windows compatibility/dependency adjustments are recorded with the source; original runtime bitwise equivalence is not claimed.
OpenCV RGB linear256 preprocessing, summed feature cosine distances, Gaussian sigma4, maximum image score preserved. [All category results](../../results/benchmarks/rdpp/results.csv), [macro metrics](../../results/benchmarks/rdpp/macro.json).
Macro Image AUROC/AP 98.569113/99.430984%; Pixel AUROC/AP 98.069895/59.599740%; AU-PRO0.30 94.327889%. Original-mask and upstream approximate PRO remain separately labeled in category JSON.

## Reproducibility and limitations

[Recorded workspace and publication manifest](../../reproducibility/rdpp/README.md). Code/configuration/results are evidence snapshots, not a packaged one-command application. Weights, datasets, prediction archives, caches and virtual environments are not published. Retained checkpoint hashes identify local artifacts. No training or inference was run for this publication.

Final evaluation: 2026-11-09, 1725 held-out images per method. Common256 PIL nearest masks, all normal pixels included; exact normalized AU-PRO to FPR0.30. No recalibration, threshold fitting or test-based model selection. Same image/mask IDs verified across RD++ and DeSTSeg. Single seed111; SD Not recorded. RTX A4000, batch1 FP32, three warmups. RD++ model latency excludes map postprocessing; DeSTSeg model latency includes interpolation/top100. Pipeline timing includes compressed export, excludes GT/metrics/hashing; not pure production latency. Training timings also have different scopes. See category resource/provenance records.
