# DFR — Deep Feature Reconstruction

[Project overview](../../README.md) · [Method index](../README.md) · [Research findings](../../docs/research_findings.md)

**Status: completed, 15 MVTec AD categories.** Results describe the recorded local implementation, not an exact paper reproduction.

## How it works

DFR learns to reconstruct normal pretrained features rather than RGB pixels. Feature reconstruction error becomes a spatial anomaly score. A compact feature autoencoder can still respond to normal texture variation or image boundaries, so a high AUROC does not guarantee a clean thresholded mask.

**Architecture:** Image → frozen VGG19 multiscale features → spatial alignment and regional averaging → 1×1 convolutional feature autoencoder → feature error map → upsampling.

## Our experiment

RGB 256×256; first 12 VGG19 ReLU outputs; bilinear alignment; 4×4 averaging/stride 4; BN-enabled FeatCAE; PCA 90% from the first 10 sorted normal training images; batch 4; Adam 1e-4; 150 epochs; seed 0. The final checkpoint is used. Bottle is reused from its separately completed run. Repo settings differ from the paper.

[Configuration evidence](../../experiments/configurations.json) · [Category results](../../results/benchmarks/dfr/dfr_all_results.csv)

## Results and what we learned

All 15 categories: Image AUROC 93.85%, Pixel AUROC 95.25%, repo PRO/AUPRO 89.94%. The separate shared evaluator reports Pixel AUROC 95.2474%, AP 47.5443%, AUPRO 89.6401%; do not substitute these into the original table.

False positives include both expansion around defects and separate distant regions. The raw64 audit found above-threshold support for the examined distant positives before final upsampling: interpolation alone does not explain them. Nearest alignment reduced predicted area but also recall and ranking quality. Native128 improved boundary/F1 measures in the pilot while increasing memory and time; it was not a universal improvement.

## Visual evidence

![Recorded DFR evidence](../../assets/figures/dfr_native64_native128.png)

Leather color/000. Resolution ablation, with paired scale and binary masks. [All-method gallery](../../assets/GALLERY.md) and [figure provenance](../../assets/figures/SOURCES.csv) identify the original source, selection and scale. This illustration is not a matched-protocol comparison.

## Upstream and local work

[Original implementation](https://github.com/YoungGod/DFR) · [Paper](https://arxiv.org/abs/2012.07122) · [Recorded provenance](upstream.json) · [Local changes](CHANGES.md).

Upstream model design is credited to its authors. Our contribution is the documented execution, adaptation and/or evaluation analysis described above. See the [protocol definitions](../../docs/evaluation_protocols.md) before cross-method comparisons. Relocated training/inference has not been verified.
