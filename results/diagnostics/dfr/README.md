# DFR failure analysis

The [common evaluation](../../common_evaluation/report.md) demonstrates that ranking and threshold transfer differ. [Nearest alignment](../../ablations/dfr_nearest/comparison.csv) and [native-resolution](../../ablations/dfr_native128/comparison.csv) tables retain pixel precision, recall, FPR, predicted/GT area and distant false-positive findings for Leather, Grid and Wood. Spatial illustrations are in the [results guide](../../README.md).

Nearest retains the baseline latent dimensions; native128 changes kernel/stride and recomputes PCA. The old nearest manifest's generic PCA wording is inconsistent with saved nearest configurations. These exploratory analyses followed test inspection; they do not define an independently tuned new benchmark.
