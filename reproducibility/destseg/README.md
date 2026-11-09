# destseg: experiment documentation and provenance

This folder retains configuration, environment, plans, failure records and provenance for the reported experiments. It does not distribute implementation code, training/evaluation scripts or code patches. Method implementations belong to their original authors.

Reference implementation: [upstream repository](https://github.com/apple-aiml-research/ml-destseg), pinned revision [f168ba576e0a917ae5e037b677e7d37b741ea3d8](https://github.com/apple-aiml-research/ml-destseg/tree/f168ba576e0a917ae5e037b677e7d37b741ea3d8).

workspace/ preserves the original relative paths of selected non-code records. Archived documents may mention local source files and patches that are intentionally not published here. These records are historical evidence, not execution instructions for this repository. See the [method guide](../../methods/destseg/README.md) for results and compatibility limits.

manifest.json lists only retained published records, with original and published SHA256 hashes. ${PROJECT_ROOT} and ${USER_HOME} replace local absolute prefixes. Checkpoints, datasets, predictions and environments are not bundled. Training-only records preserve their original status; final evaluations are separate under results/benchmarks/. No protocol, experiment settings or numeric results were changed by this publication cleanup.
