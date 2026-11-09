# efficientad: experiment documentation and provenance

This folder retains configuration, environment, plans, failure records and provenance for the reported experiments. It does not distribute implementation code, training/evaluation scripts or code patches. Method implementations belong to their original authors.

Reference implementation: [upstream repository](https://github.com/open-edge-platform/anomalib), pinned revision [592aafc373091dda8206510a0853eb11c8e4099d](https://github.com/open-edge-platform/anomalib/tree/592aafc373091dda8206510a0853eb11c8e4099d). EfficientAD uses Anomalib, a third-party implementation, rather than the paper authors' release.

workspace/ preserves the original relative paths of selected non-code records. Archived documents may mention local source files and patches that are intentionally not published here. These records are historical evidence, not execution instructions for this repository. See the [method guide](../../methods/efficientad/README.md) for results and compatibility limits.

manifest.json lists only retained published records, with original and published SHA256 hashes. ${PROJECT_ROOT} and ${USER_HOME} replace local absolute prefixes. Checkpoints, datasets, predictions and environments are not bundled. Training-only records preserve their original status; final evaluations are separate under results/benchmarks/. No protocol, experiment settings or numeric results were changed by this publication cleanup.
