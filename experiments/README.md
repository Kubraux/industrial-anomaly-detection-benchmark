# Experiments

Publication location: experiments/ at the GitHub repository root. The local staging folder is experiments_simplified/.

Evidence snapshot: 23 October 2026. This publication subset covers the eleven completed MVTec AD method benchmarks and significant ablations, calibration studies, evaluation variants and failure investigations. Completion describes local execution, not reproduction of paper results.

- [registry.csv](registry.csv): one row per benchmark or substantial study; status, settings, result location and limitations.
- [CONFIGURATIONS.md](CONFIGURATIONS.md) and [configurations.json](configurations.json): verified training summaries and consolidated original settings, including category-specific differences.
- [PROTOCOLS.md](PROTOCOLS.md): scoring, evaluation, model selection and leakage limitations.

Methods and architectures are documented in [methods/](../methods/README.md). Metrics remain in their referenced result files. Registry and configuration source paths are relative to the repository root (locally, github_release); workspace_external paths begin ../ and will not resolve when publishing github_release alone. Those result dependencies still need a separate publication decision. No standalone reproducibility claim is made.

Coverage counts refer to the stated unit, not independent repetitions. Blank coverage is unknown or not meaningful for a grouped study. Diagnostic groups may contain several subcases. Test-selected GANomaly and historical SimpleNet runs, oracle test thresholds, prior test inspection and differing AUPRO definitions remain explicitly disclosed.

## Configuration portability

Machine-specific workspace paths in configurations.json use the placeholder WORKSPACE_ROOT (in angle brackets) with forward slashes. Resolve it to your own research workspace when interpreting a path. Relative suffixes, category-specific settings and all experiment parameters are preserved. Source references and sha256 values are unchanged: each hash identifies the original, unmodified archived file, not the sanitized publication representation. Original absolute paths remain in the research archive. These snapshots are documentation, not ready-to-run configurations.

## Archive and exclusions

Publish this subset as experiments/. Keep the original detailed archive and unmodified research configurations outside GitHub; they are not part of the published tree. The local detailed archive remains unchanged. No directories were moved by this preparation.

Excluded from this subset: per-experiment evidence JSON, repeated method pages, bulk evidence/hash inventories, validation scripts/reports, the historical registry copy, unexecuted configuration templates, routine successful smoke tests, duplicate pilots and individual low-level probes. Significant failed AE/DPAE diagnostics and historical stopped runs remain summarized in retained studies. AnoGAN recovery attempts/CUDA checks and individual PatchSVDD probes remain in the detailed backup. The deleted historical AE baseline has only indirect cleanup evidence, so it is not a verified benchmark here; its limitations remain in methods/ and the detailed archive.

All research evidence, original configurations, results and the detailed archive remain untouched. To add a study, use a new stable registry ID, link verified results, record actual settings and distinguish execution from planned budgets.
