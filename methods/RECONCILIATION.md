# Method and experiment reconciliation

## Final publication status - 23 October 2026 (Europe/Istanbul)

SuperSimpleNet completed all 15 MVTec AD categories with fixed final checkpoints. The controller completion timestamp is 2026-10-23T04:14:44+03:00. Every category has a complete status, a 300-epoch training history, a recorded final-checkpoint hash and final.pt. Category metrics match results.csv; equal-category macro metrics match supersimplenet_mean.json.

| Image AUROC (%) | Pixel AUROC (%) | AUPRO@0.30 (%) |
|---:|---:|---:|
| 98.305 | 97.521 | 91.193 |

The individual guide and concise method index now report final results. Intermediate SuperSimpleNet progress snapshots have been removed from publication prose; original experiment records and RECONCILIATION_EVIDENCE.csv remain untouched. The latter describes the earlier audit, not the final capture.

| Final evidence source (workspace-relative) | SHA-256 |
|---|---|
| SuperSimpleNet_MVTec_results_20261020/results.csv | 082052ceff29253258fee436edc18c1919501113571e3471ce7f96a9d9bce2bf |
| reports/simplenet_supersimplenet_20261020/supersimplenet_mean.json | 06fa599815bfad22082c63fa6ad70f7e540e8a5e841fd754070d6819ecc6a25d |
| SimpleNet_SuperSimpleNet_setup_20261020/status.json | 7eb3cbc06ce2f1690568349488c2961a8e4198c821a4cb15a31019e2216080ef |

No experiment or inference was rerun. Source paths outside github_release are evidence identifiers, not bundled GitHub links. Single-seed, metric-definition and hardware-counter limitations remain documented in the method guide.

## Decision rules and scope

Direct execution evidence means surviving evaluation tables, training histories, diagnostic results or failed-run records. A source checkout, literature row, downloaded pretrained weight or planned configuration alone is not execution evidence. Aggregate CSV rows (mean, standard deviation and similar rows) were excluded when checking 15-category coverage.

The review inventories readable local records across all top-level experiment/source/report folders, excluding package environments, caches and raw image datasets from content searches. Duplicate release copies, report snapshots and backup third_party files are not independent experiments. Paper-provided results under SuperSimpleNet-main/paper_results are external references. No inference/training was rerun, binary checkpoint contents were not re-evaluated, and missing/deleted or unavailable compressed-only evidence is not certified.

Twelve method-guide groups have surviving direct execution records; the combined AE-DPAE guide covers two separately executed controlled models. A distinct historical repository AE baseline has only indirect cleanup evidence. AnomalyNCD remains source-only. This is a record-based inventory, not proof of scientific correctness of every experiment.

## Methods with surviving direct execution evidence

| Method | Execution status | Primary workspace-relative evidence | Guide action |
|---|---|---|---|
| PatchCore | 15/15 category evaluations | patchcore-inspection-main/patchcore_final_results.csv | [Existing guide preserved](patchcore/README.md) |
| DFR | 15/15 category evaluations | DFR_results_all/dfr_all_results.csv | [Existing guide preserved](dfr/README.md) |
| RD4AD | 15/15 category evaluations | RD4AD_results_all/all15_results.csv | [Existing guide preserved](rd4ad/README.md) |
| PaDiM | 15/15 category evaluations | PaDiM_MVTec_all15_20261012/results.csv | [Existing guide preserved](padim/README.md) |
| AnoGAN adaptation | 15/15 category evaluations | AnoGAN_MVTec2019_all15_20261008/results.csv | [Existing guide preserved](anogan/README.md) |
| GANomaly | 15/15 category evaluations | GANomaly_official_flow_20261006/results.csv | [Existing guide preserved](ganomaly/README.md) |
| Skip-GANomaly | 15/15 category evaluations | skip-ganomaly-master/skip-ganomaly-master/skipganomaly_mvtec15_results.csv | [Existing guide preserved](skip_ganomaly/README.md) |
| PatchSVDD | 15/15 category evaluations; supersedes old partial snapshot | PatchSVDD_MVTec_all15_20261012/results.csv | [Existing guide preserved](patchsvdd/README.md) |
| SimpleNet | 15/15 category evaluations; newer CUDA run complete, older CPU run interrupted | SimpleNet_MVTec_results_20261020/results.csv | [Existing guide preserved](simplenet/README.md) |
| DRAEM | 15/15 category evaluations | DRAEM_MVTec_all15_20261015/results.csv | Created [guide](draem/README.md) |
| SuperSimpleNet | 15/15 completed; fixed final checkpoints | SuperSimpleNet_MVTec_results_20261020/results.csv, per-category metrics/status and training.csv | [Final guide](supersimplenet/README.md) |
| AE_L2_controlled | Nonfinite smoke failure; 10-batch normalized diagnostic stable; one additional AE epoch history | defect-detection-main/runs/smoke_2epoch_20260916_082748/summary.json and runs/normalized_l2_20260916_084513 | [Covered by existing combined guide](ae_dpae/README.md) |
| DPAE_controlled | Nonfinite smoke failure; 10-batch normalized diagnostic stable; no completed full comparison | Same dated smoke/normalized diagnostic directories, DPAE_controlled subfolders | [Covered by existing combined guide](ae_dpae/README.md) |


## Previously missing guides and documentation changes

- DRAEM was absent despite a complete 15-category final700 training/evaluation record and a third-party source entry. Created draem/README.md with architecture, actual protocol, verified category metrics, limitations and attribution.
- repo_AE_baseline was not separately documented. Created repo_ae_baseline/README.md to preserve its historical identity while explicitly marking the original experiment contents as unavailable. It is not counted among direct-evidence completed benchmarks.
- methods/README.md now presents the current status of every executed method and separates historical/insufficient evidence from source-only work. Its previous content is retained inside a labeled historical section.
- PatchSVDD and SimpleNet guides were subsequently updated to completed benchmarks while retaining their distinct historical runs. SuperSimpleNet now presents final results without intermediate progress notes.
- Added this report and RECONCILIATION_EVIDENCE.csv. No research results, source catalog, licenses, scripts or root manifests were edited.

## Additional executed variants, diagnostics and failed experiments

| Parent method | Evidence and classification |
|---|---|
| DFR | Bottle pilot; 15-category run; Grid/Leather/Wood nearest-alignment and native128 experiments have complete status/report records. Threshold and spatial false-positive analyses are evaluations of existing outputs, not new methods. |
| DFR / RD4AD | DFR_RD4AD_common_evaluation/status.json records 30 model-category evaluations across 15 categories. This is shared evaluation, not a separate architecture. |
| RD4AD | Bottle pilot and holdout calibration for Leather, Grid and Transistor have completed records. Calibration/retraining variants remain under RD4AD. |
| PaDiM | Bottle pilot, 15-category fitting/evaluation, and completed Bottle/Screw holdout calibration records. Gaussian fitting counts as local execution without requiring gradient training. |
| AnoGAN | Bottle smoke/pilot, 15-category training/evaluation, recovery archives and a later max-map image-AUROC evaluation. The latter is not the original scoring protocol. |
| GANomaly | Official-flow 25-epoch category records; archived capsule ablations, reset-Adam audit and diagnostic outputs. Test-dependent checkpoint selection and BatchNorm behavior are not repaired by relabeling the runs. |
| Skip-GANomaly | 15-category final evaluations plus checkpoint/graph/BatchNorm audits; audit helpers and alternate source files are not additional methods. |
| PatchSVDD | Bottle pilot reused in main run; completed Carpet lambda=0.1 and 0.01 diagnostics; pretrained, random/untrained and numerical root-cause evaluations under PatchSVDD_rootcause_20261015. Poor/collapsed runs count as executed diagnostics, not successful reproduction. |
| SimpleNet | Historical CPU smoke and stopped Screw meta-epoch 8 run (0 completed categories); newer fixed-final CUDA experiment completes 15 categories. Both remain distinct source/protocol variants within SimpleNet. |
| SuperSimpleNet | Passed train-only smoke and completed 15-category fixed-final-checkpoint training/evaluation. |
| DRAEM | Passed train-only smoke, completed downloaded-pretrained-weight evaluation/original-evaluator comparison, and separately completed local final700 training on all categories. Pretrained evaluation is not local training of those downloaded weights. |
| Controlled AE / DPAE | Both literal-sum smoke runs stopped with nonfinite state in epoch 1; normalized-loss 10-batch probes completed. AE has one epoch in a longer diagnostic directory, not proof of completed two-epoch training or final ROC evaluation. |

## Historical and insufficient evidence

The repository AE baseline has documented old Bottle outputs in cleanup_inventory.csv (checkpoint, run.json, metrics.txt, ROC/PR plots and two dated run folders), and cleanup_result.json confirms deletion. Those records establish historical output provenance only. The checkpoint, configuration and metric contents cannot be independently verified now; no completion count or metric is assigned.

AnomalyNCD has source/configuration and an existing guide but no established local execution log or metric record. It is not added to the executed-method list. Literature recommendations such as RD++, DeSTSeg, other survey methods and dataset-only folders likewise do not establish execution. Imported upstream results, planned scripts, architecture modules, generic attention blocks and model/checkpoint directory names alone do not qualify.

The old AE-DPAE pretraining report says the main experiment did not start. Later diagnostic files demonstrate limited execution, not a completed approved main comparison; both observations are retained. The longer normalized AE diagnostic's one-epoch history does not resolve all stale status metadata.

Source revision unknowns and licensing gaps remain as recorded in third_party. No missing source identity, successful reproduction, missing metric or hardware setting is inferred.

## Evidence and validation

[RECONCILIATION_EVIDENCE.csv](RECONCILIATION_EVIDENCE.csv) contains the inspected workspace-relative evidence paths, roles, sizes, hashes and capture timestamp. These paths are provenance identifiers, not portable GitHub links; most full experiment outputs remain outside the publication folder. Mutable active-run status files may naturally change after capture.

Checks performed: unique 15-category coverage of the eleven complete benchmark tables; DRAEM, newer SimpleNet and PatchSVDD CSV/category-metric consistency; their completion statuses; DRAEM 700-epoch summaries and macro arithmetic; all 15 SuperSimpleNet category statuses, 300-epoch histories, CSV/metric consistency and macro arithmetic; and existence of all local links introduced here. Checks do not certify statistical validity or binary checkpoint contents.

The root experiment registry and older result summaries remain historical, and are not updated in this methods-only task. A lack of evidence is reported explicitly rather than filled with inferred results.


## Preserved earlier index

The following earlier index is retained for other methods; the obsolete SuperSimpleNet progress entry has been removed. Its completion labels are historical; use the current inventory above and reconciliation report for this review's status.

<details>
<summary>Earlier method index (historical snapshot)</summary>

# Method guide

[Project overview](../README.md) · [Literature inventory](../literature/methods.csv) · [Experiment registry](../experiments/registry.csv)

## Completed 15-category studies

- [PatchCore](patchcore/README.md)
- [DFR — Deep Feature Reconstruction](dfr/README.md)
- [RD4AD — Reverse Distillation](rd4ad/README.md)
- [PaDiM — Patch Distribution Modeling](padim/README.md)
- [AnoGAN — MVTec2019-based adaptation](anogan/README.md)
- [GANomaly](ganomaly/README.md)
- [Skip-GANomaly](skip_ganomaly/README.md)

## Separate, non-final work

- [PatchSVDD](patchsvdd/README.md)
- [SimpleNet](simplenet/README.md)

- [Controlled AE–DPAE](ae_dpae/README.md)
- [AnomalyNCD](anomalyncd/README.md)

Other literature entries are research references rather than demonstrated local implementations. Each guide separates upstream design, our recorded configuration, results and limitations.

</details>
