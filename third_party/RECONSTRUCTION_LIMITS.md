# Reconstruction limits

Each row in [SOURCES.csv](SOURCES.csv) records what is known, not what is presumed. Unknown commits remain unknown. Hashes identify bytes but cannot recover absent source code.

| Copy / method | Reconstruction limit |
|---|---|
| ae_dpae | No verified upstream DPAE repository or commit. The retained license is a local record, not proof of official DPAE provenance. Controlled architecture assumptions do not reconstruct author code. |
| anogan | Unknown base revision; substantial adaptation described but no complete patch or licensed source snapshot bundled. |
| anomalyncd | Unknown revision; source-only record, no established local run. URL alone cannot recover the exact checkout. |
| dfr | Revision recorded, but source excluded and no complete local patch. Compatibility, scoring and ablation variants cannot be reconstructed from summaries alone. |
| draem | Recorded revision, seven-file verification record and local runner/metrics retained. Only seven upstream files have recorded verification; no full-environment reproduction claim. |
| ganomaly | Unknown revision and no complete upstream diff. Retained runners describe execution, but loader/model changes require the original backup. |
| padim | Unknown revision and no complete upstream diff. Controlled scripts survive; preprocessing and fitting changes cannot be recovered completely from descriptions. |
| patchcore | Unknown revision and no complete upstream diff. AUPRO/visualization helpers survive; complete historical configuration remains unverified. |
| patchsvdd | Unknown revision; source excluded and no complete patch. Exact-L2 bank, scoring and compatibility changes are described only. |
| rd4ad | Recorded revision and two historical patches retained, but not a verified complete diff for the archived working copy. Other changes may require the backup. |
| simplenet | Old revision unknown. Derived patch reconstructs five differing shared Python files from the newer archived snapshot, not a historical upstream commit. Local-only runner and monitor retained. |
| simplenet_20261020 | Recorded revision and local orchestration retained. Environment, data, cached weights and original paths still required; no relocated execution verified. |
| skip_ganomaly | Unknown revision; historical audit patch has an unpinned before-state and is not a complete fresh upstream diff. Exact recovery requires the original working copy. |
| supersimplenet | Recorded revision, training patch and local orchestration retained. Other Python files matched the local baseline in the prior review; patch application to a fresh remote checkout untested. |


## SimpleNet evaluation distinction

Both archived SimpleNet high-level train routines select by test image AUROC with test pixel AUROC as the tie-breaker. The older local CPU runner also uses best-test selection, so any such scores are test-dependent and may be optimistic. The 20 October local runner bypasses that routine: it trains the discriminator for 40 calls of four GAN epochs and evaluates the fixed final checkpoint. This does not remove test-selecting code from upstream SimpleNet. The routines, evidence and results have not been changed.

The derived SimpleNet patch runs from the newer archived source to the older archived source. It captures five shared Python-file differences: segmentation without dense features, CPU tensor cloning/mask storage, worker/CUDA loader compatibility, NumPy/pandas compatibility, and CPU device fallback. Eight other shared Python files are identical. The older local runner and monitor are retained separately. This is not proof of the old upstream revision or a complete patch for the old directory including documentation and new-only files.

## Patch and execution limits

The RD4AD and Skip-GANomaly patch headers describe original/patched or before/after files, not necessarily exact upstream baselines. Preserve those labels. SuperSimpleNet's patch removes intermediate test evaluation, checks finite loss and adds an epoch callback. Only the derived SimpleNet snapshot delta was reconstructed in memory against both archived sides during preparation; no patches were applied to research source files.

Local scripts retain original imports, workspace paths and dependency assumptions. They are records for understanding experiments, not standalone runnable programs. Environments, datasets, pretrained weights, checkpoints and full outputs are absent. No training, inference or claim of full reproducibility is made.
