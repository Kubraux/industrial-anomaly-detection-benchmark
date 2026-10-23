# Repository convolutional AE baseline

[Project overview](../../README.md) | [Method index](../README.md)

**Status: historical Bottle experiment outputs documented; original run and evaluation files were deleted before this review.** This entry preserves a distinct experimental identity with indirect execution evidence. Completion, exact training settings and numerical results cannot now be verified.

## Principle and architecture

The retained local source defines a convolutional autoencoder with four stride-2 encoder blocks, a convolutional bottleneck and four transposed-convolution decoder blocks ending in sigmoid. The default base width is 32. AEModel combines 0.5 times (1 - SSIM) with 0.5 times MSE. Its actual predict implementation uses channel-mean absolute reconstruction error as the pixel map and the maximum pixel error as the image score.

These facts describe the retained source, not a recovered configuration for the deleted run. This baseline is distinct from the [controlled AE-L2 and DPAE](../ae_dpae/README.md); it is not established as the paper's Bergmann AE baseline.

## Recorded setup

The cleanup inventory identifies old MVTec Bottle outputs under defect-detection-main/runs/ae/bottle, including dated run folders, checkpoint files, run.json, evaluation metrics.txt and ROC/PR plots. The cleanup result confirms deletion of that directory. The surviving experiment report describes the old baseline as SSIM+MSE with an Adam training recipe.

The actual epoch count, image size, seed, split, batch size, learning rate, hardware, checkpoint selection and code revision of those runs are unrecoverable from the inspected records. Current Magnetic Tile controlled-model settings must not be assigned to this older Bottle experiment. Normal-image reconstruction is the method's intended training approach; its original split cannot be certified.

## Outcomes and limitations

No numerical AUROC, AUPRC, localization or threshold result is reported here: the contents of the old metrics.txt and run.json files are unavailable. A cleanup inventory of outputs is historical provenance, not a substitute for original metric contents, a successful-training log or a reproduction claim.

The historical baseline is included so that an experiment documented by cleanup records is not silently omitted. It is listed separately from methods with surviving direct execution records and from source-only methods.

## Sources and local work

Inspected workspace-relative sources (outside github_release):

- defect-detection-main/reports/cleanup_inventory.csv: paths and sizes of checkpoint, run and evaluation files.
- defect-detection-main/reports/cleanup_result.json: recorded deletion.
- defect-detection-main/reports/experiment_report.md: distinction between repo_AE_baseline, AE_L2_controlled and DPAE_controlled.
- defect-detection-main/src/models/ae.py: retained architecture, loss and scoring implementation.
- defect-detection-main/LICENSE: local license record.

See the [reconciliation evidence inventory](../RECONCILIATION_EVIDENCE.csv) for hashes. No verified original paper, official repository or commit is established for this particular implementation. The third-party AE-DPAE catalog record does not establish an official upstream for this distinct baseline. No missing source identity, historical modification or license attribution is invented.

No new experiment was run for this documentation. The removed results have not been reconstructed.
