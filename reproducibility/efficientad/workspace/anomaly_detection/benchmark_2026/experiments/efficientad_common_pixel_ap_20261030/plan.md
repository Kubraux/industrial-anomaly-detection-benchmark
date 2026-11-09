# Prospective plan: efficientad_common_pixel_ap_20261030
Status: planned. Created 2026-10-30 (Europe/Istanbul), before checkpoint loading or numerical evaluation.
User explicitly authorizes inference from saved final checkpoints for both variants/all15, common-evaluator Pixel AP, no training/recalibration or historical overwrite.
Protocol v1.4 read in full this turn; existing EfficientAD audit Section 5, legacy setup protocol, original evaluator/runner, pinned model source/README and original paper/supplement reviewed.

## Section 9 / fixed specification
- RQ1 localization metric completion; historical T1 single-class normal-only, implementation-specific common256 evaluation extension.
- Methods: EfficientAD-S and EfficientAD-M, family B, Anomalib 2.2.0 commit 592aafc373091dda8206510a0853eb11c8e4099d (not author-official implementation).
- Paper: https://arxiv.org/html/2303.14535v3 (method normalization, Appendix A inference and Appendix E padding/timing); reference https://github.com/open-edge-platform/anomalib/tree/v2.2.0/src/anomalib/models/image/efficient_ad .
- Dataset: original archive/ MVTec AD 15 categories; 1,725 official test images, 3,629 normal train images; archive release checksum unknown. archive/readme.txt states CC BY-NC-SA4.0. No redistribution.
- Categories: bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper.
- Historical seed42, steps70000 fixed final, 2,900 fit /729 held-out normal calibration images; no model/epoch/variant selection. Preserve normal80/20 split manifests and verify recorded fit/validation/test hash disjointness.
- Checkpoints: EfficientAD_[S|M]_Anomalib_training_20261023/<category>/final.pt. Strict load complete saved state; require step70000 and matching configuration. Never load recovery or smoke files.
- Pretrained teacher: small/medium release identity in original configuration; teacher, student, AE, mean/std and quantile tensors loaded from final checkpoint. Auxiliary Imagenette9469 is historical disclosure, no auxiliary images used now. Teacher's exact distillation lineage remains unverified.
- Input unchanged: PIL RGB; torchvision v2 ToTensor then Resize256x256 antialias=True, bilinear default; ImageNet normalization internal; float32, batch1, padding=False,pad_maps=True, deterministic cuDNN, no TF32/AMP.
- Calibration unchanged: use checkpoint channel mean/std and held-out quantiles (.9/.995), no updates. Equal-weight calibrated local/global maps from original model forward; image score is model map maximum.
- Both continuous image scores and full256 maps obtained and retained as predictions.npz, with paired common masks, labels and ordered names. No visualization substituted for numeric outputs.
- Common evaluator: unchanged compare_dfr_rd4ad_common.py shared_gt plus its imported sklearn average_precision_score. DO NOT call worker, calibrate, threshold, test_evaluator, or any original run/report entrypoint.
- Common grid256: original masks converted L, PIL nearest resized256 then >0; no fill, crop or per-image normalization. Include normal test images as zero masks. Model's 256 output already matches grid; do not downsample previously upsampled maps.
- Pixel AP: non-interpolated sklearn AP positive1, all pixels pooled per category, no weights; equal category macro over15 separately for S and M. No seed uncertainty invented.
- Historical pixel AUROC/PRO used native-resolution masks; common256 AP is a separately labeled evaluation variant, not a replacement or direct resolution-equivalent result. User request for common evaluator authorizes this separate evaluation; changelog records it.
- No new image performance claim: regenerated image scores are checked against retained historical scores with preregistered atol1e-6/rtol1e-5, all samples; stop/fail category on mismatch. This is fidelity validation, not test-dependent tuning.
- Validate original image/mask dimensions, exact file identity/SHA versus historical manifests, unique ordered test population, binary mask polarity, nonempty anomaly masks; zero normal masks. Require common mask byte SHA equals existing RD4AD common-evaluator mask SHA for same order.
- Verify model-state hash before/after including all buffers; validate calibration tensor values against training.json. Checkpoint/config/manifest/source/result hashes before/after. No training, validation calibration, threshold fit, optimizer or gradients.
- Smoke: hand AP tie/perfect/pooled-mask cases plus deterministic first-fit-image30 warmups/category; no state update. Failure logs retained, no silent parameter modifications/retries.
- Region metrics and operating thresholds not_run in this AP-specific task; image AP already available in saved_prediction_ap_20261030.

## Resource budget and execution
NVIDIA RTX A4000 16,376MiB, ~15,235MiB free; disk ~63GiB free at planning.
Sequential30 categories, expected <=30min, GPU <=4GiB, RAM <=8GiB, disk <2GiB for maps/masks compressed (uncompressed ~1.13GB for both variants). Abort on resource constraints without numerical approximations.
Record environment/source hashes, checkpoint hashes, per-image synchronized CUDA-event model time and pipeline wall time, GPU allocated/reserved peaks and process RAM; timing excludes audit/hash/compression/AP except total evaluation time. 30 fit-image warmups, one pass/test image; diagnostic timings, not paper throughput.
Use isolated existing Python3.11 environment with -B; local cache destinations under new run. No dependency installation/download expected.
New writes ONLY under anomaly_detection/benchmark_2026, except authorized changelog entry. Historical inputs remain read-only.
Next step: frozen script/config, then sequential inference + AP, starting S/bottle. Persist per-category output immediately, retain failures; full macro only on15 successful categories.

## Deliverables
plan.md, config.yaml, environment.txt, run.log, metrics.json, metrics.csv, summary.json, RESULTS.md, resource_usage.json, reproducibility.md, source_manifest.json, failures.json, assets/; per-variant/category predictions.npz, test_manifest.json, verification.json, timings.json, metrics.json, config.json, checkpoint provenance.

## Post-run status
Completed 30/30 category evaluations, no failures. All 3,450 image scores reproduced exactly; common-mask and unchanged-state checks passed. See RESULTS.md. Prospective settings were not changed during execution.
