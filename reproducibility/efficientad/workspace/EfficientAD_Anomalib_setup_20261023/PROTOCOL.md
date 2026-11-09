# EfficientAD S/M protocol frozen before training

Anomalib 2.2.0, commit 592aafc373091dda8206510a0853eb11c8e4099d. Source snapshot SHA256: 96a2de8a6335441ff2e6151c68ac650f8e1c43a4c006513c878fa80026f9b8e4.

Sources: https://github.com/open-edge-platform/anomalib/tree/v2.2.0/src/anomalib/models/image/efficient_ad ; https://arxiv.org/html/2303.14535v3 . Original model README and Python files preserved under source/. The README credits rximg and nelson1425; their configuration is not imported here. Some Anomalib docstrings incorrectly say EfficientNet: actual code is PDN, with 384 teacher / 768 student channels and the original autoencoder.

## Training and isolation

Seed 42 for initialization, augmentation and fit permutations. Independent auxiliary permutation seed 43 (derived from 42). Each category's sorted train/good paths are permuted with numpy.default_rng(42); floor(0.8*N) fit, remainder normal validation. Identical saved manifests for S and M, with decoded dimensions and SHA256. No anomalous MVTec image is admitted by the training reader. Teacher channel statistics use only fit (Anomalib's own helper). The test dataset is constructed only by the separate final evaluator. No threshold is fitted.

70,000 optimizer updates, batch 1, float32, input 256x256; Adam(student+AE), lr=0.0001, weight_decay=0.00001, betas=(0.9,0.999), eps=1e-8. StepLR gamma 0.1 is called after every optimizer update, step_size=66500: updates 1..66500 use 1e-4, subsequent updates use 1e-5. Anomalib's Lightning configure_optimizers omits scheduler interval='step'; this driver explicitly schedules by optimizer step. It uses the installed Anomalib EfficientAdModel with unchanged loss, augmentation, PDN and AE implementations, in a plain PyTorch loop rather than Lightning Engine. No automatic Anomalib datamodule validation, postprocessor, evaluator, early stopping or best checkpoint callbacks run.

Tensor conversion then antialiased bilinear resize. ImageNet channel normalization occurs inside the networks. Anomalib default padding=False, pad_maps=True is retained (56x56 features, pad maps 4 pixels, resize256). Paper architectural tables describe padded64x64; Appendix E explicitly says reported experiments/timing use padding disabled. This is not an unexplained architectural modification. Loss: hard squared teacher/student error at quantile .999 plus auxiliary student penalty, augmented teacher/AE MSE and AE/student MSE. Brightness/contrast/saturation coefficient uniform[.8,1.2], selected by Anomalib. Frozen teacher; no AMP, TF32 disabled. Deterministic cuDNN; exact cross-device bitwise reproduction is not guaranteed.

The final optimizer step is fixed before any test access. Recovery snapshots every5000 steps include optimizer, scheduler, RNGs and permutation queues. Final weights are calibrated exactly once using all held-out normal validation maps: separate .9/.995 quantiles, scale .1, then equal-weight map average. CPU float32 quantiles implement the same formula as Anomalib; all validation maps fit below its 2^24-element subsampling limit. No validation loss/metric selects epoch or checkpoint. Final checkpoint contains the calibrated model state. Smoke tests instead use four fit images for statistics, three optimizer steps, three validation images for calibration, and never create a test loader; their checkpoints are separate and discarded for full training.

## Teacher and auxiliary data provenance

Anomalib release bundle: https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip . SHA256 c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a. Files pretrained_teacher_small.pth and pretrained_teacher_medium.pth are loaded strictly into their corresponding architectures. Per-file hashes are recorded in each config. Teacher distillation is NOT rerun. The paper distills WideResNet-101 features using ImageNet (60,000 iterations, batch16); the downloaded release does not provide a full reproducible training manifest/RNG history for these teacher files, so their exact distillation history is not independently verified. Matching shapes and checksums verify artifact identity, not equivalence to the authors' teacher runs.

Imagenette2: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz ; SHA256 6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd. Use its 9469 training images only for the student penalty. Imagenette validation is not used. Auxiliary transform follows Anomalib: resize to512x512, RandomGrayscale(.3), center crop256, tensor. This is a 10-class ImageNet subset rather than the full ImageNet auxiliary population in the paper. The explicit train-only auxiliary path also avoids ImageFolder recursively mixing downloaded Imagenette train/val folders. MVTec validation is used only for map calibration.

## Evaluation

One post-training evaluation per final category model. Image AUROC uses the full256x256 anomaly-map maximum (Anomalib pred_score). Pixel AUROC uses all original-resolution pixels pooled across the category; maps resized bilinearly with align_corners=False; masks kept at native resolution. AUPRO uses8-connected defect regions, equal per-region weight, global background FPR including good test images, tied scores grouped, exact trapezoidal integral truncated/interpolated at FPR .30 and divided by .30. This matches DRAEM_setup/metrics.py's definition; the streaming implementation is checked against its exhaustive reference on synthetic maps, including ties/chunk boundaries. Previous resized-grid results differ in evaluation resolution.

30 warm-up passes using a fit image. Each test image is measured once, CUDA events and explicit synchronization. Separate disk decode/resize/H2D preprocessing, model GPU events, synchronized model wall time, native resize/D2H postprocessing. FPS=1000/mean_ms. PyTorch peak allocated/reserved memory is reported for training and inference. Paper timing uses float16, A6000,1000 warmups/1000 repeats, includes transfers, and batch16 for throughput; our float32 A4000 batch1 FPS is not its throughput measurement. Gallery uses the first lexicographically sorted anomalous image, not the best result; one equal-size four-panel PNG/category/model.

## Commands

Run in workspace root with the isolated environment:

```powershell
& EfficientAD_Anomalib_setup_20261023/venv/Scripts/python.exe EfficientAD_Anomalib_setup_20261023/experiment.py run
```

This audits train files, tests metrics, performs S and M Bottle smoke tests, trains S15 then evaluates S15, trains M15 then evaluates M15. The process stops and writes failure.json/evaluation_failure.json on error. No automatic test-dependent retraining. Full package versions: requirements.lock.txt. Individual worker: `experiment.py train s bottle`; evaluation: `evaluate.py s bottle`. Do not rerun a completed worker over its final checkpoint. Pending CSV entries are explicitly marked. Outputs are in new EfficientAD_[S/M]_Anomalib_training_20261023 and results folders; report in reports/efficientad_anomalib_20261023/COMPARISON.md.

## Environment reproduction and remaining limitations

Create a Python3.11 virtual environment; install torch==2.5.1 and torchvision==0.20.1 from https://download.pytorch.org/whl/cu121, then install requirements.lock.txt (same extra index for the +cu121 builds). Verify pip check. Download the two URLs listed above, verify SHA256, and extract under assets. Source files in the installed wheel were byte-for-byte checked against the pinned Git commit for torch_model.py and lightning_model.py. All9469 auxiliary training image headers verified with PIL; compressed archive checksum passed. Windows import required requests/ftfy/regex beyond Anomalib's core metadata; these additions are isolated and locked. The source package is unchanged.

Recovery files are saved for manual continuation; the current entrypoint intentionally stops if a final checkpoint already exists and does not automatically resume interrupted jobs. A crash is not silently converted into a fresh experiment. Training_seconds covers optimizer loop/data loading/recovery saves, excluding initial teacher statistics and final calibration. Local process must remain alive and machine must remain awake. No power settings were changed.
