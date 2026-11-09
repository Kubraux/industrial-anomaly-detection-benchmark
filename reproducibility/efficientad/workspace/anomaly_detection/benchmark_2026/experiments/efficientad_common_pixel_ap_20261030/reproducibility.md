# Reproducibility

Plan and config were written before any checkpoint load or numerical execution. Paper and supplementary inference/calibration details, reference README, authoritative protocol and historical audit were reread for this run.

Invocation from project root: EfficientAD_Anomalib_setup_20261023/venv/Scripts/python.exe -B -u anomaly_detection/benchmark_2026/experiments/efficientad_common_pixel_ap_20261030/regenerate_maps.py

The runner refuses to overwrite completed outputs. Any rerun requires a new prospective plan and run directory. Existing calibrated final checkpoints remain inputs only.

The unchanged common evaluator is imported without running its main/worker/calibrate functions. Only shared_gt and its average_precision_score are called; neither normal-data threshold fitting nor calibration occurs. Source hash is in environment.txt and source_manifest.json. sklearn version and all runtime versions are recorded. Numerical AP smoke tests cover ties, perfect ranking and manually verified pooled masks.

The exact preprocessing matches the retained historical evaluator. Model checkpoint config must equal original training config; strict model-state loading includes calibration. Calibration values must equal training.json. Thirty fit-image warmups do not update state; full state is hashed before/after all forward passes. Inference uses inference_mode and eval, no optimizer.

Every test image/mask is hashed against the historical test manifest. Fit/validation/test disjointness uses saved training hashes, while test hashes and the fit warmup file are freshly checked; this does not independently reconstruct historical training behavior. Original dataset archive version/hash remains unknown, with individual test identities retained.

All 3,450 common-grid maps are saved losslessly as float32 arrays with corresponding bool masks, names, scores and labels. No figure/quantized visualization substitutes for predictions. Original-resolution maps can be reconstructed by the original bilinear align_corners=False resize, but native Pixel AP is not part of this common-grid run.

The recorded process duration includes validation, inference, compression, AP and integrity checks. CUDA-event times measure the forward pass; synchronized model wall times are separate. Pipeline timing includes decode/resize/H2D, forward and map D2H, excluding audit hashes/mask reads/compression/AP. Warmup30, batch1, one pass/test image. GPU peak counters exclude warmup but include model residency; process RAM is a lifetime peak. Training/calibration time is null because neither was performed.

The dated RESULTS.md and metrics.json are measured results, not paper-reported values. Historical audit limits remain. See source_manifest.json for immutable source identities and verification.json in each category for alignment and state checks. failures.json retains an empty list for this successful run.
