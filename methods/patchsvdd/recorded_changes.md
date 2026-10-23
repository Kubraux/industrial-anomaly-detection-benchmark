# Changes and retained behavior
- Compatibility: standalone Windows runner, num_workers=0, explicit RGB BICUBIC resize, deterministic seed1024, atomic status and per-epoch recovery checkpoint. Original network/dataset implementations untouched and snapshotted.
- Evaluation: only final299th training pass evaluated. Original300 loop included an untrained iteration0 and299 train passes. Epoch tests removed; no early stopping or best-test selection.
- Full exact float32 L2, chunks512x8192, no coreset/subsampling. Replaces KDTree/approximate NGT backend; TF32 disabled, selected distance recalculated by direct subtraction. Full normal banks saved.
- Masks: explicit nearest-neighbor256 and >128 binary; original repo implicit Pillow interpolation could interpolate masks. Report as evaluation protocol difference.
- AUPRO@0.30 added and verified on synthetic arrays against independent brute threshold implementation.
- Visualization: fixedseed1024 2normal+1perdefect single input/GT/heatmap panel; no test maps persisted.
- Preserved NormalizedLinear: fc3.weight of each position classifier has no gradient due to repo no_grad normalization; other encoder/classifier parameters must update. No silent algorithmic fix.
- Hyperparameters not explicit in paper are sourced from README/training code or marked local. Published Bottle AUROCs98.6/98.1 are reference, not selection targets.
