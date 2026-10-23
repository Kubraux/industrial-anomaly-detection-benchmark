# DFR bottle experiment: changes

Original source files are saved in source_backup before modification.

- Windows compatibility: MVTec.py returns sorted image paths with forward slashes. Existing good-class checks and mask path derivation remain unchanged; same image order and pixel transformations.
- modern PyTorch compatibility (NumPy dependency): anoseg_dfr.py replaces np.bool with np.bool_. Boolean metric semantics unchanged.
- dataset path adaptation: separate run_dfr_experiment.py calls original classes with archive/<category>/train/good and test. Original main.py is retained as evidence; its hardcoded Linux paths and 15-category loop are bypassed.
- metric/evaluation logging: runner records PCA input paths/shapes, parameter update assertions, finite loss, synchronized timing, memory, raw maps, metadata and checkpoints. Calls original training/optimization and official metrics_evaluation.
- metric/evaluation logging: training-time validation is disabled to keep test images/masks out of training execution. Fixed final epoch; no test-based selection.
- metric/evaluation logging: deployment/visualization threshold uses original estimate_thred_with_fpr(0.005) on training data only. Official evaluation sweep is a metric integral, not threshold tuning; its oracle best-IoU is not used as a deployment result.
- metric/evaluation logging: deterministic seed 0, cuDNN benchmark off, TF32 off. Original repo does not set a seed. Full bitwise cross-platform reproducibility is not claimed.
- algorithmic change = none relative to active official repo: preserve frozen reflection-padded VGG19, 12 layers, bilinear alignment, 4x4 aggregation/stride4, PCA90 on first 10 sorted training images, 1x1 FeatCAE with BN, MSE, Adam lr1e-4, batch4, 150 epochs.

Paper comparison source: https://arxiv.org/html/2012.07122 (v1, section IV-A, Appendix B, Tables II/III).
Official main: https://github.com/YoungGod/DFR/blob/master/DFR-source/main.py
Paper: 700 epochs, nearest-neighbor feature alignment, CAE appendix lists conv/ReLU without BN, L2 formulas; repo: 150 epochs, bilinear, BN on, mean squared error. Paper describes randomly sampled features for PCA; repo uses first eval loader batch (10 sorted training images).
Paper bottle 12-scale pixel ROC-AUC 0.95, PRO-AUC 0.92; 16-scale 0.97/0.93. These are not exact matched settings for this repo-default experiment.
README does not specify epochs; main.py line 46 sets 150 and comment says 'default 700, for wine 150'.
`nAdditional paper difference: Appendix B uses intermediate width (C+d)//2; active repo uses (C+2d)//2. Repo architecture is retained.
