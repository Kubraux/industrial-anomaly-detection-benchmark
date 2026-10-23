# Model selection, scoring and evaluation limits

Evidence snapshot: 23 October 2026. Configuration values are recorded facts; absent values are unknown. A model's configured epoch budget does not prove that it completed that budget.

| Run family | Selection / threshold rule | Limitation |
|---|---|---|
| PatchCore | Normal feature-bank fitting; nearest-neighbor scoring | Coreset fraction and full historical command unknown; CPU evaluation does not prove CPU fitting |
| DFR main / ablations | Final150; normal-training pixel quantile for threshold diagnostics | Training-normal calibration is not held-out calibration; variants remain separate |
| RD4AD main | Final200 | Bottle periodic test access is recorded; final selection alone does not establish an untouched test set |
| RD4AD holdout | Final200; normal holdout threshold; A old100/train, B new80/train, C new80/holdout | B-C isolates threshold; A-C changes fitting data too; prior test inspection limits confirmation |
| PaDiM holdouts | Fixed q99.5 pixel primary, q99.9 sensitivity; imageq95; higher quantile,strict > | Normal calibration only, but prior test inspection is explicit; few calibration images limit tail resolution |
| AnoGAN | Normal-holdout area threshold; later max(rawL2map) image AUROC | Later scoring rule is a separate evaluation, not a reproduction of original thresholded accuracy |
| GANomaly official-flow | Best test AUROC checkpoint; train-mode tests update BN | Test-set model selection and persistent BN contamination; oracle test F1 threshold leaks test labels |
| GANomaly archived variants | Fixed final epoch; train-normalq99 and oracle F1 reported separately | Same previously inspected test data; exploration is not independent tuning validation |
| Skip-GANomaly | Main fixed final20 and one test | Earlier audit identified test-epoch selection flaw; corrected final protocol must not be retroactively assigned to old runs |
| PatchSVDD | Main final299; lambda fixed0.001 | Later Carpet lambda, random/pretrained and numerical tests are post-hoc; do not substitute best variants into main table |
| SimpleNet old CPU | Best test image AUROC, pixel AUROC tie-break | Explicit test-based checkpoint selection; interrupted before any completed category |
| SimpleNet new | Forty calls to original discriminator routine, four GAN epochs each; final checkpoint | Upstream high-level train() remains test-selecting; local runner bypasses it |
| SuperSimpleNet | Fixed final300; intermediate tests removed | Single seed, JIMS rather than ICPR model; no threshold fitting; suspect VRAM counters are not physical memory requirements |
| DRAEM | Fixed final700; no test model/threshold selection | Code schedule/augmentation differ from paper; BGR and linear-resize/uint8 GT retained; pretrained-weight evaluation separate |
| Controlled AE/DPAE | Failed smoke and normalized short probes | Local architecture/prototype assumptions; no final evaluation or successful main comparison |
| Historical AE | Unknown | Deleted run/configuration/metric contents cannot be recovered from cleanup paths |

## Metric definitions

Image AUROC, pixel AUROC and operating-threshold accuracy are different outcomes. Pixel maps are not automatically validated binary masks. Category macro averages are not repeated-seed uncertainty.

- DFR native PRO and RD4AD native per-image PRO are not interchangeable with exact global-FPR AUPRO.
- Common DFR/RD4AD evaluation uses nearest256 GT, equal-weight 8-connected regions, global background FPR, exact tied-score sweep, interpolation at0.30; normal-training q99.5 higher threshold with strict >.
- PaDiM, PatchSVDD and newer SimpleNet/SuperSimpleNet use their recorded local exact AUPRO implementations; differing crop/mask grids still prevent matched-protocol claims.
- DRAEM local AUPRO@0.30 is not pixel AP. Its ground-truth mask interpolation/casting is preserved, not silently corrected.
- GANomaly oracle test-F1-max values are descriptive upper-bound selections and must not be labeled deployment thresholds. Neither final nor best-test official-flow scores are independent of train-mode test BN behavior.
- A fixed quantile learned on training/held-out normals is not inherently test-label threshold leakage. Prior inspection and exploratory protocol choices must nevertheless be disclosed.

No benchmark is declared leakage-free solely because a final checkpoint was used.
