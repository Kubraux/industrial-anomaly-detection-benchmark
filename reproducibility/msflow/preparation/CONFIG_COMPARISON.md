# MSFlow paper/reference versus prepared run

Paper read in full: archived arXiv2308.15300v1,14pages, linked by official README; journal-version exact equivalence not claimed. No separate supplementary file is linked by that README; ablations are in the paper. Primary source commit is pinned.

| Item | Paper / release | Prepared setting |
|---|---|---|
| Architecture | WRN50 first3 stages;2/5/8 conditional flows plus fusion;64-position encoding | Exact official model files; custom reflect padding preserved |
| Batch | Paper16; default.py16; main.py overrides to8 | 8, explicit release-default run, not exact paper batch reproduction |
| Resolution | Paper512 except transistor256; CLI mistakenly resolves before class loop |512 except transistor256, as paper and saved upstream transistor log |
| Schedule |100 epochs; code25x4; Adam1e-4; decay70/90 gamma0.33 |Preserved;12subepoch warmup from release; smoke interrupts after4 updates |
| Extractor |Paper frozen; release eval/outside optimizer but autograd enabled | Preserve release autograd initially; hash verifies weights unchanged |
| Test selection |Release tests every meta epoch; separate best_det/best_loc/pro | No test loader, no periodic test; fixed-final100 only |
| Seeds |9826 once before all categories |9826 per category for independent resumability, explicit deviation; smoke first bottle |
| Runtime |Python3.9,torch>=1.10,CUDA11.3; FrEIA unpinned |Python3.11,torch2.5.1+cu121; FrEIA last pre-Jan2024 revision pinned; no bitwise original-runtime claim |
| Data workers |4 |0 Windows; unchanged batches/shuffle semantics |
| Scores |top3% multiplicative image; additive pixel AUROC; multiplicative PRO |Both maps preserved; eventual common-evaluator mapping requires explicit report |
| PRO |800 thresholds, legacy np.bool and FPR rescaling |Reference evaluation not invoked; exact common AU-PRO adapter remains pending |
| License |CC BY-NC-ND4.0 in source |Local research source/adaptations only; no code redistribution |

Wandb/test/evaluation imports are not required by the training-only adapter. It executes only the source-verified model_forward and train_meta_epoch functions from the upstream AST, retaining loss/backprop/gradient clip/scheduler steps verbatim. Full original train.py remains immutable reference.
Only source-normal training examples and synthetic scores/masks may be used for preparation. No test data used for model/hyperparameter/threshold selection.

## Pre-smoke attempt002 dependency correction
Protocol freshly reread. Attempt001 stopped at0updates: FrEIA Dec2023 strict spatial check rejects reference SequenceINN(C,1,1) runtime(C,H,W). Install original published PyPI FrEIA0.2 sdist, hash e9f1732b4a238b85b0ad3bc14b279a653a6063d14152028e164b989582026bdc, release date 2022-07-04T08:51:18. Original MSFlow did not pin FrEIA, so exact author runtime remains unverified. No MSFlow model/loss source edits. Only the isolated MSFlow dependency changes; failure and attempt001 config/environment preserved. Batch8/512/FP32/seed9826/4updates unchanged.
