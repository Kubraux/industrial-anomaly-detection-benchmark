# MSFlow readiness — 2026-11-09

**Preparation and GPU smoke passed. Full15 training has NOT started. Final benchmark evaluation/export adapter and persistent sequential launch queue remain pending.**

## Frozen configuration
Official MSFlow commit e23975598bc7eb6f74604739d8f04d1519d704da; all19 downloaded blobs match pinned Git hashes. Original model/loss source unchanged. Custom reflect-padding WideResNet50_2 (ImageNet V1), first3 stages, average pooling, parallel flows2/5/8 plus fusion; positional channels64,clamp1.9.

Batch8 is the effective official main.py default. Paper/default.py use16: this is explicitly a release-default configuration, not an exact paper reproduction. Input512 except transistor256 per paper and saved reference log, correcting the CLI assignment before category loop. Seed9826 per category for independent resumption differs from original once-before-loop. Windows workers0,FP32. Adam1e-4,100epochs (25x4),12-subepoch warmup,decay70/90 gamma0.33,clip2. Teacher eval/outside optimizer with reference autograd retained. No test evaluation or best-test checkpoints; fixed-final only for future full runs.

## Measured validation
- Attempt001: failed before any optimizer update because a later FrEIA shape assertion rejects MSFlow dynamic dimensions. Failure/config/environment preserved.
- Attempt002: bottle train/good,4 optimizer updates,finite losses/gradients. All flow modules changed; teacher unchanged; zero test images read.
- Peak allocated **8.014GiB**, reserved **8.676GiB**, RTX A4000 16GB. Short-smoke measurements, not full-training maxima.
- Last3 update times: [0.6588259004056454, 0.6264059999957681, 0.6190810995176435]. Includes data/preprocessing/H2D/update and finite-gradient instrumentation; not inference latency.
- Normal-only additive/multiplicative512 maps and top3% image score validated. Independent-process strict checkpoint reload reproduced all outputs exactly (maximum difference0).
- Published FrEIA0.2 sdist SHA256 e9f1732b4a238b85b0ad3bc14b279a653a6063d14152028e164b989582026bdc; Python3.11.9,torch2.5.1+cu121,torchvision0.20.1+cu121. pip check passed. Original author's exact FrEIA version is not recorded; no bitwise historical-runtime claim.

## Before full15 launch
Training kernel and save/reload are ready for the documented release-default setup. Prepare persistent sequential coordinator, completed-category skip, immutable per-category configs/source pins and time/disk guards. Full runner mode has an explicit allow flag and was not executed. Fresh training must start from original pretrained teacher/random flows, not smoke weights.

There are 45800 expected updates at100epochs/batch8 across3629 training images. Rough forecast **8.1hours** from only3 post-first bottle updates; excludes initialization/checkpoint overhead and is highly provisional. Suggested safety budget24hours, not yet launched. Fifteen final flow packages require **6.66GiB**; free disk currently **15.82GiB**. Reserve additional prediction/export space before evaluation.

Final evaluator is pending: native additive Pixel AUROC/AP and multiplicative reference PRO must be separately labeled. A common single-map suite must be specified prospectively (suggest additive primary, multiplicative supplementary) without selecting per category using test outcomes. Original512/256 grids and any common256 extension must stay separate. Legacy np.bool/800-threshold PRO is reference-only, not validated. No final benchmark metrics were measured.

## Records
[Plan](plan.md), [comparison](CONFIG_COMPARISON.md), [configuration](config.json), [validation](validation.json), [resources](resource_usage.json), [environment](environment.txt), [failure](smoke_attempt_001/failure.json).

Original protocol v1.5 and historical artifacts are unchanged. Source license CC BY-NC-ND4.0; code/adaptations stay local. Nothing published to GitHub.
