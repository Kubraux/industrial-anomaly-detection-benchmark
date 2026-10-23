# Controlled AE–DPAE

[Project overview](../../README.md) · [Method index](../README.md)

**Status: diagnostic-only; main comparison incomplete.**

## Principle and architecture

The controlled AE is the L2 ablation of the DPAE encoder/decoder. DPAE uses an additional encoder and prototype-related losses; latent disagreement supplies its score. This local AE is not the paper’s Bergmann AE baseline.

## Recorded setup

Magnetic Tile; 761 normal fitting images; 191 normal and 392 defective test images; 224 px; latent 32; seed 42; SGD. Prototype and some layer details are explicit local assumptions.

[Configuration/evidence](../../experiments/ae_dpae/configs).

## Outcomes and limitations

Literal-sum smoke tests failed at epoch1, batch 5 with nonfinite BN state. Normalized-loss10 batch diagnostics completed for both models; AE has one further epoch history but stale status metadata. No completed main comparison, final ROC result or validated four-panel segmentation figure exists.

[Recorded status/results](../../results/incomplete/ae_dpae/reports/experiment_report.md).

## Sources and local work

No verified upstream DPAE implementation was established; see the recorded [local assumptions](../../results/incomplete/ae_dpae/reports/pretraining_verification.md).

No new experiment was run for this documentation. Relocated execution is unverified.
