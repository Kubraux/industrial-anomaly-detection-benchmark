# Industrial Visual Anomaly Detection

**Learning normal appearance. Localizing defects. Examining the reliability of anomaly decisions.**

An ongoing research project on detecting and localizing defects in industrial imagery. We study how models fitted to normal data behave on unseen defects, how anomaly scores become usable decisions, and which implementation and evaluation choices affect the evidence. MVTec AD is the current published benchmark; the broader research agenda includes additional industrial datasets, improved localization and future defect-type classification.

[Methods](methods/README.md) | [Results](results/README.md) | [Experiments](experiments/README.md) | [Visual gallery](assets/GALLERY.md) | [Source records](third_party/SOURCES.csv)

| Published method studies | Categories per study | Category-level result records |
|:---:|:---:|:---:|
| **11 completed** | **15 MVTec AD categories** | **165** |

Counts refer to the [published benchmark tables](results/summary.csv), not independent repetitions or every experiment in the research workspace. Evidence snapshot: **23 October 2026**; homepage reviewed **26 October 2026**.

![DRAEM: Grid input, ground truth and recorded anomaly response](assets/figures/draem_grid.png)

*Texture localization with the locally trained DRAEM final checkpoint: Grid/glue/007. The recorded preprocessing and GT conventions apply; the map does not imply a calibrated binary decision. [Method and limitations](methods/draem/README.md).*

## Methods and benchmarking

The published studies span complementary ways of representing normality and scoring deviations.

| Approach | Completed method guides |
|---|---|
| Normal-feature memory and distributions | [PatchCore](methods/patchcore/README.md), [PaDiM](methods/padim/README.md) |
| Feature reconstruction and distillation | [DFR](methods/dfr/README.md), [RD4AD](methods/rd4ad/README.md) |
| Self-supervised patch representations | [PatchSVDD](methods/patchsvdd/README.md) |
| Generative reconstruction and latent discrepancy | [AnoGAN adaptation](methods/anogan/README.md), [GANomaly](methods/ganomaly/README.md), [Skip-GANomaly](methods/skip_ganomaly/README.md) |
| Learning with synthetic anomalies | [SimpleNet](methods/simplenet/README.md), [SuperSimpleNet, JIMS extension](methods/supersimplenet/README.md), [DRAEM](methods/draem/README.md) |

Each guide distinguishes the original method from the actual local implementation, completed runs and historical diagnostics. [Full results](results/README.md) retain category-level measurements and additional metrics. **Different preprocessing, checkpoint selection, scoring and PRO/AUPRO implementations preclude a directly comparable leaderboard.** Completion establishes recorded local execution, not successful paper reproduction.

## Key research findings

Our contribution is the experimental investigation: calibration, controlled interventions, failure analysis and scrutiny of evaluation behavior.

- **Ranking does not determine thresholded segmentation quality.** In the common DFR/RD4AD evaluation, RD4AD has higher category-macro pixel AUROC (**97.78% vs 95.25%**) but lower pixel F1 (**15.04% vs 41.56%**) under the specified training-normal quantile rule. This is an observed threshold-transfer outcome, not a general ranking of the architectures. [Matched-evaluator evidence](results/common_evaluation/report.md).
- **Calibration exposes trade-offs.** In the PaDiM Bottle holdout study, changing q99.5 to q99.9 raises pooled Dice from **64.41% to 71.42%**, while recall falls from **97.99% to 92.43%**. The test set had previously been inspected; this is sensitivity analysis, not independently validated threshold selection. [Recorded calibration results](results/calibration/padim_bottle/metrics.csv).
- **False-positive investigations need carefully defined interventions.** DFR alignment and spatial-resolution ablations examine false-positive area alongside ranking metrics. Native128 changes both aggregation and PCA capacity, so its effects cannot be attributed solely to output resolution. [Ablations and failure analyses](results/ADDITIONAL_STUDIES.md).
- **Evaluation behavior can change the interpretation of a result.** GANomaly's best-test selection and persistent test BatchNorm updates affect its reported scores; later scoring variants and pretrained-weight evaluations are recorded separately. [Protocol distinctions](results/PROTOCOLS.md).

![DFR and RD4AD: independently scaled maps and recorded thresholded masks](assets/figures/common_leather.png)

*Same Leather sample, different score scales and recorded normal-training thresholds. White denotes predicted anomaly in the binary panels; cyan traces GT. This selected example illustrates threshold transfer, not representative dataset-wide behavior.*

![PatchSVDD: recorded normal and defective Screw failure examples](assets/figures/patchsvdd_screw.png)

*Failure evidence is retained alongside successful localization: these Screw cases were selected by score, with product maps sharing their original display range. No score inversion or new inference was performed. [PatchSVDD analysis](methods/patchsvdd/README.md).*

[Explore all methods in the visual gallery](assets/GALLERY.md). Figure sources, hashes, selection rules and rendering transformations are recorded in [figure provenance](assets/figures/SOURCES.csv).

## Repository guide

| Resource | Contents |
|---|---|
| [Methods](methods/README.md) | Architectures, actual implementations, outcomes and limitations |
| [Experiments](experiments/README.md) | Compact registry, retained settings and evaluation protocols |
| [Results](results/README.md) | Benchmark tables, calibration, ablations and negative results |
| [Visual gallery](assets/GALLERY.md) | Method coverage, illustrative examples and quantitative-only evidence |
| [Third-party sources](third_party/README.md) | Repository references, recorded revisions, local changes and licenses |
| [Local code](code/README.md) | Selected orchestration scripts for inspection |

## Reproducibility and limitations

The published summaries describe single-seed studies. Equal-category macro averages are not uncertainty estimates. Some studies include test-dependent checkpoint selection, test-updated state or prior test inspection; historical SimpleNet CPU selection differs from the newer fixed-final run. These issues remain explicit in the [evaluation protocols](experiments/PROTOCOLS.md).

Selected runners, patches and source records are available, but some historical revisions, complete modifications and configuration details are missing. Raw datasets, large checkpoints and full research outputs are not bundled. Some provenance references require the original workspace; training and inference after relocation have not been verified. See [reconstruction limits](third_party/RECONSTRUCTION_LIMITS.md). Failed or partial diagnostics are retained separately from completed benchmarks.

## Future directions

Planned research includes cross-dataset evaluation on additional industrial data, further anomaly detection methods, sharper localization with fewer false positives, and efficiency studies under consistent hardware and measurement scopes. Defect-type classification is a future extension, not a completed capability of this benchmark. Broader seed coverage and more rigorous calibration and protocol controls remain priorities.

## References and attribution

Original papers and implementation sources are linked from the [method guides](methods/README.md), [reference list](literature/REFERENCES.md) and [source catalog](third_party/SOURCES.csv). Upstream algorithms and code remain credited to their authors; local adaptations are identified separately.

**No blanket project license has been selected.** Consult the [licensing scope](LICENSE.md) and [retained third-party licenses](third_party/licenses/README.md). This repository does not grant new rights to upstream code, datasets or figures.
