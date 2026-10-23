# Figure gallery

Evidence snapshot: 23 October 2026. Every completed method has visual coverage below, with evidence type stated explicitly. Examples are chosen for scientific interpretation and category diversity, not unbiased performance estimation. Different methods' map colors and metrics do not define a matched-protocol comparison.

| Method | Figure | Category / sample | Evidence and limits |
|---|---|---|---|
| [PatchCore](../methods/patchcore/README.md) | [View](figures/patchcore_hazelnut.png) | Hazelnut crack/000 | Input / GT / map; class-shared PowerNorm display. |
| [DFR](../methods/dfr/README.md) | [View](figures/dfr_native64_native128.png) | Leather color/000 | Resolution ablation, with paired scale and binary masks. |
| [RD4AD](../methods/rd4ad/README.md) | [View](figures/rd4ad_grid.png) | Grid bent/011 | Input / GT / map; explicitly selected high-FP case. |
| [PaDiM](../methods/padim/README.md) | [View](figures/padim_calibration.png) | Bottle good/010; broken_large/005 | GT and two holdout-threshold masks; inspected test sensitivity study. |
| [AnoGAN adaptation](../methods/anogan/README.md) | [View](figures/anogan_grid.png) | Grid bent and good | Existing pseudocolored reconstruction/residual diagnostic; no GT/mask supplied. |
| [GANomaly](../methods/ganomaly/README.md) | [View](figures/ganomaly_carpet_diagnostic.png) | Carpet color and good | Archived reconstruction failure; final official-flow qualitative coverage remains unverified. |
| [Skip-GANomaly](../methods/skip_ganomaly/README.md) | [View](figures/skip_ganomaly_categories.png) | All15 | Quantitative category chart only; no saved final-run qualitative images found. |
| [PatchSVDD](../methods/patchsvdd/README.md) | [View](figures/patchsvdd_screw.png) | Screw good/005; thread_top/003 | Score-selected failure; product-map display shared across rows. |
| [SimpleNet](../methods/simplenet/README.md) | [View](figures/simplenet_cable.png) | Cable | Input / GT / map from completed fixed-final CUDA run. |
| [SuperSimpleNet](../methods/supersimplenet/README.md) | [View](figures/supersimplenet_transistor.png) | Transistor | Final300 JIMS-extension example; separate image head. |
| [DRAEM](../methods/draem/README.md) | [View](figures/draem_grid.png) | Grid glue/007 | Locally trained final700; original GT/preprocessing conventions retained. |

[Shared DFR/RD4AD comparison](figures/common_leather.png) additionally shows independently scaled maps and normal-training-threshold masks on the same Leather sample. The main [results page](../results/README.md) embeds four key examples; other figures are linked here and in method guides.

All figures are 1200x675. [Rendering conventions](README.md) and [source hashes/crop rectangles](figures/SOURCES.csv) apply. Original source plots and replaced presentation files remain in the research archive. Representative success, background response and failure cases coexist; selection by diagnostic extremes is explicitly identified.

Missing evidence: no final-run qualitative image artifact was found for Skip-GANomaly. GANomaly's included panel is an earlier archived diagnostic, not qualitative verification of the official-flow aggregate. AnoGAN's reconstruction panel has no GT/binary-mask column and its residual display limits are unverified. These gaps are not filled with synthesized maps.
