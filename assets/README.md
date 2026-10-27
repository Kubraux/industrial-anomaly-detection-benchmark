# Presentation figures

[Gallery and method coverage](GALLERY.md) | [Results](../results/README.md) | [Figure provenance](figures/SOURCES.csv)

Individual presentation figures are 1200x675 with Arial typography, fixed title/footer margins and a compact layout. Single-row three-panel figures use 360px square slots; two-row examples use up to244px slots. Larger column counts reduce slot size without stretching image contents. Source-image rectangles retain their aspect ratios and full displayed field of view.

GT and binary masks use nearest-neighbor resizing; other panels use Lanczos. These are reformatted existing plots, not newly inferred model outputs. No missing mask is synthesized. Original color normalization is retained; independent/shared/unknown scales are stated on each figure. Removed source colorbars mean these panels are qualitative illustrations, not numerical score readouts. AnoGAN's original pseudocolored diagnostic inputs remain pseudocolored.

Skip-GANomaly is a standard category-level plot of unchanged AUROC fields, not a localization example. Components are diagnostic, while the recorded combined score remains primary.

SOURCES.csv records source/output SHA-256, original crop rectangles, output panel positions, method/category, selection and scaling notes. Archive-source paths are workspace-relative provenance identifiers, not bundled GitHub links. Original experiments remain unchanged. Prior presentation files and their provenance are retained outside GitHub under publication_archive/visual_refresh_20261023; earlier full outputs remain under publication_archive/results_assets_20261023.

EfficientAD-S through Anomalib was added 26 October 2026 from its recorded Tile/crack/000 output; the main homepage keeps its two existing figures. Source metadata additionally includes the all-category overview below; the existing 12-method gallery is retained. This does not change previous figure selections or evaluation grids.

## All-category SuperSimpleNet overview

[Full-resolution overview](figures/supersimplenet_all15.png) · [Per-category source records](figures/supersimplenet_all15_sources.csv) · [Method](../methods/supersimplenet/README.md)

Evidence snapshot: **27 October 2026**. The overview uses genuine saved comparison.png panels from the completed SuperSimpleNet final300 run dated 20 October 2026, covering all 15 categories. Each visualization.json records the lexicographically first anomalous test image, fixed independently of scores. All categories are included, including broad background responses and imperfect localization. This deterministic coverage is neither random sampling nor evidence of average performance.

The 3000×2060 canvas is an intentional high-resolution exception for 45 readable panels. Rows follow alphabetical category order, three category triplets per row. Each original 1820×520 comparison plot contributes the complete square input, GT and anomaly-map panels at rectangles (20,66,451,497), (470,66,901,497), and (920,66,1351,497). Original plot titles and overlay panels are omitted. Input and map tiles use Lanczos resizing; GT tiles use nearest-neighbor resizing. The original preprocessed input and GT (>0.5 in the saved evaluation pipeline) are retained, rather than substituting native images or recomputing masks.

The source evaluator rendered jet anomaly maps with **per-image min/max**. This composition neither recolors nor numerically renormalizes them. Identical colors across samples do not denote identical anomaly scores; no calibrated segmentation threshold is shown. Per-category source plot, selection metadata, completion status and metric hashes, crop rectangles and output positions are recorded in the source CSV. Workspace-relative archive identifiers are provenance references, not bundled source files.

### MVTec AD attribution and figure license

Input imagery and ground truth originate from **MVTec AD**, by Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger (MVTec Software GmbH), *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. See the [official dataset and terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and [original paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/datasets/mvtec_ad.pdf).

The dataset is licensed under [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/). **supersimplenet_all15.png, simplenet_all15.png, efficientad_s_all15.png and draem_all15.png are shared as dataset-derived adaptations under that same license.** Changes comprise the experiment's recorded preprocessing and model heatmaps, followed here by cropping existing panels, resizing, rearranging and relabeling. Attribution, the license link, noncommercial use and ShareAlike conditions must accompany reuse. This figure-specific notice does not assign a blanket project or upstream-code license.

## EfficientAD-S and DRAEM all-category overviews

Added **27 October 2026**. [EfficientAD-S overview](figures/efficientad_s_all15.png) and [DRAEM overview](figures/draem_all15.png) follow the same 3000×2060 layout, typography, alphabetical category ordering and 280px panel slots as the SuperSimpleNet overview. All 15 categories are included, including imperfect localization. The original comparison plots remain unchanged in the research workspace; no inference, training or metric reevaluation was performed.

**EfficientAD-S through Anomalib:** completed fixed-final records under EfficientAD_S_Anomalib_results_20261023. The saved evaluator selects the first lexicographically sorted defective test image. The input and GT are native-resolution renderings; GT uses >0, and the calibrated map is interpolated to native resolution before its original display. The original jet display uses per-image min/max, not a shared score or decision scale. Crop rectangles are (21,35,539,554), (581,35,1099,554), (1141,35,1659,554), from 2240×560 plots. This is the Anomalib implementation, not an assertion of reproducing the original authors' implementation. [Per-category provenance](figures/efficientad_s_all15_sources.csv).

**DRAEM:** completed final700 records under DRAEM_MVTec_all15_20261015. Each saved visual_selection.json records one defective image randomly sampled before scores with seed 1024. These are the original samples, not substitutions to match another model. The recorded model uses BGR/255, cv2 linear resizing to 256×256; saved input panels convert BGR to RGB for display. GT retains the evaluator's uint8-then-bool conversion, which drops fractional boundary pixels. The inferno map preserves the original fixed anomaly-probability range **[0,1]**; no per-image normalization is introduced. Crop rectangles are (40,74,560,594), (640,74,1160,594), (1240,74,1760,594), from 2400×600 plots. [Per-category provenance](figures/draem_all15_sources.csv).

Source panels are fitted into slots with aspect ratio preserved and centered, using Lanczos for input/maps and nearest-neighbor for GT. Plot overlays/titles are omitted; no map is recolored, thresholded or recomputed. A one-pixel raster-boundary difference in EfficientAD-S is retained through aspect-preserving fitting. The source CSVs record each sample, source/completion/metric/selection/rendering-code hash, and exact input/output rectangles; these are workspace-relative provenance identifiers rather than bundled large outputs.

These overviews illustrate category coverage, not general performance or a controlled comparison. Methods differ in samples, preprocessing, GT handling and map scales; even DRAEM's fixed probability colors do not establish calibration across category models. The [MVTec AD attribution and CC BY-NC-SA 4.0 terms](#mvtec-ad-attribution-and-figure-license) above apply to both new dataset-derived figures.

## SimpleNet all-category overview

[Full-resolution figure](figures/simplenet_all15.png) · [Per-category provenance](figures/simplenet_all15_sources.csv) · [Method](../methods/simplenet/README.md)

Added **27 October 2026**, using the completed fixed-final CUDA run dated 20 October 2026. All 15 comparison.png files, training completion statuses, 40-call training histories and published category metric records were checked. Each saved visualization.json specifies the lexicographically first anomalous test image, fixed independently of scores. The original selections are retained, including imperfect localization and background responses. This is illustrative category coverage, not random sampling or an estimate of general performance.

The 3000×2060 layout, category ordering, typography and 280px slots match the SuperSimpleNet overview. From each 1820×520 source plot, complete input, GT and map rectangles are (20,66,451,497), (470,66,901,497) and (920,66,1351,497). Input/GT retain the original evaluation preprocessing (resize 329, center crop 288, with >0.5 GT binarization); original image denormalization is preserved. No native-resolution image is substituted. GT tiles use nearest-neighbor resizing and other tiles Lanczos, preserving aspect ratio. Original titles/overlay panels are omitted.

Maps retain the saved evaluator's jet coloring and **per-image min/max display scale**. No recoloring, numerical renormalization, thresholding, inference or metric reevaluation was performed. Identical colors do not indicate equal scores across samples or methods. The provenance CSV records source plots, selection records, completion statuses, metrics and rendering-code hashes, plus exact panel rectangles. Workspace-relative source identifiers refer to the research archive.

This newer fixed-final run must not be confused with historical SimpleNet CPU experiments that selected checkpoints using test metrics. Cross-method differences in preprocessing, sample selection and map scales prevent a matched-protocol interpretation. The [MVTec AD attribution and CC BY-NC-SA 4.0 figure license](#mvtec-ad-attribution-and-figure-license) above also apply to simplenet_all15.png.
