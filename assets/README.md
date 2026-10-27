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

The dataset is licensed under [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/). **supersimplenet_all15.png is shared as a dataset-derived adaptation under that same license.** Changes comprise the experiment's recorded preprocessing and model heatmaps, followed here by cropping existing panels, resizing, rearranging and relabeling. Attribution, the license link, noncommercial use and ShareAlike conditions must accompany reuse. This figure-specific notice does not assign a blanket project or upstream-code license.
