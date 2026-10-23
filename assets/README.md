# Presentation figures

[Gallery and method coverage](GALLERY.md) | [Results](../results/README.md) | [Figure provenance](figures/SOURCES.csv)

All outputs are 1200x675 with Arial typography, fixed title/footer margins and a compact layout. Single-row three-panel figures use 360px square slots; two-row examples use up to244px slots. Larger column counts reduce slot size without stretching image contents. Source-image rectangles retain their aspect ratios and full displayed field of view.

GT and binary masks use nearest-neighbor resizing; other panels use Lanczos. These are reformatted existing plots, not newly inferred model outputs. No missing mask is synthesized. Original color normalization is retained; independent/shared/unknown scales are stated on each figure. Removed source colorbars mean these panels are qualitative illustrations, not numerical score readouts. AnoGAN's original pseudocolored diagnostic inputs remain pseudocolored.

Skip-GANomaly is a standard category-level plot of unchanged AUROC fields, not a localization example. Components are diagnostic, while the recorded combined score remains primary.

SOURCES.csv records source/output SHA-256, original crop rectangles, output panel positions, method/category, selection and scaling notes. Archive-source paths are workspace-relative provenance identifiers, not bundled GitHub links. Original experiments remain unchanged. Prior presentation files and their provenance are retained outside GitHub under publication_archive/visual_refresh_20261023; earlier full outputs remain under publication_archive/results_assets_20261023.
