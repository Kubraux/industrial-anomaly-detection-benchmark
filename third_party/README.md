# Lightweight third-party source catalog

Prepared on 22 October 2026 from the existing release archive. The original third_party directory remains unchanged as the backup; this folder does not replace it automatically. Backup verification accepts its explicit location.

## Start here

- [SOURCES.csv](SOURCES.csv): source records for the documented methods, including distinct older and 20 October SimpleNet copies.
- [Reconstruction limits](RECONSTRUCTION_LIMITS.md): missing revisions, incomplete patches and evaluation caveats.
- [Licenses and notices](licenses/README.md): original texts, copied without changes.
- [Local modifications](local_changes/README.md): selected experiment additions; upstream model and dataset trees are excluded.
- [Patches](patches/README.md): four historical patches and one newly derived snapshot delta.
- [Provenance](provenance/README.md): copy hashes, original records, backup inventory and link checks.

Scope is the union of methods referenced by the existing archive and its linked method records, not every literature-only entry. Repository URLs and commit SHAs are copied from existing provenance. A repository reference is not automatically an official author implementation: the recorded PaDiM repository identifies itself as unofficial, and AnoGAN includes an unofficial adaptation reference. AE-DPAE has no verified upstream repository; that field stays empty. No alternative upstream is invented.

## Retained and excluded

Retained: ten license records plus PatchCore NOTICE; source statuses, revisions and change summaries; existing RD4AD, Skip-GANomaly and SuperSimpleNet patches; a clearly labeled SimpleNet snapshot diff; selected local runners, audits, metric and visualization helpers; the newer SimpleNet/SuperSimpleNet orchestration and dependency record; essential hashes and original reproduction notes.

Excluded from this new folder: bundled upstream model, backbone, dataset and training source trees; duplicate upstream READMEs and omitted artwork; duplicate license copies beside working trees; experiment results, datasets, checkpoints, environments and papers. Exclusion from this folder does not delete anything from the backup or release. The archive inventory lists each original file's disposition.

## Links, preservation and maintenance

Repository and recorded commit URLs were checked through the browser; [results](provenance/remote_link_checks.csv) distinguish retrieved pages from the DFR commit URL's cache miss. That failure does not establish that the commit is invalid. No remote checkout or code equivalence is inferred from a page being accessible.

All newly authored Markdown navigation and SOURCES.csv local references stay inside this folder. Historical source/hash records, script imports, patch headers and REPRODUCE.md contain original workspace paths; these are historical evidence, not current navigational links or bundled-file promises. Copy-manifest source paths are relative to the release root. Archive-inventory paths are relative to the original third_party directory.

From this folder, run "python -B validate.py" for offline checks of the inventory, hashes, catalog consistency and authored links. To also check an unchanged original backup, run "python -B validate.py --backup-dir /path/to/original/archive" (quote paths containing spaces). For the current layout, "python -B validate.py --backup-dir ../third_party" works when run from this folder. Relative backup paths resolve from the current working directory; absolute paths are also accepted. No sibling location is assumed, and omitting --backup-dir skips backup verification. Neither command executes research code or accesses the network.

Catalog checks require nonempty unique method_id values, nonempty method names, distinct record directories matching the available STATUS.md records, and consistent upstream identities, repository URLs, commit SHAs and commit URLs. Multiple source variants may share a method name. Method and source counts are computed from the catalog, not fixed; new methods can be added with their records and manifest updates.

For future updates, add a dated source record and exact source evidence; keep unknown values explicit. Retain license text with any local patch containing upstream code. Update the local copy and file manifests together with changes; never overwrite historical source records to make them appear current. The original release manifest is unchanged and does not cover this separate folder.
