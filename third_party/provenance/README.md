# Provenance records

- [Copy manifest](copy_manifest.csv): exact source/destination paths, byte counts and hashes for copied files. Sources are release-root-relative.
- [Original archive inventory](archive_inventory.csv): complete backup inventory with retained/excluded disposition.
- [Original source inventory](original_source_code_inventory.csv) and [original 22 October copy inventory](original_third_party_update_20261022.csv): historical records retained unchanged; paths describe original layouts.
- [SimpleNet snapshot comparison](simplenet_snapshot_comparison.csv): base and target hashes for the derived patch.
- [Remote link checks](remote_link_checks.csv): browser availability observations, not source verification.
- [File manifest](files.csv): hashes for every lightweight file except this manifest itself.
- Source-specific original metadata and local status summaries are under records/, indexed by [SOURCES.csv](../SOURCES.csv).

No old release hashes were rewritten. Historical README hash discrepancies in the original archive remain historical. The new folder has its own manifest.
