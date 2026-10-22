# Patch records

- [RD4AD main.py](rd4ad/main.py.patch) and [test.py](rd4ad/test.py.patch): historical local patches; not a complete verified upstream diff.
- [Skip-GANomaly audit](skip_ganomaly/audit_changes.patch): historical before/after audit; base revision unknown.
- [SuperSimpleNet training](supersimplenet/supersimplenet_training.patch): preserved official/runtime patch.
- [SimpleNet newer-to-older snapshot](simplenet/newer_to_older_snapshot.patch): generated only from shared Python files in the two existing archived copies. Apply direction is newer archived copy to older archived copy. [Base/target hashes](../provenance/simplenet_snapshot_comparison.csv) identify the compared files.

The first four files are unchanged historical evidence. The fifth is a derived record, not a recovered historical patch. See [limits](../RECONSTRUCTION_LIMITS.md) before using any patch. All corresponding [license records](../licenses/README.md) are retained.
