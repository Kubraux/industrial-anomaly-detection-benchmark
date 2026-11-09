# efficientad: recorded workspace

Published 2026-11-09. workspace/ preserves selected paths relative to the original project root. Copying this partial snapshot alone is insufficient to run an experiment. Obtain the pinned upstream implementation and declared data/teacher assets separately; use the isolated environment lock files and documented patches. Review the recorded plans and replace local path placeholders before any new execution. Do not invoke upstream reference training/evaluation entrypoints for benchmark selection.

${PROJECT_ROOT} and ${USER_HOME} replace machine-specific absolute prefixes; original file and published-file SHA256 values are in manifest.json. Historical numeric results and configuration values are unchanged. Files omitted intentionally: datasets, checkpoints, predictions, optimizer states, caches, per-image inventories, large logs and unrelated experiments. Original historical records stay local. Source licenses are retained next to copied source (Anomalib license in third_party/licenses). The full upstream repository is identified by pinned commit; source bundles here are partial.

Training-only records correctly retain test-not-run state; final evaluation results are separate under results/benchmarks/. Archived plans describe their historical readiness, not current status. This publication does not activate any new protocol version.
