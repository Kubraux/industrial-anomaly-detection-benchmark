# MSFlow full training
Approved release-CLI batch8, seed9826 per category,100epochs,FP32;512 except transistor256. Paper batch16 difference retained. Active root protocol v1.5 and unchanged official source/worker/environment dependencies pinned in pins.json. Configuration contains inherited smoke-only fields (smoke limits are not used in full mode); the coordinator enforces cumulative24hours. No test evaluation or model selection.

Windows venv launches a Python wrapper and actual interpreter; progress.json records the actual coordinator and worker PIDs, launch.json records the initial wrapper PID. Persistent hidden local process continues after the agent ends. Windows execution request prevents idle system sleep; explicit sleep/shutdown/reboot still interrupts.

Read progress.json for category, completed list, update count and current log paths. categories/<category>/attempt_NNN.stdout.log contains live updates; final.pt and resource_usage.json appear only after all100epochs complete. No inference/export is scheduled automatically.

Resume from repository root after the previous process has stopped:
```cmd
anomaly_detection\benchmark_2026\environments\msflow\Scripts\python.exe -B -u anomaly_detection\benchmark_2026\experiments\msflow_full_mvtec_seed9826_20261109\run_all.py
```
The coordinator rejects duplicate/orphan workers, verifies completed checkpoint/config/resource hashes, skips completed categories and restarts incomplete categories in new attempt directories. It never overwrites historical attempts. No mid-epoch recovery. Budget persists across restart; no automatic extension or setting fallback.

GPU smoke measured8.014GiB allocated/8.676GiB reserved. Initial full-run updates confirmed the same peaks. Rough runtime8hours extrapolated, not a guarantee; measured per-category times saved on completion. Final evaluator still requires independent validation and disk preflight. Common primary additive pixel map and supplementary reference multiplicative PRO are prospectively registered in plan.md.
