# Running RD++ — 2026-11-02
Persistent launcher PID40752; actual Python worker PID22760. Consult process.json and progress.json for current process/status; PIDs can change on resume.

From the project root in PowerShell:
```powershell
$r = 'anomaly_detection/benchmark_2026/experiments/rdpp_full_mvtec_seed111_20261030'
Get-Content "$r/progress.json"
Get-Content "$r/launch_20261102_103706.stdout.log" -Tail 20 -Wait
```
Per-batch progress and epoch are in the active_directory/progress.json named in the top-level progress.json. Per-category loss_trace.jsonl, run.log, config.json, train_manifest.json, final.pt and completion verification are under attempts/<category>/attempt_NNN/<category>.

To resume after the process has stopped, use the same isolated Python and run_all.py with Start-Process -WindowStyle Hidden and NEW redirected stdout/stderr filenames, following a fresh protocol/pre-run check. A Windows file lock rejects duplicate workers. Completed categories are verified and skipped, restoring RNG boundaries; interrupted categories restart from their original boundary in a fresh attempt. No mid-epoch recovery is promised. Final-only model checkpoints; entry/next-category RNG files are not model checkpoints.

The worker is independent of the agent session. It requests Windows stay awake while active; shutdown, restart or sign-out can terminate it. No automatic boot restart is configured. The15 categories run sequentially; no test evaluation is launched. About22hours is the earlier compute-only estimate, excluding I/O/initialization;48hours is the per-invocation safety limit.
