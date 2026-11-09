# Automatic DeSTSeg queue

User-authorized 2026-11-03: RD++ success15/15 and worker exit -> batch32 smoke -> finite losses/gradients, unchanged teacher, strict checkpoint reload and normal-only map export -> all15 fixed-final category trainings. No target test evaluation. Official architecture/loss/data source unchanged; seed111 reset independently per category is declared in plan. Smoke6steps is not used as full initialization. Source/environment/data hashes checked before launch and each attempt. Common predictive evaluation remains a separate task.

The queue survives agent exit. Windows shutdown, logout or reboot may stop it; no boot task exists. Stay-awake is requested while the queue lives. Duplicate monitor/worker launches are rejected. Failures stop progression; no silent retries or hyperparameter changes. A failed category can later restart from step0 in a new attempt. Completed categories are skipped only after artifact verification.

From the project root in PowerShell:
```powershell
$r = 'anomaly_detection/benchmark_2026/experiments/destseg_auto_mvtec_seed111_20261103'
Get-Content "$r/progress.json"
Get-Content "$r/queue.stdout.log" -Tail 20 -Wait
```
During training, progress.json names the active attempt; its logs/progress.json records steps, losses, memory and elapsed time. Per-attempt stdout/stderr are alongside attempt directories. Completion hashes and per-category resources are saved separately. Full inference latency and predictive metrics are Not recorded until a separate evaluation.

To stop the queue and its active training child while retaining artifacts:
```powershell
New-Item -ItemType File -Path "$r/STOP"
```
A manual restart requires confirming prior queue/worker exit, rereading the protocol and reviewing the failure; no automatic restart is configured. Do not delete attempts. The STOP file must be deliberately removed before a resumed queue. Use the isolated environments/destseg/Scripts/python.exe with destseg_queue.py and new redirected log filenames.

Dry-run validation passed before activation: five state-transition cases, all pinned assets, no GPU allocation or training launch. This does not replace the deferred GPU smoke. Safety ceilings: wait7days; smoke15min; training48hours charged wall time; allocatedVRAM14GiB, sampledRSS16GiB; initial disk12GiB, runtime floor2GiB. Checkpoint/export gate uses only normal training images. A failed smoke leaves full training unstarted.

2026-11-04 update: user approved120hour total run ceiling, superseding48hours. Other safety limits unchanged. Verified successful smoke attempt002 reused; full schedule unchanged. Use destseg_queue.py.
