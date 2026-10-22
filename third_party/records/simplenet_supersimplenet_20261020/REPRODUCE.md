# Reproduce

Run from the workspace root. Preserve the original `archive` dataset directory and cached ImageNet V1 weights; SHA256 manifests are in `preflight`.

```powershell
python -m venv SimpleNet_SuperSimpleNet_setup_20261020/venv
SimpleNet_SuperSimpleNet_setup_20261020/venv/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r SimpleNet_SuperSimpleNet_setup_20261020/pip_freeze.txt
SimpleNet_SuperSimpleNet_setup_20261020/venv/Scripts/python.exe -u SimpleNet_SuperSimpleNet_setup_20261020/experiment.py controller
```

Do not launch a second controller while one is active. PID is in `controller.pid`. For an independent rerun, copy the workspace layout into a new location (or change the dated output roots explicitly); the controller skips categories with existing metric files. Recovery checkpoints contain model, optimizer/scheduler, epoch and RNG state but automatic interrupted-epoch resume is NOT implemented. Inspect and preserve partial outputs before restarting.

The controller checks both models before full training. Any error stops the sequence and is recorded in `status.json` and per-category logs. It never changes hyperparameters based on results. Keep Windows awake and retain sufficient free disk space during the long run.

Training outputs: `SimpleNet_MVTec_training_20261020` and `SuperSimpleNet_MVTec_training_20261020`.
Evaluation outputs: `SimpleNet_MVTec_results_20261020` and `SuperSimpleNet_MVTec_results_20261020`.
Report: `reports/simplenet_supersimplenet_20261020` (updates after each completed category; full means only after all 15).

SimpleNet uses the unmodified official discriminator training function, called 40 times with 4 GAN epochs each. SuperSimpleNet's exact training-code changes are in `supersimplenet_training.patch`. Source commit IDs and hashes are in `versions.json` and `source_hashes.json`. timm 0.6.7 satisfies anomalib's supported version range and avoids a Python 3.11 dataclass import issue in timm 0.6.12; both methods use torchvision WRN50 weights, not timm backbones.
