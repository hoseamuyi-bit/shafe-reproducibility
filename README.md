# Reference Choice and Registration Can Change Super-Resolution Rankings

This repository accompanies a study of how spatial registration and capture time change full-reference super-resolution rankings when the restorations are held fixed. It contains the experiment source and a CPU-only entry point for recomputing results from frozen per-scene records.

## Recompute the results

With Python 3.11, run these commands from the repository root. The first command installs the verified NumPy version in your Python environment. The replay downloads about 18 MB of frozen records from GitHub, checks each archive's SHA-256, and writes the reports to `reproduced/`.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

To use the three archives already downloaded into a directory, run `python reproduce.py --archives /path/to/archives`. Use `--experiment estimators`, `--experiment rankings`, or `--experiment arkitscenes` to run a single group. Use `--output /path/to/reports` to choose the result directory.

| Source | What the replay computes | Output in `reproduced/` |
| --- | --- | --- |
| [Estimator and registration studies](estimators/) | Gate operating point, paired estimator contrasts, held-out registration checks, and controlled deformation and metric audits | `estimators/` |
| [COCO and RealSR ranking studies](rankings/) | Frozen four-method rankings for 144 COCO scenes and 32 RealSR scenes. The COCO primary contrast is a 20.72 percentage-point reduction in rank reversals. | `rankings/analysis.json` and `rankings/summary.json` |
| [ARKitScenes temporal reference study](arkitscenes/) | Prespecified visit-level test, bootstrap intervals, sensitivity comparisons, and checks of stored camera-file replay records. The primary comparison uses 36 common-support visits. | `arkitscenes/verification.txt` |

## Frozen experiment records

The original protocols, per-scene scores, source manifests, and pixel-replay records are available in [SHAFE Replication Data v1](https://github.com/hoseamuyi-bit/shafe-reproducibility/releases/tag/replication-data-v1). Each ZIP also contains its original README and retains the directory layout used by the archived scripts.

| Experiment | Download | SHA-256 |
| --- | --- | --- |
| Estimators and registration | [`SHAFE_experiment_reproducibility.zip`](https://github.com/hoseamuyi-bit/shafe-reproducibility/releases/download/replication-data-v1/SHAFE_experiment_reproducibility.zip) | `dd2cc42f992dd2e62cd28d9806c80e1bf228c2ea84ddb9b689aa351280f4ac55` |
| COCO and RealSR rankings | [`SHAFE_COCO_RealSR_rankings.zip`](https://github.com/hoseamuyi-bit/shafe-reproducibility/releases/download/replication-data-v1/SHAFE_COCO_RealSR_rankings.zip) | `2006fa0d5cfc1702697e4450ccc848f6ab5bf64e577e34bc04b1bd00af70083a` |
| ARKitScenes confirmation | [`SHAFE_ARKitScenes_confirmation.zip`](https://github.com/hoseamuyi-bit/shafe-reproducibility/releases/download/replication-data-v1/SHAFE_ARKitScenes_confirmation.zip) | `743bcc7030f8520e4a6bec402b5cedf6542e986ca262dd9d957157d3e412d51c` |

The replay checks saved acquisition and pixel-replay records as part of the audits. To rerun model inference or regenerate scores from pixels, obtain the upstream images and weights identified in the corresponding archive, then configure the dataset and model paths in a separate run directory. The archive READMEs give the relevant entry points and environment details.
