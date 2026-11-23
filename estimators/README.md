# Estimator and registration studies

[`recompute.py`](recompute.py) computes the gate operating point, paired intervals, SHAFE score differences, and the response to known translations from the formal, external, and transfer case tables. [`registration.py`](registration.py) and [`gate.py`](gate.py) contain the registration and action-selection implementation. The `verify_*.py` scripts check the archived estimator replay, the DIV2K holdout, and the COCO and RealSR confirmation and registration extensions. [`pixel_metric_audit.py`](pixel_metric_audit.py) and [`rank_audit.py`](rank_audit.py) check the metric comparisons and two-model rankings.

From the repository root, run `python reproduce.py --experiment estimators`. The launcher maps these source files into the archived layout before running the audits. See the estimator ZIP's `README.md` for the source manifests and pixel-level entry points. [Frozen experiment records](../README.md#frozen-experiment-records)

For scripts that use the earlier development data, set `SHAFE_PRIOR_ROOT` to the workspace with the source inputs and weights. The statistics replay uses the original development module from the pinned archive.
