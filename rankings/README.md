# COCO and RealSR ranking studies

[`recompute_statistics.py`](recompute_statistics.py) validates the locked protocols, reruns the COCO and RealSR analyses, and compares the results against the archived reports. [`analyze_ranking.py`](analyze_ranking.py) computes the COCO statistics and [`camera_ranking_statistics.py`](camera_ranking_statistics.py) computes the RealSR statistics. The input preparation, model inference, scoring, and pixel replay code is also provided here for inspection.

From the repository root, run `python reproduce.py --experiment rankings`. The report covers 144 COCO scenes and 32 RealSR scenes. The ranking ZIP includes the manifests, frozen locks, per-scene scores, and a `RESULTS.md` with exact counts and verification scope. [Frozen experiment records](../README.md#frozen-experiment-records)
