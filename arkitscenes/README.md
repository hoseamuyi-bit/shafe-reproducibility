# ARKitScenes temporal reference study

[`verify_confirmation_statistics.py`](verify_confirmation_statistics.py) reconstructs the visit roster, checks the stored screening and reconstruction records, and replays the primary paired analysis and exact test. It also reproduces the registered cross-metric, alignment, and resampling contrasts. The acquisition, geometry, and scoring scripts are available here for inspection.

From the repository root, run `python reproduce.py --experiment arkitscenes`. The study has 42 scored visits and a 36-visit subset with common round-trip support for the primary contrast. The ARKitScenes ZIP includes the saved scores, pose and frame manifests, and acquisition replay receipts. [Frozen experiment records](../README.md#frozen-experiment-records)
