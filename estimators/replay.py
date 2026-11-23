"""Rerun targeted geometry and official SHAFE scoring without using cached results."""

from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev_pipeline import OLD, dump, read, sha
from experiment import load, variants
from freeze import LOCK, verify
from gate import actions
from measure import measure_record
from registration import candidate
from score import BATCH_SIZE, DEVICE, score_record

sys.path.insert(0, str(OLD / "work/shafe_formal"))
from run_formal import paper_scores
from run_shafe import DEFAULT_WEIGHT, load_metric


SELECTED = {
    "formal": {
        ("DIV2K_0845", "strong"), ("DIV2K_0880", "strong"),
        ("Urban100_img_008_SRF_4_HR", "strong"), ("Urban100_img_035_SRF_4_HR", "strong"),
    },
    "external": {("RealSR_Canon_001", "camera"), ("RealSR_Nikon_003", "camera")},
}


def key(row):
    return row["scene"], row["degradation"], row["condition"]


def main():
    cv2.setNumThreads(2)
    frozen = verify()
    weight_hash = sha(DEFAULT_WEIGHT)
    if weight_hash != frozen["weights"]["shafe"]["sha256"]:
        raise ValueError("Official metric weight drift")
    gates_path = ROOT / "calibration/gates.json"
    gates = load(gates_path)
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = DEVICE
    metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    audited = []
    for split, selected in SELECTED.items():
        base = ROOT / split
        inputs = load(base / "batch_inputs.json")["records"]
        records = [r for r in inputs if (r["scene"], r["degradation"]) in selected]
        if len(records) != len(selected):
            raise ValueError(f"Missing replay inputs for {split}")
        saved = {key(r): r for r in load(base / "decisions.json")["rows"]}
        saved_scores = {key(r): r for r in load(base / "scores.json")["rows"]}
        for record in records:
            measured = measure_record(record, reference=False, include_direct=True)
            for observed in measured:
                original = saved[key(observed)]
                for field in ("hr_added", "sr_added", "candidate_correct", "direct_vector",
                              "lr_pixel_sha256", "hr_pixel_sha256", "sr_pixel_sha256"):
                    assert observed[field] == original[field], (key(observed), field)
                for side in ("hr_fit", "sr_fit"):
                    for field in ("vector", "tile_vectors", "executable"):
                        assert observed[side][field] == original[side][field], (key(observed), side, field)
                vectors, confidence, _, _, _ = actions(observed, gates)
                assert vectors == original["actions"], key(observed)
                for name in confidence:
                    target = original["confidence"][name]
                    value = confidence[name]
                    assert value is target if target is None else abs(value - target) <= 1e-6

            packed = {row["condition"]: row for row in measured}
            scores, pair_count = score_record(
                record, packed, gates, lambda pairs: paper_scores(metric, pairs, DEVICE, BATCH_SIZE))
            max_error = 0
            for row in scores:
                original = saved_scores[key(row)]
                assert row["score_pair_ids"] == original["score_pair_ids"], key(row)
                error = max(abs(original["scores"][name] - value) for name, value in row["scores"].items())
                assert error <= 1e-5, (key(row), error)
                max_error = max(max_error, error)
            result = {"split": split, "scene": record["scene"], "degradation": record["degradation"],
                      "n": len(scores), "unique_shafe_pairs": pair_count, "max_shafe_difference": max_error}
            if record["scene"] == "DIV2K_0845":
                lr, hr, sr, source = [read(record[name]) for name in ("lr", "hr", "sr", "source")]
                replaced = next(h for label, h, _, _, _ in variants(record, hr, sr, source)
                                if label == "hr_replacement")
                slower = candidate(replaced, lr, reference=True)
                fast = packed["hr_replacement"]["hr_fit"]
                assert slower["vector"] == fast["vector"]
                assert abs(slower["residual"] - fast["residual"]) <= 1e-6
                result["explicit_projection_check"] = {"vector": slower["vector"],
                                                       "residual_difference": abs(slower["residual"] - fast["residual"])}
            audited.append(result)
            print(result, flush=True)
    dump(Path(__file__).with_name("fresh_replay.json"),
         {"formal_lock_sha256": sha(LOCK), "gates_sha256": sha(gates_path), "weight_sha256": weight_hash,
          "device": torch.cuda.get_device_name(), "replayed": audited})


if __name__ == "__main__":
    main()
