"""Fetch author-released HR pixels and audit a provisional scene-level split.

This does not evaluate registration or expose formal method outcomes. Perceptual
duplicate candidates remain a review requirement before the split is frozen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import zipfile

import numpy as np
from PIL import Image
from scipy.fft import dctn

HERE = Path(__file__).resolve().parent
AUTHOR_COMMIT = "8f6dd8c1d20cb7e8792a7177b4f6fd677633f598"
DIV_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def urban_item(entry):
    target = HERE / "data" / "Urban100" / Path(entry["path"]).name
    url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/{AUTHOR_COMMIT}/{entry['path']}"
    if target.exists():
        raw = target.read_bytes()
    else:
        raw = subprocess.check_output([
            "curl.exe", "--fail", "--silent", "--show-error", "--location",
            "--retry", "2", "--max-time", "40", url],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    git_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if git_sha != entry["sha"]:
        raise ValueError(f"Git object mismatch: {target.name}")
    with Image.open(io.BytesIO(raw)) as im:
        im.verify()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(raw)
    return {"path": str(target), "url": url, "git_blob_sha1": git_sha}


def fetch_urban():
    tree = json.loads((HERE.parent / "selfex_tree.json").read_text(encoding="utf-8-sig"))
    if tree["sha"] != AUTHOR_COMMIT:
        raise ValueError("Unrecognized author repository revision")
    entries = [e for e in tree["tree"] if e["path"].startswith("data/Urban100/image_SRF_4/")
               and e["path"].endswith("_HR.png")]
    assert len(entries) == 100
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(e, pool.submit(urban_item, e)) for e in entries]
        for entry, future in futures:
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append({"path": entry["path"], "error": str(error)})
    save_json(HERE / "data" / "urban_source.json", {
        "commit": AUTHOR_COMMIT,
        "pixel_scope": "author x4 HR release; source edges cropped to multiples of four; no resampling",
        "files": rows, "failures": failures})
    print(f"Urban100 fetched/verified: {len(rows)}, failures: {len(failures)}", flush=True)
    if failures:
        raise RuntimeError("Incomplete Urban100; rerun to retry missing files")


def extract_div():
    archive = HERE / "data" / "source" / "DIV2K_valid_HR.zip"
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.startswith("DIV2K_valid_HR/") and n.endswith(".png")]
        assert len(names) == 100
        target = HERE / "data" / "DIV2K"
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            raw = zf.read(name)  # includes ZIP CRC verification
            (target / Path(name).name).write_bytes(raw)
    save_json(HERE / "data" / "div_source.json", {"url": DIV_URL,
        "archive_sha256": digest(archive.read_bytes()), "bytes": archive.stat().st_size,
        "image_count": len(names)})
    print(f"DIV2K CRC-verified: {len(names)}", flush=True)


def audit():
    rows, fingerprints, missing = [], {}, []
    for dataset in ("DIV2K", "Urban100"):
        paths = sorted((HERE / "data" / dataset).glob("*.png"))
        if len(paths) != 100:
            missing.append({"dataset": dataset, "expected": 100, "found": len(paths)})
        for path in paths:
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                small = np.array(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
                coeff = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
                phash = coeff > np.median(coeff)
                scene_id = dataset + "_" + path.stem
                fingerprints[scene_id] = phash
                x, y = (w - 560) // 2, (h - 560) // 2
                rows.append({"id": scene_id, "dataset": dataset, "source": str(path),
                    "sha256": digest(raw), "pixel_sha256": digest(np.array(rgb).tobytes()),
                    "width": w, "height": h, "eligible": min(w, h) >= 560,
                    "exclusion": None if min(w, h) >= 560 else "less_than_512_plus_2x24_pixels",
                    "render_crop_xyxy": [x, y, x + 560, y + 560] if min(w, h) >= 560 else None})
    pairs = []
    for i, first in enumerate(rows):
        for second in rows[i+1:]:
            distance = int(np.count_nonzero(fingerprints[first["id"]] != fingerprints[second["id"]]))
            exact = first["pixel_sha256"] == second["pixel_sha256"]
            if exact or distance <= 8:
                pairs.append({"a": first["id"], "b": second["id"], "phash_distance": distance,
                              "exact": exact, "review_status": "pending"})
    # Conservative provisional grouping: all flagged pairs stay together until review.
    parents = {r["id"]: r["id"] for r in rows}
    def root(item):
        while parents[item] != item:
            item = parents[item]
        return item
    for pair in pairs:
        parents[root(pair["b"])] = root(pair["a"])
    assignment, counts = {}, {}
    rng = random.Random(20261122)
    for dataset in ("DIV2K", "Urban100"):
        eligible = [r for r in rows if r["dataset"] == dataset and r["eligible"]]
        groups = {}
        for row in eligible:
            groups.setdefault(root(row["id"]), []).append(row["id"])
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        tally = {"dev": 0, "calibration": 0, "formal": 0}
        for group in group_ids:
            split = "dev" if tally["dev"] < 10 else "calibration" if tally["calibration"] < 15 else "formal"
            for item in groups[group]:
                assignment[item] = split
            tally[split] += len(groups[group])
        counts[dataset] = {"files": len([r for r in rows if r["dataset"] == dataset]),
            "eligible_scenes": len(eligible), "eligible_groups": len(groups), "splits": tally}
    for row in rows:
        row["group"] = root(row["id"])
        row["split"] = assignment.get(row["id"], "excluded")
    save_json(HERE / "data" / "manifest_provisional.json", {"status": "NOT_FROZEN",
        "seed": 20261122, "crop": "original-pixel central 512 plus 24 each side; no upscaling",
        "freeze_requirements": ["review perceptual duplicate candidates and scene identities",
            "record any sample-size revision and precision estimate", "finish development geometry checks"],
        "counts": counts, "missing": missing, "duplicate_candidates": pairs, "scenes": rows})
    print(json.dumps({"counts": counts, "near_duplicate_pairs_pending": len(pairs), "missing": missing}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["urban", "div", "audit"])
    args = parser.parse_args()
    {"urban": fetch_urban, "div": extract_div, "audit": audit}[args.action]()
