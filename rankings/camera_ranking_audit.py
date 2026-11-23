"""Reuse the locked four-method scoring on published RealSR x4 camera pairs."""
import argparse
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from ranking_common import MODELS, NEXT, ROOT, atomic_json, file_sha, load, verify_lock
from camera_ranking_statistics import analyze_camera
from dev_pipeline import pixel_sha, read, save
from ranking_models import artifact, pending_outputs, run_vosr, stamp, torch_outputs
from ranking_scores import Scorer, score_scene


FOLDER = ROOT / 'camera_ranking'
INPUT_MANIFEST = NEXT / 'external/batch_inputs.json'


def freeze_or_verify():
    _, parent = verify_lock()
    lock_path = FOLDER / 'lock.json'
    if lock_path.exists():
        lock = load(lock_path)
        if lock['coco_lock_sha256'] != parent:
            raise ValueError('Parent experiment lock changed')
        for name, digest in lock['sha256'].items():
            if file_sha(name) != digest:
                raise ValueError(f'Camera source changed: {name}')
    else:
        if (FOLDER / 'scores').exists() or (FOLDER / 'sr').exists():
            raise ValueError('Camera outputs already exist before the lock')
        rows = load(INPUT_MANIFEST)['records']
        if len({r['scene'] for r in rows}) != 32 or Counter(r['dataset'] for r in rows) != {'Canon': 16, 'Nikon': 16}:
            raise ValueError('Expected all 32 previously registered RealSR scenes')
        records = []
        for row in rows:
            records.append(dict(row, split=row['dataset'], hr_added=[row['magnitude'] * n for n in row['hr_direction']],
                                sr_added=[row['magnitude'] * n for n in row['sr_direction']]))
        files = [INPUT_MANIFEST, ROOT / 'camera_ranking_audit.py', ROOT / 'camera_ranking_statistics.py']
        lock = dict(coco_lock_sha256=parent, records=records, bootstrap_seed=20261202,
                    primary='Descriptive native direct versus LK paired order disagreement on SHAFE. '
                            'All six pairs per scene. Report strict and wider tie thresholds.',
                    secondary='Other metrics, each camera, fixed and global LR comparisons, '
                              'and the sensitivity to adding known synthetic HR or SR shifts.',
                    reference='In injected conditions the control undoes only the added HR shift. '
                              'This returns to the published geometry up to interpolation. '
                              'No independently measured native offset or human rating is available.',
                    sha256={str(p): file_sha(p) for p in files})
        atomic_json(lock_path, lock)
    return lock['records'], file_sha(lock_path)


def verify_inputs(rows):
    for row in rows:
        for image, suffix in [('source', 'source_sha256'), ('camera_lr_source', 'camera_lr_source_sha256'),
                              ('hr', 'hr_sha256'), ('lr', 'lr_sha256')]:
            if file_sha(row[image]) != row[suffix]:
                raise ValueError(f'Camera input changed: {row[image]}')
        for source, prepared, crop in [('source', 'hr', 'hr_crop_xyxy'),
                                       ('camera_lr_source', 'lr', 'lr_crop_xyxy')]:
            x1, y1, x2, y2 = row[crop]
            rebuilt = read(row[source])[y1:y2, x1:x2]
            if not np.array_equal(rebuilt, read(row[prepared])):
                raise ValueError(f'Camera crop failed source verification: {row[prepared]}')
        if pixel_sha(read(row['lr'])) != row['lr_pixel_sha256']:
            raise ValueError(f"Camera LR pixel hash changed: {row['scene']}")


def infer(rows, digest):
    for model in MODELS:
        pending = list(pending_outputs(rows, FOLDER, model, digest))
        if not pending:
            continue
        if model == 'VOSR':
            run_vosr(pending, FOLDER, digest)
        elif model == 'Bicubic':
            for row, path in pending:
                with Image.open(row['lr']) as image:
                    enlarged = image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                    save(path, np.asarray(enlarged))
                stamp(path, row, digest)
        else:
            torch_outputs(model, pending, digest)


def score(rows, digest):
    cv2.setNumThreads(1)
    scorer = None
    for i, row in enumerate(rows, 1):
        scene = row['scene']
        paths = {model: artifact(FOLDER, model, scene) for model in MODELS}
        checks = dict(config=digest, hr_sha256=row['hr_sha256'], lr_sha256=row['lr_sha256'],
                      sr_sha256={m: file_sha(path) for m, path in paths.items()})
        destination = FOLDER / 'scores' / (scene + '.json')
        if destination.is_file() and load(destination)['checks'] == checks:
            continue
        if scorer is None:
            scorer = Scorer()
        atomic_json(destination, dict(checks=checks, **score_scene(row, paths, scorer)))
        if i % 8 == 0:
            print(f'RealSR scoring: {i}/{len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    options = parser.parse_args()
    rows, digest = freeze_or_verify()
    verify_inputs(rows)
    print(f'RealSR lock verified: {digest}', flush=True)
    if options.freeze_only:
        return
    infer(rows, digest)
    score(rows, digest)
    results = analyze_camera()
    atomic_json(FOLDER / 'analysis.json', results)
    print(results['pooled']['native_discordance']['strict']['direct vs lk_lr'], flush=True)


if __name__ == '__main__':
    main()
