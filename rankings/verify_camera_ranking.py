"""Audit RealSR input and output provenance, rank counts, and sampled scores."""
from collections import Counter
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from ranking_common import CONDITIONS, METRICS, METHODS, MODELS, ROOT, atomic_json, file_sha, load, verify_lock
from dev_pipeline import pixel_sha
from ranking_models import artifact
from ranking_scores import Scorer
from verify_ranking import independent_crop, recompute_pixel_metrics


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def check_inputs(row, checks):
    for source, digest_field, crop, field in (
        ('source', 'source_sha256', 'hr_crop_xyxy', 'hr'),
        ('camera_lr_source', 'camera_lr_source_sha256', 'lr_crop_xyxy', 'lr'),
    ):
        assert file_sha(row[source]) == row[digest_field]
        assert file_sha(row[field]) == row[f'{field}_sha256'] == checks[f'{field}_sha256']
        x1, y1, x2, y2 = row[crop]
        assert np.array_equal(rgb(row[source])[y1:y2, x1:x2], rgb(row[field]))


def check_native_scores(cases, images, original_hr, scorer, errors):
    pairs = {}
    for model in MODELS:
        case = cases['zero', model]
        sr = independent_crop(images[model])
        for method in METHODS:
            hr = independent_crop(original_hr, *(case['actions'][method] or [0, 0]))
            pair = f'{pixel_sha(sr)}:{pixel_sha(hr)}'
            assert case['score_pair_ids'][method] == pair
            pairs.setdefault(pair, (sr, hr, case['scores'][method]))

    values = scorer.evaluate([(sr, hr) for sr, hr, _ in pairs.values()])
    for fresh, (sr, hr, stored) in zip(values, pairs.values(), strict=True):
        pixels = recompute_pixel_metrics(sr, hr)
        diffs = np.abs([*pixels, fresh['SHAFE'], fresh['ST-LPIPS']] - np.array([stored[m] for m in METRICS]))
        assert np.all(diffs < [1e-9, 1e-9, 2e-6, 2e-6])
        np.maximum(errors, diffs, out=errors)
    return len(pairs)


def main():
    cv2.setNumThreads(1)
    parent, parent_digest = verify_lock()
    folder = ROOT / 'camera_ranking'
    lock_path = folder / 'lock.json'
    lock, digest = load(lock_path), file_sha(lock_path)
    assert lock['coco_lock_sha256'] == parent_digest
    assert all(file_sha(source) == expected for source, expected in lock['sha256'].items())
    records = lock['records']
    assert len({row['scene'] for row in records}) == len(records) == 32
    assert Counter(row['dataset'] for row in records) == {'Canon': 16, 'Nikon': 16}
    audit_scenes = set()
    for camera in ('Canon', 'Nikon'):
        names = sorted(row['scene'] for row in records if row['dataset'] == camera)
        audit_scenes.update((names[0], names[-1]))

    tolerances = {'strict': parent['ranking']['epsilon'],
                  'wider_ties': parent['ranking']['wider_tie_sensitivity']}
    counts = Counter()
    errors = np.zeros(len(METRICS))
    scorer = None
    rescored = 0
    historical_vosr_matches = 0
    for row in records:
        scene = row['scene']
        result = load(folder / 'scores' / f'{scene}.json')
        checks = result['checks']
        assert (checks['config'], result['scene'], result['split']) == (digest, scene, row['dataset'])
        assert result['lr_pixel_sha256'] == row['lr_pixel_sha256'] == pixel_sha(rgb(row['lr']))
        check_inputs(row, checks)

        images = {}
        for model in MODELS:
            path = artifact(folder, model, scene)
            stamp = load(path.with_suffix('.json'))
            assert stamp == dict(config=digest, lr_sha256=row['lr_sha256'], image_sha256=file_sha(path))
            assert stamp['image_sha256'] == checks['sr_sha256'][model]
            images[model] = rgb(path)
            assert images[model].shape == (512, 512, 3)
        historical_vosr_matches += bool(np.array_equal(images['VOSR'], rgb(row['sr'])))

        cases = {(case['condition'], case['model']): case for case in result['cases']}
        assert len(cases) == len(result['cases']) == len(CONDITIONS) * len(MODELS)
        for condition in CONDITIONS:
            by_model = [cases[condition, m] for m in MODELS]
            for anchor in ('lk_lr', 'global_lr'):
                assert len({tuple(case['actions'][anchor]) for case in by_model}) == 1
            hr_added = row['hr_added'] if condition in ('hr_global', 'both_global') else [0, 0]
            sr_added = row['sr_added'] if condition in ('sr_global', 'both_global') else [0, 0]
            for case in by_model:
                assert (case['hr_added'], case['sr_added']) == (hr_added, sr_added)
            if condition == 'zero':
                for model, case in zip(MODELS, by_model, strict=True):
                    assert case['sr_rgb_pixel_sha256'] == pixel_sha(images[model])

        for a, b in combinations((cases['zero', model]['scores'] for model in MODELS), 2):
            for metric in METRICS:
                direct = a['direct'][metric] - b['direct'][metric]
                lk = a['lk_lr'][metric] - b['lk_lr'][metric]
                for label, tolerance in tolerances.items():
                    ordered = min(abs(direct), abs(lk)) > tolerance[metric]
                    opposite = ordered and direct * lk < 0
                    for group in ('pooled', row['dataset']):
                        counts[group, label, metric, 'opposite'] += opposite
                        counts[group, label, metric, 'both_ordered'] += ordered

        if scene in audit_scenes:
            if scorer is None:
                scorer = Scorer()
            rescored += check_native_scores(cases, images, rgb(row['hr']), scorer, errors)

    results = load(folder / 'analysis.json')
    for (group, label, metric, field), value in counts.items():
        assert results[group]['native_discordance'][label]['direct vs lk_lr'][metric][field] == value
    summary = dict(scenes=len(records), verified_camera_source_files=2 * len(records),
                   rebuilt_input_crops=2 * len(records), verified_sr_artifacts=len(records) * len(MODELS),
                   historical_vosr_pixel_matches=historical_vosr_matches,
                   native_rank_counts={group: {label: {metric: counts[group, label, metric, 'opposite']
                                                       for metric in METRICS} for label in tolerances}
                                       for group in ('pooled', 'Canon', 'Nikon')},
                   pixel_score_audit_scenes=sorted(audit_scenes), rescored_unique_native_pairs=rescored,
                   max_abs_score_error=dict(zip(METRICS, errors.tolist(), strict=True)))
    atomic_json(folder / 'verification.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
