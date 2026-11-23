"""Test translation estimates under a smooth, spatially limited warp of HR."""

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import random
import sys

import cv2
import numpy as np

import registration_benchmarks as benchmarks
from dev_pipeline import luminance, pixel_sha, project, shift
from experiment import DIRECTIONS
from holdout_analysis import paired_summary
from holdout_run import OLD, ROOT, atomic_dump, images_for, load, sha
from registration import candidate
from robust_controls import gray8, lucas_kanade
from robust_registration import tiled_candidates
from score import BATCH_SIZE, DEVICE, luminance_crop
from verify_holdout import event


DEST = Path(__file__).resolve().with_suffix('')
LOCK = DEST / 'lock.json'
INPUTS = benchmarks.coco_confirm.INPUTS
SIDES = (256, 384)
AMPLITUDES = (4, 16)
TRANSLATIONS = (0, 2)
SEED = 20261130
SCORED = (*benchmarks.METHODS, 'oracle')
GRID_Y, GRID_X = np.indices((512, 512), dtype=np.float32)


def profile(center, diameter):
    radius = np.hypot(GRID_X - center[0], GRID_Y - center[1]) / (diameter / 2)
    ramp = np.clip((radius - .6) / .4, 0, 1)
    weight = .5 + .5 * np.cos(np.pi * ramp)
    weight[radius >= 1] = 0
    return weight


def deform(image, weight, vector):
    dx, dy = np.asarray(vector, dtype=np.float32)
    return cv2.remap(image.astype(np.float32), GRID_X - weight * dx,
                     GRID_Y - weight * dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def condition(diameter, amplitude, translation):
    return f'diameter{diameter}_local{amplitude}_global{translation}'


def freeze():
    benchmarks.verify()
    if LOCK.exists() or (DEST / 'results.json').exists() or (DEST / 'scenes').exists():
        raise ValueError('Deformation experiment is already frozen or run')
    records = load(INPUTS)['records']
    rng = random.Random(SEED)
    parameters = {r['scene']: dict(center=[rng.choice((224, 288)), rng.choice((224, 288))],
                                  local_direction=list(rng.choice(DIRECTIONS)))
                  for r in sorted(records, key=lambda r: r['scene'])}
    tracked = (Path(__file__), benchmarks.PLAN, INPUTS, ROOT / 'registration.py',
               Path(__file__).with_name('robust_registration.py'))
    prior = DEST / 'lock_v1_before_dtype_fix.json'
    revision = dict(previous_sha256=sha(prior), cause='Float32 remap coordinates for integer NumPy input; '
                    'the initial run stopped on the first scene before measuring any outcomes. '
                    'Parameters and experimental design are unchanged.') if prior.exists() else None
    atomic_dump(LOCK, dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Descriptive sensitivity experiment on the same 144 COCO test2017 scenes. '
              'Design frozen before computing any local deformation outcomes.',
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in tracked},
        revision=revision,
        parameters=parameters, seed=SEED, diameters=SIDES, peak_local_hr_pixels=AMPLITUDES,
        added_global_hr_pixels=TRANSLATIONS, methods=SCORED,
        perturbation='Translate the sampling grid locally before adding a global translation to HR. '
                     'Use a constant displacement up to 60% of the disk radius, then cosine taper '
                     'to zero at the disk boundary. The two amplitudes share center and direction. '
                     'The global translation uses the frozen COCO direction. LR and SR are identical '
                     'across all conditions. No pixel intensities or material from another region are pasted in.',
        hypothesis='Compare the robustness of global residual, tile trimming, and local track methods '
                   'when local content moves rather than being replaced. The large support is a stress test.',
        geometry='For no global translation, count any applied correction as false movement. For '
                 '2 px global translation, count only moving actions leaving no more than 0.25 HR px '
                 'per axis as correct. Preserve unavailable cases in the denominator.',
        scoring='SHAFE over the 448 px shared central crop. For each condition the oracle reverses only '
                'the global translation, preserving the deformed HR content and fixed SR.',
        inference='Report descriptive per-condition paired 95% bootstrap intervals (5000 source-scene '
                  'replicates), without decision thresholds or multiplicity adjustment.',
    ))
    print(f'Deformation lock {sha(LOCK)}', flush=True)


def registrations(lr, hr):
    anchor = luminance(lr)
    reference = project(luminance(hr))
    tracks = benchmarks.lk_track_vectors(gray8(anchor), gray8(reference))
    median, count = lucas_kanade(gray8(anchor), gray8(reference))
    assert len(tracks) == count
    dense = benchmarks.dense_estimates(anchor, reference)
    dense['lk_consensus'] = benchmarks.track_consensus(tracks)
    full = candidate(hr, lr)
    tiles = tiled_candidates(hr, lr)['tile_trimmed']
    actions = dict(raw=[0., 0.], median_lk=median)
    for name, fit in (('simple_lr', full), ('tile_trimmed', tiles)):
        actions[name] = fit['vector'] if fit['executable'] else [0., 0.]
    actions.update({m: fit['action'] for m, fit in dense.items()})
    valid = dict(raw=True, simple_lr=full['valid'], tile_trimmed=tiles['valid'],
                 median_lk=count >= benchmarks.MIN_TRACKS, **{m: f['valid'] for m, f in dense.items()})
    return actions, valid, count


def process(record, params, evaluate):
    lr, hr, sr, _ = images_for(record)
    fixed = luminance_crop(shift(sr, [0, 0]))
    fixed_hash = pixel_sha(fixed)
    rows, pairs = [], {}
    direction = np.asarray(record['direction'])
    for diameter in SIDES:
        weights = profile(params['center'], diameter)
        for amplitude in AMPLITUDES:
            deformation = deform(hr, weights, amplitude * np.asarray(params['local_direction']))
            for translation in TRANSLATIONS:
                added = direction * translation
                moving = shift(deformation, added)
                actions, valid, count = registrations(lr, moving)
                actions['oracle'] = (-added).tolist()
                ids = {}
                for method, action in actions.items():
                    cropped = luminance_crop(shift(moving, action) if any(action) else moving)
                    pair = fixed_hash + ':' + pixel_sha(cropped)
                    pairs.setdefault(pair, (fixed, cropped))
                    ids[method] = pair
                rows.append(dict(scene=record['scene'], dataset=record['dataset'], group=record['group'],
                                 condition=condition(diameter, amplitude, translation), diameter=diameter,
                                 local_peak=amplitude, global_magnitude=translation, hr_added=added.tolist(),
                                 hr_pixel_sha256=pixel_sha(moving), lr_pixel_sha256=pixel_sha(lr),
                                 actions=actions, valid=valid, n_tracks=count, pair_ids=ids))
    values = dict(zip(pairs, evaluate(list(pairs.values())), strict=True))
    for row in rows:
        row['scores'] = {method: values[pair] for method, pair in row['pair_ids'].items()}
    return rows, len(pairs)


def summarize(rows):
    by_condition = {}
    comparisons = tuple((m, 'simple_lr') for m in benchmarks.METHODS[2:]) + (('median_lk', 'tile_trimmed'),)
    for name in sorted({r['condition'] for r in rows}):
        cases = [r for r in rows if r['condition'] == name]
        endpoint = 'correct' if cases[0]['global_magnitude'] else 'false'
        outcome = partial(event, endpoint=endpoint)
        geometry = paired_summary(cases, benchmarks.METHODS, outcome, comparisons)
        geometry['endpoint'] = endpoint
        geometry['counts'] = {m: sum(outcome(r, m) for r in cases) for m in benchmarks.METHODS}
        mae = paired_summary(cases, benchmarks.METHODS, lambda r, m: abs(r['scores'][m] - r['scores']['oracle']),
                             comparisons)
        unavailable = {m: sum(not r['valid'][m] for r in cases) for m in benchmarks.METHODS}
        by_condition[name] = dict(geometry=geometry, shafe_mae=mae, unavailable=unavailable)
    return by_condition


def run():
    benchmarks.verify()
    plan = load(LOCK)
    for name, digest in plan['hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Deformation inputs changed: {name}')
    cv2.setNumThreads(1)
    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT) == load(benchmarks.coco_confirm.PLAN)['shafe']['weight_sha256']
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)

    def evaluate(pairs):
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    rows, unique_pairs = [], 0
    records = load(INPUTS)['records']
    assert len(records) == len(plan['parameters']) == 144
    for number, record in enumerate(records, 1):
        scene = record['scene']
        path = DEST / 'scenes' / f'{scene}.json'
        stamp = dict(lock_sha256=sha(LOCK), sr_sha256=sha(record['sr']))
        cache = load(path) if path.exists() else {}
        if cache.get('stamp') != stamp:
            result, count = process(record, plan['parameters'][scene], evaluate)
            cache = dict(stamp=stamp, rows=result, unique_pairs=count)
            atomic_dump(path, cache)
        rows.extend(cache['rows'])
        unique_pairs += cache['unique_pairs']
        if number % 12 == 0:
            print(f'{number}/{len(records)} scenes, {unique_pairs} SHAFE pairs', flush=True)
    assert len(rows) == len({(r['scene'], r['condition']) for r in rows}) == 1152
    results = summarize(rows)
    atomic_dump(DEST / 'results.json', dict(lock_sha256=sha(LOCK), unique_pairs=unique_pairs,
                                           strata=results, rows=rows))
    for name, result in results.items():
        print(name, result['geometry']['counts'], flush=True)


def self_test():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    weights = profile((224, 288), 256)
    assert weights[288, 224] == 1 and weights[288, 224 + 128] == 0
    assert 0 < weights[288, 224 + 100] < 1
    np.testing.assert_array_equal(deform(image, weights, [0, 0]), image)
    warped = deform(image, weights, np.array([4, -4], dtype=np.int64))
    np.testing.assert_array_equal(warped[weights == 0], image[weights == 0])
    np.testing.assert_array_equal(warped[284, 228], image[288, 224])
    shifted = shift(warped, [2, 0])
    np.testing.assert_array_equal(luminance_crop(shift(shifted, [-2, 0])), luminance_crop(warped))
    print('Passed support, sign, zero-warp and oracle checks', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('self-test', 'freeze', 'run'))
    args = parser.parse_args()
    {'self-test': self_test, 'freeze': freeze, 'run': run}[args.step]()
