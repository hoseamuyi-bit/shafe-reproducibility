"""Evaluate the frozen robust candidates on controlled local mismatches in RealSR."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import random
import sys

import cv2
import numpy as np
from PIL import Image

from holdout_analysis import linf, paired_summary
from holdout_data import patch_plan
from holdout_run import BASE, LOCK, OLD, ROOT, atomic_dump, images_for, load, sha, verify_lock
from dev_pipeline import pixel_sha, read, shift
from registration import candidate
from robust_registration import tiled_candidates
from score import BATCH_SIZE, DEVICE, luminance_crop


FOLDER = BASE / 'camera_local'
PLAN = FOLDER / 'lock.json'
INPUTS = ROOT / 'external/batch_inputs.json'
METHODS = ('simple_lr', 'tile_mean', 'tile_trimmed')
CONTRASTS = [('simple_lr', 'tile_trimmed'), ('tile_mean', 'tile_trimmed')]


def freeze():
    verify_lock()
    if PLAN.exists() or (FOLDER / 'geometry.json').exists():
        raise ValueError('Camera local protocol already in use')
    records = load(INPUTS)['records']
    patches = {}
    for row in records:
        rng = random.Random('20261126|camera-local|' + row['scene'])
        with Image.open(row['source']) as image:
            width, height = image.size
        patches[row['scene']] = patch_plan(rng, 256, width, height, *row['hr_crop_xyxy'][:2])
    previous = FOLDER / 'lock_v1.json'
    amendment = None
    if previous.exists():
        assert patches == load(previous)['patches']
        amendment = dict(previous_sha256=sha(previous),
                         reason='Match the archived float32 no-op SR rendering when hashing; '
                                'prior verification failed before writing any measurement. '
                                'Patches, interventions, endpoints and analysis are unchanged.')
    atomic_dump(PLAN, dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        follow_up_lock_sha256=sha(LOCK), input_sha256=sha(INPUTS), code_sha256=sha(__file__),
        source='Same 32 RealSR prealigned test pairs as the camera injection experiment, 16 per camera',
        perturbation='Replace one 256 px square with a distant patch of the same camera HR source. '
                     'Use the same replacement with either zero or 2 HR pixel translation. '
                     'Keep original LR and VOSR output fixed.',
        primary_summary='After subtracting the action for the original camera pair, report the '
                        'fraction of cases with any axis more than 1 HR pixel away from the known '
                        'inverse injection, with scene-level paired bootstrap within camera.',
        secondary='Repeat for continuous candidate vectors and 0.25/0.5/1.0 px tolerances. '
                  'Describe SHAFE absolute score deviation from known inverse injection. '
                  'Test sets quantify incremental injection responses, with the original '
                  'unperturbed pair used as a baseline for the geometry.',
        amendment=amendment, patches=patches))


def verify_plan():
    verify_lock()
    plan = load(PLAN)
    if (plan['follow_up_lock_sha256'], plan['input_sha256'], plan['code_sha256']) != (
            sha(LOCK), sha(INPUTS), sha(__file__)):
        raise ValueError('Camera local protocol or inputs changed')
    return plan


def variants(record, patch, hr):
    full = read(record['source'])
    x, y = patch['target_xy']
    sx, sy = patch['source_xy']
    size = patch['size']
    replaced = hr.copy()
    replaced[y:y+size, x:x+size] = full[sy:sy+size, sx:sx+size]
    zero = [0, 0]
    added = (2 * np.array(record['hr_direction'])).tolist()
    for name, source, vector in [('zero', hr, zero), ('shift2', hr, added),
                                 ('replacement_256', replaced, zero),
                                 ('shift2_replacement_256', replaced, added)]:
        yield name, shift(source, vector), vector


def measure(task):
    cv2.setNumThreads(1)
    record, patch = task
    lr, hr, sr, _ = images_for(record)
    sr_hash = pixel_sha(shift(sr, [0, 0]))
    rows = []
    for name, reference, added in variants(record, patch, hr):
        fits = dict(simple_lr=candidate(reference, lr), **tiled_candidates(reference, lr))
        invalid = [key for key, fit in fits.items() if not fit['valid']]
        vectors = {key: fit['vector'] for key, fit in fits.items()}
        actions = {key: fit['vector'] if fit['executable'] else [0, 0] for key, fit in fits.items()}
        rows.append(dict(scene=record['scene'], group=record['group'], dataset=record['dataset'],
                         condition=name, hr_added=added, hr_pixel_sha256=pixel_sha(reference),
                         sr_pixel_sha256=sr_hash, invalid=invalid, vectors=vectors, actions=actions))
    return rows


def geometry():
    registered = verify_plan()['patches']
    records = load(INPUTS)['records']
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks = [(row, registered[row['scene']]) for row in records]
        rows = [row for group in pool.map(measure, tasks) for row in group]
    archived = {row['scene']:row for row in load(BASE/'camera/geometry.json')['rows']
                if row['condition'] == 'zero'}
    originals = {row['scene']:row for row in rows if row['condition'] == 'zero'}
    for scene, row in originals.items():
        original = archived[scene]
        assert all(row['actions'][m] == original['actions'][m] for m in METHODS)
        assert row['hr_pixel_sha256'] == original['hr_pixel_sha256']
        assert row['sr_pixel_sha256'] == original['sr_pixel_sha256']
    assert len(rows) == len(records) * 4
    atomic_dump(FOLDER/'geometry.json', dict(lock_sha256=sha(PLAN), rows=rows))
    print(f'{len(rows)} cases; clean actions verified against RealSR archive', flush=True)


def score():
    plan = verify_plan()
    sys.path.insert(0, str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT) == load(LOCK)['shafe']['weight_sha256']
    geometry_path = FOLDER/'geometry.json'
    measured = load(geometry_path)
    assert measured['lock_sha256'] == sha(PLAN)
    indexed = {(row['scene'], row['condition']):row for row in measured['rows']}
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    rows = []
    for record in load(INPUTS)['records']:
        _, hr, sr, _ = images_for(record)
        scene = record['scene']
        fixed_sr = shift(sr, [0, 0])
        fixed = luminance_crop(fixed_sr)
        references, records = {}, []
        for name, reference, vector in variants(record, plan['patches'][scene], hr):
            row = indexed[scene, name]
            assert row['hr_pixel_sha256'] == pixel_sha(reference) and row['hr_added'] == vector
            assert row['sr_pixel_sha256'] == pixel_sha(fixed_sr)
            actions = dict(row['actions'], raw=[0, 0], oracle=[-n for n in vector])
            keys = {}
            for method, action in actions.items():
                action = tuple(action)
                key = name, action
                if key not in references:
                    references[key] = fixed, luminance_crop(shift(reference, action) if any(action) else reference)
                keys[method] = key
            records.append((row, keys))
        values = dict(zip(references, paper_scores(metric, list(references.values()), DEVICE, BATCH_SIZE), strict=True))
        rows.extend(dict(row, scores={m:values[k] for m, k in keys.items()}) for row, keys in records)
    atomic_dump(FOLDER/'scores.json', dict(geometry_sha256=sha(geometry_path), rows=rows))


def analyze(include_scores):
    verify_plan()
    geometry_path = FOLDER/'geometry.json'
    measured = load(geometry_path)
    assert measured['lock_sha256'] == sha(PLAN)
    rows = measured['rows']
    original = {r['scene']:r for r in rows if r['condition'] == 'zero'}
    scored = {}
    if include_scores:
        data = load(FOLDER/'scores.json')
        assert data['geometry_sha256'] == sha(geometry_path)
        scored = {(r['scene'], r['condition']):r['scores'] for r in data['rows']}
    report = {}
    for name in ('shift2', 'replacement_256', 'shift2_replacement_256'):
        cases = [row for row in rows if row['condition'] == name]
        outcomes = dict(invalid_cases={m:sum(m in r['invalid'] for r in cases) for m in METHODS})
        for kind in ('actions', 'vectors'):
            def error(row, method):
                baseline = original[row['scene']][kind][method]
                return linf(np.array(row[kind][method]) - baseline + row['hr_added'])

            for threshold in (.25, .5, 1.):
                def exceeds(row, method):
                    return (method in row['invalid'] or method in original[row['scene']]['invalid']
                            or error(row, method) > threshold)

                outcomes[f'{kind}_above_{threshold:g}px'] = paired_summary(cases, METHODS, exceeds, CONTRASTS)
        if scored:
            def score_error(row, method):
                scores = scored[row['scene'], row['condition']]
                return abs(scores[method] - scores['oracle'])

            outcomes['shafe_mae'] = paired_summary(cases, ('raw', *METHODS), score_error, CONTRASTS)
        report[name] = outcomes
    atomic_dump(FOLDER/('full_analysis.json' if scored else 'geometry_analysis.json'),
                dict(lock_sha256=sha(PLAN), geometry_sha256=sha(geometry_path), strata=report))
    for name, outcomes in report.items():
        print(name, outcomes['actions_above_1px']['paired_differences'][0], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('freeze', 'geometry', 'score', 'analyze'))
    parser.add_argument('--scores', action='store_true')
    arguments = parser.parse_args()
    if arguments.step == 'analyze':
        analyze(arguments.scores)
    else:
        {'freeze': freeze, 'geometry': geometry, 'score': score}[arguments.step]()
