"""Check soft replacement edges and transfer the fixed LK control to RealSR."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import sys

import cv2
import numpy as np

from camera_local import FOLDER as CAMERA_FOLDER, INPUTS as CAMERA_INPUTS, variants as camera_variants
from coco_confirm import FOLDER, INPUTS, MATCHERS, PLAN as COCO_LOCK, reference_inputs, verify
from holdout_analysis import linf, paired_summary
from holdout_run import OLD, ROOT, atomic_dump, images_for, load, sha
from dev_pipeline import luminance, pixel_sha, project, shift
from registration import candidate
from robust_controls import gray8, lucas_kanade
from robust_registration import tiled_candidates
from score import BATCH_SIZE, DEVICE, luminance_crop
from verify_holdout import event


DEST = FOLDER/'sensitivity'
PLAN = DEST/'lock.json'
TAPER = 24
LOCAL = ('replacement_256', 'shift2_replacement_256')
CAMERA_METHODS = (*MATCHERS, 'raw')
COMPARISONS = (('simple_lr', 'tile_trimmed'), ('simple_lr', 'median_lk'), ('tile_trimmed', 'median_lk'))


def sources():
    return (COCO_LOCK, CAMERA_INPUTS, CAMERA_FOLDER/'lock.json', CAMERA_FOLDER/'geometry.json',
            CAMERA_FOLDER/'scores.json', CAMERA_FOLDER.parent.parent/'camera_local.py')


def freeze():
    verify()
    if PLAN.exists() or any(DEST.glob('*_analysis.json')):
        raise ValueError('Sensitivity protocol or results already exist')
    atomic_dump(PLAN, dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(__file__),
        hashes={str(p.relative_to(ROOT)): sha(p) for p in sources()},
        coco='For the same COCO images, patch positions and 2px shifts, taper the outer 24px of the '
             'replacement with a separable raised cosine from 0 at the outer edge to 1 in the interior. '
             'Repeat the same paired geometric endpoints and quantify the absolute rates and effect '
             'of tapering. This is a sensitivity analysis because tapering changes both the seam and '
             'the amount of changed content. Describe regions with and without donor overlap separately.',
        realSR='Apply the fixed Lucas–Kanade baseline to the previously archived RealSR local variants. '
               'Compare error above 1 HR px in the change of action relative to the unperturbed case, '
               'at 0px and 2px added translation plus local replacement, and the clean 2px translation. '
               'These data previously informed the tile-method comparison. Natural offsets are unknown.',
        scoring='RealSR absolute SHAFE error relative to the known inverse injection. Reuse frozen '
                'comparison scores and compute missing Lucas–Kanade scores with identical inputs.'))
    print(f'Sensitivity plan frozen: {sha(PLAN)}', flush=True)


def check():
    verify()
    plan = load(PLAN)
    if plan['code_sha256'] != sha(__file__):
        raise ValueError('Sensitivity source changed')
    for name, expected in plan['hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Sensitivity input changed: {name}')


def tapered_scene(record):
    cv2.setNumThreads(1)
    lr, hr, full = reference_inputs(record)
    patch, = record['patches']
    x, y = patch['target_xy']
    sx, sy = patch['source_xy']
    size = patch['size']
    assert size == 256
    edge = np.minimum(np.arange(size), np.arange(size)[::-1])
    taper = (0.5 - 0.5 * np.cos(np.pi*np.minimum(edge/TAPER, 1))).astype(np.float32)
    weight = np.outer(taper, taper)[:, :, None]
    replacement = hr.astype(np.float32)
    target = replacement[y:y+size, x:x+size]
    target[:] = (1-weight)*target + weight*full[sy:sy+size, sx:sx+size]
    anchor = gray8(luminance(lr))
    rows = []
    for condition, translation in zip(LOCAL, ([0, 0], [2*x for x in record['direction']]), strict=True):
        moving = shift(replacement, translation)
        estimates = dict(simple_lr=candidate(moving, lr), **tiled_candidates(moving, lr))
        actions = {m: v['vector'] if v['executable'] else [0, 0] for m, v in estimates.items()}
        actions['median_lk'], tracks = lucas_kanade(anchor, gray8(project(luminance(moving))))
        rows.append(dict(scene=record['scene'], group=record['group'], dataset=record['dataset'],
                         condition=condition, hr_added=translation, overlap_pixels=patch['overlap_pixels'],
                         actions=actions, tracks=tracks))
    return rows


def taper_analysis():
    check()
    geometry_path = FOLDER/'geometry.json'
    geometry = load(geometry_path)
    assert geometry['lock_sha256'] == sha(COCO_LOCK)
    hard = {(r['scene'], r['condition']):r for r in geometry['rows'] if r['condition'] in LOCAL}
    records = load(INPUTS)['records']
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = [r for group in pool.map(tapered_scene, records) for r in group]
    assert {(r['scene'], r['condition']) for r in rows} == set(hard)
    reports = {}
    for condition in LOCAL:
        endpoint = 'false' if condition == LOCAL[0] else 'correct'
        for scope in ('all', 'disjoint', 'overlap'):
            cases = [r for r in rows if r['condition'] == condition and
                     (scope == 'all' or bool(r['overlap_pixels']) == (scope == 'overlap'))]
            if not cases:
                continue
            labels = tuple(f'{mode}/{m}' for mode in ('hard', 'tapered') for m in MATCHERS)

            def value(row, label):
                mode, method = label.split('/')
                case = hard[row['scene'], condition] if mode == 'hard' else row
                assert case['hr_added'] == row['hr_added']
                return event(case, method, endpoint)

            contrasts = [(f'tapered/{m}', f'hard/{m}') for m in MATCHERS]
            pairs = COMPARISONS if endpoint == 'false' else [(b, a) for a, b in COMPARISONS]
            contrasts.extend((f'{mode}/{a}', f'{mode}/{b}') for mode in ('hard', 'tapered') for a, b in pairs)
            reports[f'{condition}/{scope}'] = paired_summary(cases, labels, value, contrasts)
    atomic_dump(DEST/'coco_taper_analysis.json', dict(lock_sha256=sha(PLAN), geometry_sha256=sha(geometry_path),
                                                     strata=reports, rows=rows))


def realsr():
    check()
    sys.path.insert(0, str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT) == load(COCO_LOCK)['shafe']['weight_sha256']
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    geometry_path = CAMERA_FOLDER/'geometry.json'
    scores_path = CAMERA_FOLDER/'scores.json'
    scores = load(scores_path)
    assert scores['geometry_sha256'] == sha(geometry_path)
    existing = {(r['scene'], r['condition']):r for r in scores['rows']}
    records = load(CAMERA_INPUTS)['records']
    patches = load(CAMERA_FOLDER/'lock.json')['patches']
    rows = []
    cv2.setNumThreads(1)
    for i, record in enumerate(records, 1):
        lr, hr, sr, _ = images_for(record)
        anchor = gray8(luminance(lr))
        shifted_sr = shift(sr, [0, 0])
        fixed = luminance_crop(shifted_sr)
        scene = record['scene']
        baseline = None
        for condition, moving, translation in camera_variants(record, patches[scene], hr):
            row = existing[scene, condition]
            assert row['hr_pixel_sha256'] == pixel_sha(moving) and row['hr_added'] == translation
            assert row['sr_pixel_sha256'] == pixel_sha(shifted_sr)
            action, tracks = lucas_kanade(anchor, gray8(project(luminance(moving))))
            if condition == 'zero':
                baseline = dict(row['actions'], median_lk=action, raw=[0, 0])
            actions = dict(row['actions'], median_lk=action, raw=[0, 0])
            assert baseline is not None
            vector = dict(actions, oracle=[-x for x in translation])
            equal = next((m for m in row['scores'] if vector[m] == action), None)
            if equal is not None:
                lk_score = row['scores'][equal]
            else:
                reference = luminance_crop(shift(moving, action) if any(action) else moving)
                lk_score, = paper_scores(metric, [(fixed, reference)], DEVICE, BATCH_SIZE)
            remaining = {m: linf(np.asarray(v)-baseline[m]+translation) for m, v in actions.items()}
            rows.append(dict(row, actions=actions, tracks=tracks, incremental_error=remaining,
                             scores=dict(row['scores'], median_lk=lk_score)))
        if i % 8 == 0:
            print(f'RealSR fixed Lucas–Kanade control {i}/{len(records)}', flush=True)
    assert len(rows) == len(existing) == 32*4
    reports = {}
    for condition in ('shift2', *LOCAL):
        cases = [r for r in rows if r['condition'] == condition]
        reports[condition] = dict(
            error_gt_1px=paired_summary(cases, CAMERA_METHODS, lambda r, m: r['incremental_error'][m] > 1,
                                        COMPARISONS),
            shafe_mae=paired_summary(cases, CAMERA_METHODS,
                                    lambda r, m: abs(r['scores'][m]-r['scores']['oracle']), COMPARISONS),
            lk_fewer_than_eight_tracks=sum(r['tracks'] < 8 for r in cases))
    atomic_dump(DEST/'realsr_lk_analysis.json', dict(lock_sha256=sha(PLAN), prior_scores_sha256=sha(scores_path),
                                                    strata=reports, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('freeze', 'taper', 'realsr'))
    {'freeze': freeze, 'taper': taper_analysis, 'realsr': realsr}[parser.parse_args().step]()
