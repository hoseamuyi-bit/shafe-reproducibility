"""Supplementary robust-registration controls on the existing frozen cases."""

import argparse
from datetime import datetime, timezone

import cv2
import numpy as np

from holdout_analysis import linf, paired_summary
from holdout_run import BASE, LOCK, atomic_dump, images_for, load, records_for, sha, variants_for, verify_lock
from dev_pipeline import luminance, pixel_sha, project


FOLDER = BASE / 'robust_controls'
PLAN = FOLDER / 'lock.json'
METHODS = ('simple_lr', 'tile_mean', 'tile_trimmed', 'median_block', 'median_lk')
COMPARISONS = [('tile_trimmed', method) for method in METHODS if method != 'tile_trimmed']
CONDITIONS = ('zero', 'hr_global', 'replacement_256', 'shift2_replacement_256')


def freeze():
    verify_lock()
    if PLAN.exists():
        raise ValueError('Supplementary comparison already specified')
    FOLDER.mkdir(parents=True, exist_ok=True)
    atomic_dump(PLAN, dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        follow_up_lock_sha256=sha(LOCK), code_sha256=sha(__file__),
        status='Supplementary controls selected after initial RealSR results; do not amend the frozen follow-up.',
        dataset='Synthetic holdout, strong degradation, all 96 source scenes', conditions=CONDITIONS,
        median_block='Componentwise median of the original 4x4 local optima over the archived search grid.',
        median_lk='Componentwise median of LR-to-projected-HR pyramidal Lucas–Kanade feature displacements '
                  'with a forward-backward check, converted to an HR correction.',
        decision='Round each component to 0.25 HR px. Apply only if the maximum component exceeds 0.25 HR px. '
                 'LK needs at least eight surviving tracks. Report unavailability separately.',
        endpoints='Same per-condition action endpoints and paired scene bootstrap as the frozen follow-up. '
                  'Report benchmark contrasts descriptively.'))


def to_action(vector):
    rounded = np.rint(4 * np.asarray(vector)) / 4
    return rounded.tolist() if linf(rounded) > .25 else [0., 0.]


def lucas_kanade(anchor, reference):
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[12:116, 12:116] = 255
    points = cv2.goodFeaturesToTrack(anchor, 96, .01, 6, mask=mask, blockSize=5)
    if points is None or len(points) < 8:
        return [0., 0.], 0
    params = dict(winSize=(9, 9), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, matched, _ = cv2.calcOpticalFlowPyrLK(anchor, reference, points, None, **params)
    reverse, returned, _ = cv2.calcOpticalFlowPyrLK(reference, anchor, forward, None, **params)
    displacement = (forward - points)[:, 0, :]
    fb_error = np.linalg.norm((reverse - points)[:, 0, :], axis=1)
    good = ((matched[:, 0] > 0) & (returned[:, 0] > 0) & (fb_error <= .5)
            & (np.max(np.abs(displacement), axis=1) <= 2))
    n = int(good.sum())
    if n < 8:
        return [0., 0.], n
    return to_action(-4 * np.median(displacement[good], axis=0)), n


def gray8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def run():
    verify_lock()
    plan = load(PLAN)
    if (plan['follow_up_lock_sha256'], plan['code_sha256']) != (sha(LOCK), sha(__file__)):
        raise ValueError('Supplementary comparison changed')
    geometry_path = BASE / 'synthetic' / 'geometry.json'
    geometry = load(geometry_path)
    assert geometry['lock_sha256'] == sha(LOCK)
    indexed = {(r['scene'], r['condition']): r for r in geometry['rows']
               if r['degradation'] == 'strong' and r['condition'] in CONDITIONS}
    rows = []
    cv2.setNumThreads(1)
    for record in records_for('synthetic'):
        if record['degradation'] != 'strong':
            continue
        lr, hr, sr, full = images_for(record)
        anchor = gray8(luminance(lr))
        for condition, h, _, added, _, _ in variants_for(record, hr, sr, full):
            if condition not in CONDITIONS:
                continue
            measured = indexed[record['scene'], condition]
            assert pixel_sha(h) == measured['hr_pixel_sha256'] and added.tolist() == measured['hr_added']
            archived = measured['actions']
            original_fit = measured['hr_fit']
            blocks = original_fit.get('tile_vectors', [])
            median_block = to_action(np.median(blocks, axis=0)) if blocks else [0., 0.]
            reference = gray8(project(luminance(h)))
            median_lk, n_tracks = lucas_kanade(anchor, reference)
            rows.append({**{key: measured[key] for key in ('scene', 'group', 'dataset', 'condition', 'hr_added')},
                         'actions': {**{m: archived[m] for m in METHODS[:3]},
                                     'median_block': median_block, 'median_lk': median_lk},
                         'n_tracks': n_tracks, 'median_block_valid': bool(blocks)})
    assert len(rows) == len(indexed) == 96 * len(CONDITIONS)
    report = {}
    for condition in CONDITIONS:
        cases = [r for r in rows if r['condition'] == condition]
        keep = condition in ('zero', 'replacement_256')

        def event(row, method):
            action = row['actions'][method]
            return bool(any(action)) and (keep or linf(np.asarray(action) + row['hr_added']) <= .25)

        pairs = [(b, a) for a, b in COMPARISONS] if keep else COMPARISONS
        report[condition] = paired_summary(cases, METHODS, event, pairs)
        report[condition]['lk_unavailable'] = sum(r['n_tracks'] < 8 for r in cases)
        report[condition]['block_unavailable'] = sum(not r['median_block_valid'] for r in cases)
        report[condition]['event_counts'] = {m: sum(event(r, m) for r in cases) for m in METHODS}
        print(condition, report[condition]['event_counts'], flush=True)
    atomic_dump(FOLDER / 'results.json', dict(lock_sha256=sha(PLAN), geometry_sha256=sha(geometry_path),
                                              strata=report, rows=rows))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('step', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[cli.parse_args().step]()
