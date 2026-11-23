"""Supplementary translation benchmarks on the frozen COCO and RealSR inputs."""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import sys

import cv2
import numpy as np

import camera_local
import coco_confirm
from holdout_analysis import linf, paired_summary
from holdout_run import OLD, ROOT, atomic_dump, images_for, load, sha, variants_for
from dev_pipeline import luminance, pixel_sha, project, shift
from robust_controls import gray8, lucas_kanade, to_action
from score import BATCH_SIZE, DEVICE, luminance_crop
from verify_holdout import event


HERE = Path(__file__).resolve().parent
DEST = HERE / 'registration_benchmarks'
PLAN = DEST / 'lock.json'
COCO_GEOMETRY = coco_confirm.FOLDER / 'geometry.json'
COCO_SCORES = coco_confirm.FOLDER / 'scores.json'
CAMERA_SCORES = coco_confirm.FOLDER / 'sensitivity/realsr_lk_analysis.json'
NEW = ('ecc', 'phase_correlation', 'lk_consensus')
METHODS = ('raw', 'simple_lr', 'tile_trimmed', 'median_lk', *NEW)
CAMERA_CONDITIONS = ('zero', 'shift2', 'replacement_256', 'shift2_replacement_256')
ROI = np.s_[12:116, 12:116]
RADIUS_LR = 2
INLIER_RADIUS_LR = .25
MIN_TRACKS = 8
WINDOW = cv2.createHanningWindow((104, 104), cv2.CV_32F)
PHASE_FINE = np.arange(-16, 17) / 16


def estimate(displacement, quality=None, available=True):
    vector = np.asarray(displacement, dtype=float)
    valid = bool(available and np.isfinite(vector).all() and linf(vector) <= RADIUS_LR)
    return dict(valid=valid, action=to_action(-4 * vector) if valid else [0., 0.],
                displacement_lr=vector.tolist() if np.isfinite(vector).all() else None,
                quality=quality if quality is None or np.isfinite(quality) else None)


def phase_correlation(fixed, moving):
    """Refine the integer phase-correlation peak with a local upsampled DFT."""
    fixed_fft = np.fft.fft2((fixed - fixed.mean()) * WINDOW)
    moving_fft = np.fft.fft2((moving - moving.mean()) * WINDOW)
    cross = moving_fft * fixed_fft.conj()
    cross /= np.maximum(np.abs(cross), 100 * np.finfo(np.float32).eps)
    correlation = np.fft.ifft2(cross).real
    height, width = correlation.shape
    y, x = np.unravel_index(correlation.argmax(), correlation.shape)
    ys = (y if y <= height // 2 else y - height) + PHASE_FINE
    xs = (x if x <= width // 2 else x - width) + PHASE_FINE
    ey = np.exp(2j * np.pi * np.outer(ys, np.fft.fftfreq(height)))
    ex = np.exp(2j * np.pi * np.outer(xs, np.fft.fftfreq(width)))
    fine = (ey @ cross @ ex.T).real / (height * width)
    yi, xi = np.unravel_index(fine.argmax(), fine.shape)
    return estimate([xs[xi], ys[yi]], float(fine[yi, xi]))


def dense_estimates(anchor, reference):
    fixed = np.ascontiguousarray(anchor[ROI])
    moving = np.ascontiguousarray(reference[ROI])
    if min(float(fixed.std()), float(moving.std())) < 1e-6:
        return {name: estimate([0, 0], available=False) for name in NEW[:2]}

    # ECC maps the 104px template coordinates into the full projected image.
    warp = np.float32([[1, 0, 12], [0, 1, 12]])
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    try:
        correlation, warp = cv2.findTransformECC(fixed, reference, warp, cv2.MOTION_TRANSLATION,
                                                 criteria, None, 5)
        ecc = estimate(warp[:, 2] - 12, float(correlation))
    except cv2.error as exc:
        if exc.code != cv2.Error.StsNoConv:
            raise
        ecc = estimate([0, 0], available=False)

    return dict(ecc=ecc, phase_correlation=phase_correlation(fixed, moving))


def lk_track_vectors(anchor, reference):
    """Use the frozen LK settings and verify the reproduced median on each pair."""
    mask = np.zeros(anchor.shape, dtype=np.uint8)
    mask[ROI] = 255
    points = cv2.goodFeaturesToTrack(anchor, 96, .01, 6, mask=mask, blockSize=5)
    if points is None or len(points) < MIN_TRACKS:
        return np.empty((0, 2), dtype=np.float32)
    params = dict(winSize=(9, 9), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, matched, _ = cv2.calcOpticalFlowPyrLK(anchor, reference, points, None, **params)
    reverse, returned, _ = cv2.calcOpticalFlowPyrLK(reference, anchor, forward, None, **params)
    displacement = (forward - points)[:, 0, :]
    fb_error = np.linalg.norm((reverse - points)[:, 0, :], axis=1)
    good = ((matched[:, 0] > 0) & (returned[:, 0] > 0) & (fb_error <= .5)
            & (np.max(np.abs(displacement), axis=1) <= RADIUS_LR))
    return displacement[good]


def track_consensus(tracks):
    if len(tracks) == 0:
        return dict(estimate([0, 0], 0, available=False), track_count=0)
    distances = np.linalg.norm(tracks[:, None, :] - tracks[None, :, :], axis=2)
    masks = distances <= INLIER_RADIUS_LR
    counts = masks.sum(axis=1)
    # Exhaust all one-track translation hypotheses. Break ties by inlier squared error.
    best = int(np.lexsort(((np.square(distances) * masks).sum(axis=1), -counts))[0])
    count = int(counts[best])
    return dict(estimate(np.median(tracks[masks[best]], axis=0), count, available=count >= MIN_TRACKS),
                track_count=len(tracks))


def freeze():
    coco_confirm.verify()
    if PLAN.exists() or any(DEST.glob('*_results.json')):
        raise ValueError('Supplementary benchmark already exists')
    tracked = (Path(__file__), coco_confirm.PLAN, coco_confirm.INPUTS, COCO_GEOMETRY, COCO_SCORES,
               CAMERA_SCORES, camera_local.PLAN, camera_local.INPUTS, HERE / 'camera_local.py',
               HERE / 'coco_confirm.py', HERE / 'robust_controls.py', HERE / 'holdout_analysis.py',
               HERE / 'holdout_run.py', HERE / 'verify_holdout.py', ROOT / 'dev_pipeline.py', ROOT / 'score.py')
    atomic_dump(PLAN, dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        scope='Supplementary experiment on the previously analyzed COCO and RealSR cases; '
              'parameters fixed before running the new methods on either cohort.',
        versions=dict(opencv=cv2.__version__, numpy=np.__version__),
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in tracked},
        ecc='Translation-only OpenCV ECC, Gaussian size 5, 100 iterations, epsilon 1e-6. Match within '
            'the 104px anchor crop against the full 128px projected reference starting from zero motion.',
        phase_correlation='Phase-only correlation on the two mean-subtracted center 104 px crops '
                          'with an OpenCV Hann window. Integer peak followed by local upsampled '
                          'DFT refinement (±1 LR px centered at that peak, step 1/16 LR px).',
        lk_consensus='Identical forward-backward accepted tracks to the frozen median-LK baseline. '
                     'Exhaust the observed translations, maximize the number of inliers within '
                     '0.25 LR px Euclidean, break ties by inlier SSE then track order, and refit by '
                     'inlier coordinate-wise median. Require at least eight inliers.',
        decision='Same projected luminance. Convert by -4; reject nonfinite, unavailable, or out-of-range '
                 'estimates (>2 LR px on either axis), round to 0.25 HR px, act only above 0.25 HR px.',
        endpoints='COCO correct correction and false movement on all six original conditions. RealSR '
                  'incremental action residual above 1 HR px compared to each method’s zero-injection '
                  'action; report unavailable estimates separately. Both include SHAFE oracle MAE '
                  'and descriptive paired 95% scene-bootstrap intervals. No new success threshold.'))
    print(f'Benchmark lock {sha(PLAN)}', flush=True)


def verify():
    plan = load(PLAN)
    for path, expected in plan['hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen benchmark input changed: {path}')
    if plan['versions'] != dict(opencv=cv2.__version__, numpy=np.__version__):
        raise ValueError('Benchmark library versions changed')


def synth_variants(record, hr, sr, full, _patch):
    for condition, moving, restored, added, _, _ in variants_for(record, hr, sr, full):
        if condition in coco_confirm.CONDITIONS:
            yield condition, moving, restored, added.tolist()


def camera_variants(record, hr, sr, _full, patch):
    restored = shift(sr, [0, 0])
    for condition, moving, added in camera_local.variants(record, patch, hr):
        yield condition, moving, restored, list(added)


def score_scene(record, old, geometry, generate, patch, metric, paper_scores):
    lr, hr, sr, full = images_for(record)
    anchor = luminance(lr)
    planned = set(old)
    fits, rows, queued = {}, [], {}
    for condition, moving, restored, added in generate(record, hr, sr, full, patch):
        archived = old[condition]
        measured = geometry[condition] if geometry is not None else archived
        assert measured['hr_added'] == archived['hr_added'] == added
        digest = pixel_sha(moving)
        assert measured['hr_pixel_sha256'] == digest
        fixed = luminance_crop(restored)
        sr_hash = pixel_sha(fixed)
        if 'pair_ids' in archived:
            oracle = luminance_crop(shift(moving, [-x for x in added]) if any(added) else moving)
            assert archived['pair_ids']['oracle'] == sr_hash + ':' + pixel_sha(oracle)
        else:
            assert archived['sr_pixel_sha256'] == pixel_sha(restored)
        if digest not in fits:
            reference = project(luminance(moving))
            tracks = lk_track_vectors(gray8(anchor), gray8(reference))
            median = to_action(-4 * np.median(tracks, axis=0)) if len(tracks) >= MIN_TRACKS else [0., 0.]
            assert (median, len(tracks)) == lucas_kanade(gray8(anchor), gray8(reference))
            assert (median, len(tracks)) == (archived['actions']['median_lk'], measured['tracks'])
            fits[digest] = dict(**dense_estimates(anchor, reference), lk_consensus=track_consensus(tracks))
        estimates = fits[digest]
        actions = {m: archived['actions'][m] for m in METHODS if m not in NEW}
        actions.update({m: estimates[m]['action'] for m in NEW})
        scores = {m: archived['scores'][m] for m in (*METHODS, 'oracle') if m not in NEW}
        reused = {}
        for method, action in ((m, actions[m]) for m in NEW):
            match = next((m for m, v in archived['actions'].items() if m in archived['scores'] and v == action), None)
            if action == [-x for x in added]:
                match = 'oracle'
            reused[method] = match
            if match is not None:
                scores[method] = archived['scores'][match]
                continue
            reference = luminance_crop(shift(moving, action) if any(action) else moving)
            pair = sr_hash + ':' + pixel_sha(reference)
            if pair not in queued:
                queued[pair] = (fixed, reference, [])
            queued[pair][2].append((scores, method))
        rows.append(dict(scene=record['scene'], dataset=record['dataset'], group=record['group'], condition=condition,
                         magnitude=record['magnitude'], hr_added=added, hr_pixel_sha256=digest, sr_crop_sha256=sr_hash,
                         actions=actions, benchmark_fits=estimates, scores=scores, score_reused=reused))
        planned.remove(condition)
    assert not planned
    images = [(fixed, reference) for fixed, reference, _ in queued.values()]
    if images:
        for (_, _, updates), value in zip(queued.values(), paper_scores(metric, images, DEVICE, BATCH_SIZE), strict=True):
            for scores, method in updates:
                scores[method] = value
    return rows, len(queued)


def indexed(path):
    result = {}
    for row in load(path)['rows']:
        by_condition = result.setdefault(row['scene'], {})
        assert row['condition'] not in by_condition
        by_condition[row['condition']] = row
    return result


def summarize(rows, camera):
    levels = CAMERA_CONDITIONS if camera else coco_confirm.CONDITIONS
    baseline = {r['scene']: r['actions'] for r in rows if r['condition'] == 'zero'}
    comparisons = tuple(combinations(METHODS[1:], 2))
    summary = {}
    for condition in (*levels, 'all'):
        magnitudes = (None, .5, 2, 6) if condition == 'hr_global' else (None,)
        for magnitude in magnitudes:
            cases = [r for r in rows if condition in (r['condition'], 'all') and
                     (magnitude is None or r['magnitude'] == magnitude)]
            name = condition if magnitude is None else f'{condition}/{magnitude:g}px'
            result = dict(shafe_mae=paired_summary(cases, METHODS,
                          lambda r, m: abs(r['scores'][m] - r['scores']['oracle']), comparisons),
                          unavailable={m: sum(not r['benchmark_fits'][m]['valid'] for r in cases) for m in NEW})
            if condition != 'all' and not (camera and condition == 'zero'):
                if camera:
                    def geometry(r, m):
                        return linf(np.asarray(r['actions'][m]) - baseline[r['scene']][m] + r['hr_added']) > 1
                    endpoint = 'incremental_error_gt_1px'
                else:
                    endpoint = 'correct' if any(cases[0]['hr_added']) else 'false'

                    def geometry(r, m):
                        return event(r, m, endpoint)

                result['geometry'] = paired_summary(cases, METHODS, geometry, comparisons)
                result['geometry']['endpoint'] = endpoint
                result['geometry']['counts'] = {m: sum(geometry(r, m) for r in cases) for m in METHODS}
            summary[name] = result
    return summary


def run(cohort):
    verify()
    cv2.setNumThreads(1)
    camera = cohort == 'realsr'
    inputs = camera_local.INPUTS if camera else coco_confirm.INPUTS
    originals = indexed(CAMERA_SCORES if camera else COCO_SCORES)
    geometry = None if camera else indexed(COCO_GEOMETRY)
    patches = load(camera_local.PLAN)['patches'] if camera else {}
    generate = camera_variants if camera else synth_variants
    records = load(inputs)['records']
    assert len(records) == len(originals) == (32 if camera else 144)

    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT) == load(coco_confirm.PLAN)['shafe']['weight_sha256']
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    rows, new_pairs = [], 0
    lock_hash = sha(PLAN)
    for i, record in enumerate(records, 1):
        scene = record['scene']
        sr_hash = sha(record['sr'])
        cache_path = DEST / cohort / f'{scene}.json'
        batch = load(cache_path) if cache_path.exists() else {}
        if (batch.get('lock_sha256'), batch.get('sr_sha256')) != (lock_hash, sr_hash):
            result, pairs = score_scene(record, originals[scene], None if camera else geometry[scene], generate,
                                        patches.get(scene), metric, paper_scores)
            batch = dict(rows=result, new_pairs=pairs, lock_sha256=lock_hash, sr_sha256=sr_hash)
            atomic_dump(cache_path, batch)
        rows.extend(batch['rows'])
        new_pairs += batch['new_pairs']
        if i % 12 == 0 or i == len(records):
            print(f'{cohort}: {i}/{len(records)} scenes, {new_pairs} newly scored pairs', flush=True)
    expected = 128 if camera else 864
    assert len({(r['scene'], r['condition']) for r in rows}) == len(rows) == expected
    summary = summarize(rows, camera)
    atomic_dump(DEST / f'{cohort}_results.json', dict(lock_sha256=lock_hash, rows=rows,
                                                    strata=summary, new_score_pairs=new_pairs))
    for name, stats in summary.items():
        if 'geometry' in stats:
            print(name, stats['geometry']['counts'], flush=True)


def self_test():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(811)
    anchor = cv2.GaussianBlur(rng.normal(128, 26, (128, 128)).astype(np.float32), (5, 5), .8)
    for vector in ([0, 0], [1, -1], [-.5, .5], [.5, -1.5]):
        moving = shift(anchor, vector)
        expected = -4 * np.asarray(vector)
        estimators = dict(**dense_estimates(anchor, moving),
                          lk_consensus=track_consensus(lk_track_vectors(gray8(anchor), gray8(moving))))
        for name, data in estimators.items():
            assert data['valid'] and linf(np.asarray(data['action']) - expected) <= .25, (name, vector, data)
    blank = np.zeros_like(anchor)
    assert not any(fit['valid'] for fit in dense_estimates(blank, blank).values())
    assert not track_consensus(lk_track_vectors(gray8(blank), gray8(blank)))['valid']
    assert not estimate([3, 0])['valid']
    mixed = np.float32([[.5, -.5]] * 8 + [[0, 1]] * 7)
    assert track_consensus(mixed)['action'] == [-2, 2]
    print('Passed coordinate, blank, range, and minority-outlier checks', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('self-test', 'freeze', 'verify', 'coco', 'realsr'))
    args = parser.parse_args()
    if args.step in ('coco', 'realsr'):
        run(args.step)
    else:
        {'self-test': self_test, 'freeze': freeze, 'verify': verify}[args.step]()
