"""Frozen follow-up on independent source scenes and RealSR injection transfer."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dev_pipeline import OLD, dump, local_sr, pixel_sha, read, sha, shift
from experiment import DIRECTIONS, load, variants
from freeze import LOCK as FORMAL_LOCK, verify as verify_formal
from gate import ZERO, actions
from registration import candidate, direct_candidate
from robust_registration import outside_replacement, tiled_candidates
from score import BATCH_SIZE, DEVICE, luminance_crop


BASE = HERE / 'holdout'
LOCK = BASE / 'lock.json'
GATES = ROOT / 'calibration/gates.json'
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20261125
METHODS = ('raw', 'direct', 'simple_lr', 'single', 'dual', 'tile_mean', 'tile_trimmed', 'oracle')


def atomic_dump(path, value):
    temporary = path.with_suffix('.tmp')
    dump(temporary, value)
    temporary.replace(path)


def tracked_files():
    return [Path(__file__), HERE/'holdout_data.py', HERE/'fetch_holdout.py',
            HERE/'robust_registration.py', HERE/'holdout_analysis.py', FORMAL_LOCK,
            GATES, BASE/'batch_inputs.json', ROOT/'external/batch_inputs.json']


def lock_experiment():
    if LOCK.exists():
        raise ValueError('Follow-up already frozen')
    if any(BASE.glob('*/geometry/*.json')):
        raise ValueError('Geometry already seen')
    primary = verify_formal()
    atomic_dump(LOCK, dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        status='Post-primary follow-up, frozen before measuring any holdout outcomes',
        hashes={str(p.relative_to(ROOT)): sha(p) for p in tracked_files()},
        method='8x8 LR tiles on existing 104x104 support, use tiles with anchor std >=2; '
               'if >=16 eligible, minimize mean correlation loss of best ceil(half) tiles per candidate',
        controls='Full-image normalized MSE candidate, mean all eligible tile losses, and original '
                 'single and dual gates with unchanged coefficients and calibration thresholds, plus '
                 'raw scoring, direct HR-to-SR alignment, and known inverse translation',
        prediction='Compared with ungated full-image MSE registration, best-half tile registration '
                   'reduces false corrections from a 256 px local replacement and increases correct '
                   'corrections when a true 2 px global shift co-occurs with that replacement.',
        primary_comparisons=[
            'strong/replacement_256: false correction rate, full-image minus best-half',
            'strong/shift2_replacement_256: correct correction rate, best-half minus full-image',
        ],
        success='Both primary paired 95% intervals must lie strictly above zero. Each point '
                'improvement must be at least 5 percentage points. Clean-global paired correct-rate '
                'difference best-half minus full-image has lower 95% bound >= -5 percentage points.',
        uncertainty=dict(unit='source scene', seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_N,
                         percentiles=[2.5, 97.5]),
        secondary='128 and 320 px replacements, normal degradation, pure and SR-local shifts, '
                  'mean-tile ablation, frozen gates at their existing thresholds, signed and '
                  'absolute SHAFE error against known-inverse reference, camera injection response',
        camera='Report within 1 HR pixel for action and candidate changes relative to the '
               'zero-injection pair. These releases are prealigned and natural offsets remain unknown.',
        shafe=dict(batch_size=BATCH_SIZE, weight_sha256=primary['weights']['shafe']['sha256'])))
    print(f'Follow-up freeze SHA-256 {sha(LOCK)}', flush=True)


def verify_lock():
    verify_formal()
    locked = load(LOCK)
    for name, expected in locked['hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Follow-up frozen file changed: {name}')
    return locked


def records_for(arm):
    path = BASE/'batch_inputs.json' if arm == 'synthetic' else ROOT/'external/batch_inputs.json'
    return load(path)['records']


def variants_for(record, hr, sr, full):
    if record['degradation'] == 'camera':
        for name, h, s, vh, vs in variants(record, hr, sr):
            yield name, h, s, vh, vs, None
        return
    direction = np.array(record['direction'])
    hdir = direction * record['magnitude']
    sdir = np.array(DIRECTIONS[(DIRECTIONS.index(tuple(direction)) + 2) % len(DIRECTIONS)])
    sshift = sdir * record['magnitude']
    zero = np.zeros(2)
    h0, s0 = shift(hr, zero), shift(sr, zero)
    h1, s1 = shift(hr, hdir), shift(sr, sshift)
    yield 'zero', h0, s0, zero, zero, None
    yield 'hr_global', h1, s0, hdir, zero, None
    yield 'sr_global', h0, s1, zero, sshift, None
    yield 'both_global', h1, s1, hdir, sshift, None
    displaced, _ = local_sr(s0, 192, 192, sdir*4)
    displaced = shift(displaced, zero)
    yield 'sr_local', h0, displaced, zero, zero, None
    yield 'hr2_sr_local', shift(hr, direction*2), displaced, direction*2, zero, None
    for patch in record['patches']:
        size = patch['size']
        x, y = patch['target_xy']
        sx, sy = patch['source_xy']
        replaced = h0.copy()
        replaced[y:y+size, x:x+size] = full[sy:sy+size, sx:sx+size]
        yield f'replacement_{size}', shift(replaced, zero), s0, zero, zero, patch
        yield f'shift2_replacement_{size}', shift(replaced, direction*2), s0, direction*2, zero, patch


def images_for(record):
    for field in ('hr', 'lr', 'source'):
        if sha(record[field]) != record[field+'_sha256']:
            raise ValueError(f'Input drift: {record[field]}')
    sr_path = Path(record['sr'])
    sr_provenance = load(sr_path.with_suffix('.json'))
    if sr_provenance['sr_sha256'] != sha(sr_path) or sr_provenance['lr_sha256'] != record['lr_sha256']:
        raise ValueError(f'SR provenance error: {sr_path}')
    lr, hr, sr = [read(record[field]) for field in ('lr', 'hr', 'sr')]
    if lr.shape != (128, 128, 3) or hr.shape != (512, 512, 3) or sr.shape != hr.shape:
        raise ValueError(f'Unexpected image shape for {record["scene"]}')
    if pixel_sha(lr) != record['lr_pixel_sha256']:
        raise ValueError(f'LR pixel drift for {record["scene"]}')
    full = None if record['degradation'] == 'camera' else read(record['source'])
    return lr, hr, sr, full


def measure_scene(record, folder, lock_hash, gates):
    cv2.setNumThreads(1)
    filename = f'{record["scene"]}__{record["degradation"]}.json'
    path = folder/'geometry'/filename
    lr, hr, sr, full = images_for(record)
    stamp = dict(lock_sha256=lock_hash, sr_sha256=sha(record['sr']))
    if path.exists():
        data = load(path)
        if data['stamp'] == stamp:
            return data['rows']
    global_fits, tile_fits, rows = {}, {}, []

    def fit(img, digest):
        if digest not in global_fits:
            global_fits[digest] = candidate(img, lr)
        return global_fits[digest]

    for condition, h, s, vh, vs, patch in variants_for(record, hr, sr, full):
        hhash, shash = pixel_sha(h), pixel_sha(s)
        hr_fit, sr_fit = fit(h, hhash), fit(s, shash)
        if hhash not in tile_fits:
            tile_fits[hhash] = tiled_candidates(h, lr)
        row = dict(scene=record['scene'], group=record['group'], dataset=record['dataset'],
                   degradation=record['degradation'], condition=condition, magnitude=record['magnitude'],
                   hr_added=vh.tolist(), sr_added=vs.tolist(), patch=patch,
                   lr_pixel_sha256=record['lr_pixel_sha256'], hr_pixel_sha256=hhash, sr_pixel_sha256=shash,
                   hr_fit=hr_fit, sr_fit=sr_fit, direct_vector=direct_candidate(h, s),
                   **tile_fits[hhash])
        fields, confidence, _, _, _ = actions(row, gates)
        for method in ('tile_mean', 'tile_trimmed'):
            fields[method] = row[method]['vector'] if row[method]['executable'] else ZERO
        row.update(actions={m: fields[m] for m in METHODS}, confidence=confidence)
        if patch is not None:
            row['mask_diagnostic'] = outside_replacement(h, lr, *patch['target_xy'], patch['size'])
        rows.append(row)
    cases = {row['condition']: row for row in rows}
    fixed_pairs = [('zero', 'sr_global'), ('hr_global', 'both_global')]
    if full is not None:
        fixed_pairs.append(('zero', 'sr_local'))
    for clean, changed in fixed_pairs:
        for method in ('simple_lr', 'single', 'tile_mean', 'tile_trimmed'):
            assert cases[clean]['actions'][method] == cases[changed]['actions'][method]
    atomic_dump(path, dict(stamp=stamp, rows=rows))
    return rows


def geometry(arm, workers):
    verify_lock()
    folder = BASE/arm
    gates = load(GATES)
    records = records_for(arm)
    worker = partial(measure_scene, folder=folder, lock_hash=sha(LOCK), gates=gates)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(worker, records), 1):
            results.extend(rows)
            if i % 8 == 0 or i == len(records):
                print(f'{arm} geometry {i}/{len(records)} scene-degradations', flush=True)
    unique = {(r['scene'], r['degradation'], r['condition']) for r in results}
    assert len(unique) == len(results) == len(records) * (12 if arm == 'synthetic' else 4)
    atomic_dump(folder/'geometry.json', dict(lock_sha256=sha(LOCK), rows=results))


def score_scene(record, rows, evaluate):
    _, hr, sr, full = images_for(record)
    pairs, results = {}, []
    for condition, h, s, vh, vs, _ in variants_for(record, hr, sr, full):
        measured = rows[condition]
        if (pixel_sha(h), pixel_sha(s), vh.tolist(), vs.tolist()) != (
                measured['hr_pixel_sha256'], measured['sr_pixel_sha256'],
                measured['hr_added'], measured['sr_added']):
            raise ValueError(f'Scoring variant changed: {record["scene"]}, {condition}')
        fixed_sr = luminance_crop(s)
        sr_hash = pixel_sha(fixed_sr)
        ids = {}
        for method, vector in measured['actions'].items():
            reference = luminance_crop(shift(h, vector) if any(vector) else h)
            key = sr_hash + ':' + pixel_sha(reference)
            pairs.setdefault(key, (fixed_sr, reference))
            ids[method] = key
        results.append(dict(scene=measured['scene'], degradation=measured['degradation'],
                            condition=condition, pair_ids=ids))
    values = dict(zip(pairs, evaluate(list(pairs.values())), strict=True))
    for row in results:
        row['scores'] = {m: values[key] for m, key in row['pair_ids'].items()}
    return results, len(pairs)


def scoring(arm):
    locked = verify_lock()
    sys.path.insert(0, str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    weight_hash = sha(DEFAULT_WEIGHT)
    if weight_hash != locked['shafe']['weight_sha256']:
        raise ValueError('Official SHAFE weight drift')
    folder = BASE/arm
    path = folder/'geometry.json'
    measured = load(path)
    if measured['lock_sha256'] != sha(LOCK):
        raise ValueError('Geometry protocol mismatch')
    by_pair = {}
    for row in measured['rows']:
        by_pair.setdefault((row['scene'], row['degradation']), {})[row['condition']] = row
    stamp = dict(lock_sha256=sha(LOCK), geometry_sha256=sha(path), shafe_weight_sha256=weight_hash,
                 batch_size=BATCH_SIZE, device=DEVICE)
    metric = None

    def evaluate(pairs):
        nonlocal metric
        if metric is None:
            metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
            metric.device = metric.eth_feature.device = DEVICE
            metric.eth_feature.model.to(DEVICE)
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    all_rows, unique = [], 0
    records = records_for(arm)
    for i, record in enumerate(records, 1):
        key = record['scene'], record['degradation']
        cache = folder/'scoring'/f'{key[0]}__{key[1]}.json'
        result = load(cache) if cache.exists() else {}
        if result.get('stamp') != stamp:
            rows, count = score_scene(record, by_pair[key], evaluate)
            result = dict(stamp=stamp, rows=rows, unique_pairs=count)
            atomic_dump(cache, result)
        all_rows.extend(result['rows'])
        unique += result['unique_pairs']
        if i % 8 == 0 or i == len(records):
            print(f'{arm} SHAFE {i}/{len(records)} scene-degradations ({unique} distinct pairs)', flush=True)
    assert len(all_rows) == len(measured['rows'])
    atomic_dump(folder/'scores.json', dict(stamp=stamp, rows=all_rows, unique_pairs=unique))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('step', choices=('freeze', 'geometry', 'score'))
    cli.add_argument('--arm', choices=('synthetic', 'camera'), default='synthetic')
    cli.add_argument('--workers', type=int, default=4)
    args = cli.parse_args()
    if args.step == 'freeze':
        lock_experiment()
    elif args.step == 'geometry':
        geometry(args.arm, args.workers)
    else:
        scoring(args.arm)
