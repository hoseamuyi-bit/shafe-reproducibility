"""Check whether the COCO smooth-warp result transfers to RealSR camera pairs."""

import argparse
from datetime import datetime, timezone
from itertools import product
import random
import sys

import cv2

import local_deformation as warp
import registration_benchmarks as benchmarks
from holdout_analysis import linf, paired_summary
from holdout_run import OLD, ROOT, atomic_dump, load, sha
from score import BATCH_SIZE, DEVICE


DEST = warp.DEST.with_name('camera_deformation')
LOCK = DEST / 'lock.json'
INPUTS = benchmarks.camera_local.INPUTS
BASELINE = benchmarks.DEST / 'realsr_results.json'
METHODS = benchmarks.METHODS
COMPARISONS = tuple((m, 'simple_lr') for m in METHODS[2:]) + (('median_lk', 'lk_consensus'),)
SEED = 20261201


def freeze():
    benchmarks.verify()
    if LOCK.exists() or (DEST / 'results.json').exists():
        raise ValueError('Camera deformation protocol already exists')
    records = load(INPUTS)['records']
    assert len(records) == 32 and len({r['scene'] for r in records}) == 32
    assert all(sum(r['dataset'] == camera for r in records) == 16 for camera in ('Canon', 'Nikon'))
    rng = random.Random(SEED)
    parameters = {record['scene']: dict(center=[rng.choice((224, 288)), rng.choice((224, 288))],
                                        local_direction=list(rng.choice(warp.DIRECTIONS)))
                  for record in sorted(records, key=lambda r: r['scene'])}
    files = [DEST.with_suffix('.py'), INPUTS, BASELINE, warp.LOCK]
    files.extend(ROOT / path for path in load(warp.LOCK)['hashes'])
    atomic_dump(LOCK, dict(
        created_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, parameters=parameters,
        hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        scope='Descriptive sensitivity test using the 32 previously registered RealSR V3 x4 camera pairs. '
              'Camera pair selection and all registrations are inherited. The natural baseline offset is unknown.',
        design='Use the same diameter, amplitude, and translation grid as the COCO smooth warp. '
               'Use the camera records’ frozen HR translation direction. Keep LR and VOSR SR fixed.',
        geometry='Primary response is action minus the method’s clean zero-injection action plus the known '
                 'global translation, measured in HR pixels. Report rate of max-axis residual above 1 HR px '
                 'and mean max-axis residual. Also report failures to estimate each action.',
        scoring='Reference only reverses the known global translation while preserving the local warp. '
                'Report SHAFE MAE on the common 448 px crop.',
        inference='Paired bootstrap within camera by scene for each condition. Estimates and intervals '
                  'are descriptive, without a new threshold for claims.',
    ))
    print(f'Camera deformation lock {sha(LOCK)}', flush=True)


def verify_inputs():
    benchmarks.verify()
    plan = load(LOCK)
    for name, digest in plan['hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen camera deformation input changed: {name}')
    return plan


def summarize(rows):
    result = {}
    for condition in sorted({row['condition'] for row in rows}):
        cases = [r for r in rows if r['condition'] == condition]

        def event(row, method):
            return row['incremental_error'][method] > 1

        geometry = paired_summary(cases, METHODS, event, COMPARISONS)
        geometry['counts'] = {m: sum(event(row, m) for row in cases) for m in METHODS}
        result[condition] = dict(
            geometry=geometry,
            mean_residual=paired_summary(cases, METHODS, lambda r, m: r['incremental_error'][m], COMPARISONS),
            shafe_mae=paired_summary(cases, METHODS, lambda r, m: abs(r['scores'][m] - r['scores']['oracle']),
                                     COMPARISONS),
            unavailable={m: sum(not r['valid'][m] for r in cases) for m in METHODS},
        )
    return result


def run():
    plan = verify_inputs()
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

    zero = {r['scene']: r for r in load(BASELINE)['rows'] if r['condition'] == 'zero'}
    records = load(INPUTS)['records']
    assert set(zero) == {r['scene'] for r in records} == set(plan['parameters'])
    rows, unique_pairs = [], 0
    lock_hash = sha(LOCK)
    for i, record in enumerate(records, 1):
        scene = record['scene']
        path = DEST / 'scenes' / f'{scene}.json'
        stamp = dict(lock_sha256=lock_hash, sr_sha256=sha(record['sr']))
        cached = load(path) if path.exists() else {}
        if cached.get('stamp') != stamp:
            inputs = dict(record, direction=record['hr_direction'])
            measured, count = warp.process(inputs, plan['parameters'][scene], evaluate)
            for row in measured:
                row['incremental_error'] = {
                    m: linf([a - b + h for a, b, h in zip(row['actions'][m], zero[scene]['actions'][m],
                                                          row['hr_added'], strict=True)]) for m in METHODS
                }
            cached = dict(stamp=stamp, rows=measured, unique_pairs=count)
            atomic_dump(path, cached)
        rows.extend(cached['rows'])
        unique_pairs += cached['unique_pairs']
        if i % 8 == 0:
            print(f'{i}/{len(records)} camera scenes, {unique_pairs} SHAFE pairs', flush=True)
    conditions = {warp.condition(*args) for args in product(warp.SIDES, warp.AMPLITUDES, warp.TRANSLATIONS)}
    assert len(rows) == 256 and {(r['scene'], r['condition']) for r in rows} == set(product(zero, conditions))
    strata = summarize(rows)
    atomic_dump(DEST / 'results.json', dict(lock_sha256=lock_hash, unique_pairs=unique_pairs, rows=rows, strata=strata))
    for name, summary in strata.items():
        print(name, summary['geometry']['counts'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('freeze', 'run'))
    {'freeze': freeze, 'run': run}[parser.parse_args().step]()
