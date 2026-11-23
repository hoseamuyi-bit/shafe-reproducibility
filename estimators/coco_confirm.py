"""Freeze and run the independent check of robust LR-anchored registration."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.stats import binomtest

from coco_inputs import FOLDER, N
from holdout_analysis import linf, paired_summary
from holdout_run import BASE, LOCK, OLD, ROOT, atomic_dump, images_for, load, sha, variants_for, verify_lock
from dev_pipeline import luminance, pixel_sha, project, read, shift
from experiment import infer
from registration import candidate, direct_candidate
from robust_controls import PLAN as CONTROL_LOCK, gray8, lucas_kanade
from robust_registration import tiled_candidates
from score import BATCH_SIZE, DEVICE, luminance_crop
from verify_holdout import event, exact_empirical_bootstrap


PLAN = FOLDER / 'lock.json'
INPUTS = FOLDER / 'batch_inputs.json'
CONDITIONS = ('zero', 'hr_global', 'sr_global', 'both_global', 'replacement_256', 'shift2_replacement_256')
MATCHERS = ('simple_lr', 'tile_mean', 'tile_trimmed', 'median_lk')
SCORED = ('raw', 'direct', *MATCHERS, 'oracle')
ENDPOINTS = {'replacement_256': 'false', 'shift2_replacement_256': 'correct', 'hr_global': 'correct'}
COMPARISONS = (('median_lk', 'simple_lr'), ('tile_trimmed', 'simple_lr'), ('median_lk', 'tile_trimmed'),
               ('tile_trimmed', 'tile_mean'))


def tracked():
    here = Path(__file__).resolve().parent
    return (Path(__file__).resolve(), here/'coco_inputs.py', here/'holdout_data.py', here/'holdout_run.py',
            here/'holdout_analysis.py', here/'verify_holdout.py', here/'robust_registration.py',
            here/'robust_controls.py', LOCK, CONTROL_LOCK, BASE/'robust_controls/results.json', INPUTS,
            FOLDER/'image_info_test2017.zip', ROOT/'environment_verified.json')


def freeze():
    original = verify_lock()
    if PLAN.exists() or (FOLDER/'geometry.json').exists() or (FOLDER/'geometry').exists():
        raise ValueError('Confirmation protocol or outcomes already exist')
    inputs = load(INPUTS)
    assert inputs['n_scenes'] == len(inputs['records']) == N
    historical = load(CONTROL_LOCK)
    assert historical['code_sha256'] == sha(Path(__file__).with_name('robust_controls.py'))
    atomic_dump(PLAN, dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        status='Independent confirmation after DIV2K training and its supplementary control results',
        hashes={str(p.relative_to(ROOT)): sha(p) for p in tracked()},
        population='144 COCO test2017 source scenes; original JPEGs, center HR crop, strong x4 degradation',
        conditions=CONDITIONS, registration=(
            'Use archived global loss, mean tile loss, best-half tile loss, and median Lucas–Kanade '
            'implementations and move thresholds without fitting on COCO. For SR scoring also compare raw, '
            'direct HR-to-SR, and an oracle that reverses only the known added HR translation.'),
        primary='Evaluate median Lucas–Kanade vs global loss on the two 256px replacement conditions '
                'and clean HR shifts. Compare best-half tiles with global loss as the replication of the '
                'DIV2K follow-up. Report paired median Lucas–Kanade vs best-half and all-tile ablation.',
        success='For each of the two methods versus global loss require at least 5 percentage point '
                'improvement and positive 95% interval lower bounds on both replacement endpoints, '
                'plus a clean-shift correct-rate difference with lower bound >= -5 percentage points. '
                'These are separate claims. Direct comparison between the robust methods is descriptive.',
        outcomes='False corrections require any applied motion >0.25 HR pixels; correct corrections '
                 'require motion and maximum remaining error <=0.25 HR pixels. Summarize by source '
                 'scene with 5000 paired bootstrap draws (seed 20261125). Also report exact empirical '
                 'bootstrap quantiles for the three contrasts and exact binomial limits for individual rates.',
        secondary='0.5/2/6px clean-shift strata, unavailability, zero-case and SR shift controls, '
                  'paired SHAFE MAE and signed score differences to known-inverse oracle.',
        shafe=original['shafe']))
    print(f'Frozen before geometry: {sha(PLAN)}', flush=True)


def verify():
    verify_lock()
    plan = load(PLAN)
    for name, expected in plan['hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Confirmation protocol changed: {name}')
    return plan


def reference_inputs(record):
    for field in ('hr', 'lr', 'source'):
        if sha(record[field]) != record[field+'_sha256']:
            raise ValueError(f'Input drift: {record[field]}')
    lr, hr, full = [read(record[field]) for field in ('lr', 'hr', 'source')]
    assert lr.shape == (128, 128, 3) and hr.shape == (512, 512, 3)
    assert pixel_sha(lr) == record['lr_pixel_sha256']
    return lr, hr, full


def geometry_scene(task):
    record, lock_hash = task
    cv2.setNumThreads(1)
    path = FOLDER/'geometry'/f'{record["scene"]}.json'
    if path.exists():
        cache = load(path)
        if cache['lock_sha256'] == lock_hash:
            return cache['rows']
    lr, hr, full = reference_inputs(record)
    anchor = gray8(luminance(lr))
    fits, rows = {}, []
    # HR-side estimators run independently of VOSR. The unused SR argument is only a placeholder.
    for condition, h, _, vector, _, patch in variants_for(record, hr, hr, full):
        if condition not in CONDITIONS:
            continue
        digest = pixel_sha(h)
        if digest not in fits:
            estimates = dict(simple_lr=candidate(h, lr), **tiled_candidates(h, lr))
            reference = gray8(project(luminance(h)))
            lk, tracks = lucas_kanade(anchor, reference)
            actions = {m: fit['vector'] if fit['executable'] else [0, 0] for m, fit in estimates.items()}
            actions['median_lk'] = lk
            fits[digest] = (actions, [m for m, fit in estimates.items() if not fit['valid']], tracks)
        actions, invalid, tracks = fits[digest]
        rows.append(dict(scene=record['scene'], dataset=record['dataset'], group=record['group'],
                         degradation='strong', condition=condition, magnitude=record['magnitude'],
                         hr_pixel_sha256=digest, lr_pixel_sha256=record['lr_pixel_sha256'],
                         hr_added=vector.tolist(), patch=patch, actions=actions, invalid=invalid, tracks=tracks))
    assert {row['condition'] for row in rows} == set(CONDITIONS)
    by_condition = {row['condition']: row for row in rows}
    for a, b in (('zero', 'sr_global'), ('hr_global', 'both_global')):
        assert by_condition[a]['hr_pixel_sha256'] == by_condition[b]['hr_pixel_sha256']
        assert by_condition[a]['actions'] == by_condition[b]['actions']
    atomic_dump(path, dict(lock_sha256=lock_hash, rows=rows))
    return rows


def run_geometry(workers):
    verify()
    lock_hash = sha(PLAN)
    records = load(INPUTS)['records']
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, group in enumerate(pool.map(geometry_scene, [(r, lock_hash) for r in records]), 1):
            rows.extend(group)
            if i % 12 == 0:
                print(f'COCO geometry {i}/{len(records)}', flush=True)
    assert len({(r['scene'], r['condition']) for r in rows}) == len(rows) == N * len(CONDITIONS)
    atomic_dump(FOLDER/'geometry.json', dict(lock_sha256=lock_hash, rows=rows))


def score_coco():
    plan = verify()
    sys.path.insert(0, str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT) == plan['shafe']['weight_sha256']
    path = FOLDER/'geometry.json'
    measured = load(path)
    assert measured['lock_sha256'] == sha(PLAN)
    indexed = {}
    for row in measured['rows']:
        indexed.setdefault(row['scene'], {})[row['condition']] = row
    stamp = dict(lock_sha256=sha(PLAN), geometry_sha256=sha(path), device=DEVICE, batch_size=BATCH_SIZE)
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    rows, pairs = [], 0
    for i, record in enumerate(load(INPUTS)['records'], 1):
        scene = record['scene']
        path = FOLDER/'scoring'/f'{scene}.json'
        scene_stamp = dict(stamp, sr_sha256=sha(record['sr']))
        bundle = load(path) if path.exists() else {}
        if bundle.get('stamp') != scene_stamp:
            result, n_pairs = score_input(record, indexed[scene], metric, paper_scores)
            bundle = dict(stamp=scene_stamp, rows=result, unique_pairs=n_pairs)
            atomic_dump(path, bundle)
        rows.extend(bundle['rows'])
        pairs += bundle['unique_pairs']
        if i % 12 == 0:
            print(f'COCO SHAFE {i}/{N} ({pairs} distinct pairs)', flush=True)
    assert len(rows) == len(measured['rows'])
    atomic_dump(FOLDER/'scores.json', dict(stamp=stamp, rows=rows, unique_pairs=pairs))


def score_input(record, geometry, metric, paper_scores):
    _, hr, sr, full = images_for(record)
    pairs, rows = {}, []
    for condition, h, s, vh, vs, _ in variants_for(record, hr, sr, full):
        if condition not in geometry:
            continue
        row = geometry[condition]
        assert pixel_sha(h) == row['hr_pixel_sha256'] and vh.tolist() == row['hr_added']
        direct = direct_candidate(h, s)
        actions = dict(row['actions'], raw=[0, 0], direct=direct if linf(direct) > .25 else [0, 0],
                       oracle=(-vh).tolist())
        fixed = luminance_crop(s)
        sr_digest = pixel_sha(fixed)
        keys = {}
        for method, vector in actions.items():
            reference = luminance_crop(shift(h, vector) if any(vector) else h)
            key = sr_digest + ':' + pixel_sha(reference)
            pairs.setdefault(key, (fixed, reference))
            keys[method] = key
        rows.append(dict(scene=row['scene'], group=row['group'], dataset=row['dataset'], condition=condition,
                         actions=actions, hr_added=row['hr_added'], sr_added=vs.tolist(), pair_ids=keys))
    assert len(rows) == len(CONDITIONS)
    values = dict(zip(pairs, paper_scores(metric, list(pairs.values()), DEVICE, BATCH_SIZE), strict=True))
    for row in rows:
        row['scores'] = {method: values[key] for method, key in row['pair_ids'].items()}
    return rows, len(pairs)


def geometry_reports(rows):
    reports = {}
    for condition in CONDITIONS:
        cases = [row for row in rows if row['condition'] == condition]
        for magnitude in (None, .5, 2, 6) if condition == 'hr_global' else (None,):
            subset = cases if magnitude is None else [r for r in cases if r['magnitude'] == magnitude]
            endpoint = 'correct' if any(subset[0]['hr_added']) else 'false'
            comparisons = COMPARISONS if endpoint == 'correct' else [(b, a) for a, b in COMPARISONS]
            result = paired_summary(subset, MATCHERS, lambda r, m: event(r, m, endpoint), comparisons)
            for method, value in result['estimates'].items():
                count = sum(event(r, method, endpoint) for r in subset)
                interval = binomtest(count, len(subset)).proportion_ci()
                value.update(count=count, binomial_ci95=[interval.low, interval.high],
                             unavailability=sum((r['tracks'] < 8 if method == 'median_lk'
                                                 else method in r['invalid']) for r in subset))
            if condition in ENDPOINTS and magnitude is None:
                for contrast in result['paired_differences']:
                    a, b = contrast['first'], contrast['second']
                    d = Counter(event(r, a, endpoint) - event(r, b, endpoint) for r in subset)
                    contrast.update(improved=d[1], worsened=d[-1], exact_bootstrap_ci95=exact_empirical_bootstrap(d))
            suffix = '' if magnitude is None else f'_{magnitude:g}px'
            reports[condition+suffix] = dict(result, endpoint=endpoint)
    return reports


def evaluate(include_scores):
    verify()
    path = FOLDER/'geometry.json'
    data = load(path)
    assert data['lock_sha256'] == sha(PLAN)
    reports = geometry_reports(data['rows'])
    verdicts = {}
    for i, method in enumerate(('median_lk', 'tile_trimmed')):
        false = reports['replacement_256']['paired_differences'][i]
        correct = reports['shift2_replacement_256']['paired_differences'][i]
        clean = reports['hr_global']['paired_differences'][i]
        assert false['second'] == correct['first'] == clean['first'] == method
        verdicts[method] = dict(false_reduction=false, correct_improvement=correct, clean_difference=clean,
                                meets_all=all(c['value'] >= .05 and c['ci95'][0] > 0 for c in (false, correct))
                                and clean['ci95'][0] >= -.05)
    output = dict(lock_sha256=sha(PLAN), geometry_sha256=sha(path), primary=verdicts, strata=reports)
    if include_scores:
        path = FOLDER/'scores.json'
        data = load(path)
        assert data['stamp']['geometry_sha256'] == output['geometry_sha256']
        shafe = {}
        methods = SCORED[:-1]
        for condition in (*CONDITIONS, 'all'):
            cases = [r for r in data['rows'] if condition in (r['condition'], 'all')]

            def signed(row, method):
                return row['scores'][method] - row['scores']['oracle']

            contrasts = (('simple_lr', 'median_lk'), ('tile_trimmed', 'median_lk'), ('simple_lr', 'tile_trimmed'))
            shafe[condition] = dict(
                absolute=paired_summary(cases, methods, lambda r, m: abs(signed(r, m)), contrasts),
                signed=paired_summary(cases, methods, signed, []),
                scoring_below_oracle={m: sum(signed(r, m) < -1e-6 for r in cases) for m in methods})
        output.update(scores_sha256=sha(path), shafe=shafe, unique_pairs=data['unique_pairs'])
    atomic_dump(FOLDER/('full_analysis.json' if include_scores else 'geometry_analysis.json'), output)
    for name, verdict in verdicts.items():
        print(name, verdict, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('freeze', 'infer', 'geometry', 'analyze', 'score', 'full'))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.step == 'freeze':
        freeze()
    elif args.step == 'infer':
        verify()
        infer(FOLDER)
    elif args.step == 'geometry':
        run_geometry(args.workers)
    elif args.step in ('analyze', 'full'):
        evaluate(args.step == 'full')
    else:
        score_coco()
