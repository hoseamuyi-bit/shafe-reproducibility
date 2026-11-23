"""Source-scene paired inference and metric checks for the follow-up."""

import argparse
from collections import defaultdict

import numpy as np

from holdout_run import (BASE, BOOTSTRAP_N, BOOTSTRAP_SEED, LOCK, METHODS, ROOT,
                         atomic_dump, load, sha, verify_lock)


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def draw_weights(units):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.zeros((BOOTSTRAP_N, len(units)), dtype=np.int16)
    for dataset in sorted({unit[0] for unit in units}):
        indices = [i for i, unit in enumerate(units) if unit[0] == dataset]
        count = len(indices)
        draws[:, indices] = rng.multinomial(count, np.full(count, 1 / count), size=BOOTSTRAP_N)
    return draws


def paired_summary(rows, methods, value, comparisons):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row['dataset'], row['group']].append(row)
    units = sorted(by_scene)
    sizes = np.array([len(by_scene[unit]) for unit in units])
    values = np.array([[sum(value(row, m) for row in by_scene[unit]) for m in methods] for unit in units])
    assert len(rows) == sizes.sum() and np.isfinite(values).all()
    draws = draw_weights(units)
    point = dict(zip(methods, (values.sum(axis=0)/len(rows)).tolist(), strict=True))
    samples = dict(zip(methods, ((draws @ values)/(draws @ sizes)[:, None]).T, strict=True))
    return dict(n=len(rows), n_scenes=len(units), estimates={m:dict(value=point[m], ci95=interval(samples[m]))
                                                           for m in methods},
                paired_differences=[dict(first=a, second=b, value=point[a]-point[b],
                                         ci95=interval(samples[a]-samples[b])) for a, b in comparisons])


def movement(vector):
    return bool(any(vector))


def linf(vector):
    return float(np.max(np.abs(vector)))


def synthetic_geometry(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['degradation'], row['condition']].append(row)
    comparisons = [('tile_trimmed', name) for name in ('simple_lr', 'tile_mean', 'single', 'dual')]
    result = {}
    for (degradation, condition), cases in sorted(buckets.items()):
        for magnitude in (None, .5, 2, 6) if condition in ('hr_global', 'both_global') else (None,):
            subset = cases if magnitude is None else [r for r in cases if r['magnitude'] == magnitude]
            keep = not movement(subset[0]['hr_added'])
            assert all((not movement(r['hr_added'])) == keep for r in subset)
            methods = tuple(m for m in METHODS if m != 'oracle')
            if 'replacement' in condition:
                methods += ('mask_diagnostic',)

            def action(row, method):
                if method != 'mask_diagnostic':
                    return row['actions'][method]
                fit = row[method]
                return fit['vector'] if fit['executable'] else [0, 0]

            def correct(row, method):
                vector = action(row, method)
                return movement(vector) and linf(np.asarray(vector)+row['hr_added']) <= .25

            def event(row, method):
                return movement(action(row, method)) if keep else correct(row, method)

            pairs = [(b, a) for a, b in comparisons] if keep else comparisons
            report = paired_summary(subset, methods, event, pairs)
            measures = report['estimates']
            for name in methods:
                measures[name]['event_n'] = sum(event(r, name) for r in subset)
                measures[name]['no_action_n'] = sum(not movement(action(r, name)) for r in subset)
                measures[name]['wrong_shift_n'] = sum(movement(action(r, name)) and not correct(r, name)
                                                      for r in subset) if not keep else 0
            suffix = f'_{magnitude:g}px' if magnitude is not None else ''
            result[f'{degradation}/{condition}{suffix}'] = dict(
                report, endpoint='false_correction' if keep else 'correct_correction')
    primary = [result[key]['paired_differences'][0] for key in
               ('strong/replacement_256', 'strong/shift2_replacement_256', 'strong/hr_global')]
    passes = all(x['value'] >= .05 and x['ci95'][0] > 0 for x in primary[:2]) and primary[2]['ci95'][0] >= -.05
    return dict(strata=result, primary=dict(false_reduction=primary[0], correct_improvement=primary[1],
                                 clean_global_difference=primary[2], meets_frozen_criteria=passes))


def camera_geometry(rows):
    original = {(r['scene'], r['condition']): r for r in load(ROOT/'external/decisions.json')['rows']}
    zero = {r['scene']:r for r in rows if r['condition'] == 'zero'}
    strata = defaultdict(list)
    for row in rows:
        archived = original[row['scene'], row['condition']]
        for key in ('hr_pixel_sha256', 'lr_pixel_sha256', 'sr_pixel_sha256', 'hr_added', 'sr_added'):
            assert row[key] == archived[key], (row['scene'], row['condition'], key)
        for method in ('raw', 'direct', 'simple_lr', 'single', 'dual', 'oracle'):
            assert row['actions'][method] == archived['actions'][method], (row['scene'], row['condition'], method)
        strata['both_cameras', row['condition']].append(row)
        strata[row['dataset'], row['condition']].append(row)

    def action_agrees(row, method):
        base = zero[row['scene']]['actions'][method]
        vector = np.asarray(row['actions'][method]) - base + row['hr_added']
        # SR-only inputs should not change the chosen HR action at all.
        tolerance = 0. if row['condition'] == 'sr_global' else 1.
        return linf(vector) <= tolerance

    def candidate_agrees(row, method):
        current = row[method]
        base = zero[row['scene']][method]
        return current['valid'] and base['valid'] and linf(
            np.asarray(current['vector']) - base['vector'] + row['hr_added']) <= 1.

    result = {}
    for (dataset, condition), cases in sorted(strata.items()):
        if condition == 'zero':
            result[f'{dataset}/{condition}'] = dict(n=len(cases),
                nonzero_actions={m:sum(movement(r['actions'][m]) for r in cases) for m in METHODS})
            continue
        methods = tuple(m for m in METHODS if m != 'oracle')
        action_report = paired_summary(cases, methods, action_agrees, [('tile_trimmed', 'simple_lr')])
        error_report = paired_summary(cases, ('hr_fit', 'tile_mean', 'tile_trimmed'), candidate_agrees,
                                      [('tile_trimmed', 'hr_fit')])
        result[f'{dataset}/{condition}'] = dict(action_agreement=action_report, candidate_agreement=error_report)
    return dict(archive_actions_exact_match=len(rows), strata=result)


def shafe(geometry, scored):
    by_case = {(r['scene'], r['degradation'], r['condition']):r['scores'] for r in scored}
    assert len(by_case) == len(geometry)
    buckets = defaultdict(list)
    for row in geometry:
        scores = by_case[row['scene'], row['degradation'], row['condition']]
        assert all(np.isfinite(v) for v in scores.values())
        joined = dict(row, scores=scores)
        for key in ('all', row['condition']):
            buckets[row['degradation'], key].append(joined)

    def signed(row, method):
        return row['scores'][method] - row['scores']['oracle']

    methods = tuple(m for m in METHODS if m != 'oracle')
    comparisons = [(m, 'tile_trimmed') for m in ('simple_lr', 'tile_mean', 'single', 'dual')]
    return {f'{kind}/{condition}': dict(
        absolute=paired_summary(cases, methods, lambda row, m: abs(signed(row, m)), comparisons),
        signed_mean={m:float(np.mean([signed(r, m) for r in cases])) for m in methods},
        scoring_below_oracle={m:sum(signed(r, m) < -1e-6 for r in cases) for m in methods})
        for (kind, condition), cases in sorted(buckets.items())}


def analyze(arm, include_scores):
    verify_lock()
    folder = BASE/arm
    path = folder/'geometry.json'
    geometry = load(path)
    if geometry['lock_sha256'] != sha(LOCK):
        raise ValueError('Different frozen holdout')
    result = dict(lock_sha256=sha(LOCK), geometry_sha256=sha(path), bootstrap_n=BOOTSTRAP_N,
                  bootstrap_seed=BOOTSTRAP_SEED, **(synthetic_geometry(geometry['rows']) if arm == 'synthetic'
                              else camera_geometry(geometry['rows'])))
    label = 'geometry_analysis'
    if include_scores:
        path_scores = folder/'scores.json'
        scores = load(path_scores)
        if scores['stamp']['geometry_sha256'] != result['geometry_sha256']:
            raise ValueError('Different scored geometry')
        result.update(scores_sha256=sha(path_scores), shafe_unique_pairs=scores['unique_pairs'],
                      shafe=shafe(geometry['rows'], scores['rows']))
        label = 'full_analysis'
    atomic_dump(folder/f'{label}.json', result)
    if arm == 'synthetic':
        print(result['primary'], flush=True)
    else:
        print(f"Confirmed original baseline actions and image hashes for {result['archive_actions_exact_match']} "
              'camera injections', flush=True)
    print(f'{arm}: {label} complete', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('arm', choices=('synthetic', 'camera'))
    cli.add_argument('--scores', action='store_true')
    args = cli.parse_args()
    analyze(args.arm, args.scores)
