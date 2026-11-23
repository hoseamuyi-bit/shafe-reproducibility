"""Recompute the COCO and RealSR follow-up from archived result rows."""

import argparse
from collections import Counter
from itertools import product
import json
from math import isclose
from pathlib import Path

import numpy as np

from verify_holdout import digest, event, exact_empirical_bootstrap, max_abs, read


METHODS = ('simple_lr', 'tile_mean', 'tile_trimmed', 'median_lk')
SCORED = {*METHODS, 'raw', 'direct', 'oracle'}
LOCAL = ('replacement_256', 'shift2_replacement_256')
CONDITIONS = ('zero', 'hr_global', 'sr_global', 'both_global', *LOCAL)


def index(rows):
    keyed = {(row['scene'], row['condition']): row for row in rows}
    assert len(keyed) == len(rows)
    return keyed


def equal(actual, reported):
    np.testing.assert_allclose(actual, reported, rtol=0, atol=1e-12)


def check_summary(rows, published, outcome):
    """Compute statistic vectors by scene, then replay the frozen resampling."""
    methods = list(published['estimates'])
    ordered = sorted(rows, key=lambda r: (r['dataset'], r['group']))
    unique, counts = np.unique([(r['dataset'], r['group']) for r in ordered],
                               return_counts=True, axis=0)
    assert len(rows) == published['n'] and len(unique) == published['n_scenes']
    assert len(set(counts)) == 1  # one condition, or the same condition set for every scene
    points = np.array([[outcome(r, m) for m in methods] for r in ordered])
    assert np.isfinite(points).all()
    blocks = points.reshape(len(unique), counts[0], len(methods)).mean(axis=1)
    rng = np.random.default_rng(20261125)
    distributions = np.zeros((5000, len(methods)))
    for domain in sorted(set(unique[:, 0])):
        subset = blocks[unique[:, 0] == domain]
        weights = rng.multinomial(len(subset), [1 / len(subset)] * len(subset), size=5000)
        distributions += weights @ subset
    distributions /= len(unique)
    means = dict(zip(methods, points.mean(axis=0), strict=True))
    samples = dict(zip(methods, distributions.T, strict=True))
    for method, result in published['estimates'].items():
        equal(means[method], result['value'])
        equal(np.quantile(samples[method], [.025, .975]), result['ci95'])
        if 'count' in result:
            assert int(points[:, methods.index(method)].sum()) == result['count']
    for result in published['paired_differences']:
        a, b = result['first'], result['second']
        equal(means[a] - means[b], result['value'])
        equal(np.quantile(samples[a] - samples[b], [.025, .975]), result['ci95'])
        if 'exact_bootstrap_ci95' in result:
            counts = Counter(outcome(r, a) - outcome(r, b) for r in rows)
            assert counts[1] == result['improved'] and counts[-1] == result['worsened']
            equal(exact_empirical_bootstrap(counts), result['exact_bootstrap_ci95'])


def check_coco(base):
    lock = base / 'lock.json'
    geometry_path, scores_path = base / 'geometry.json', base / 'scores.json'
    geometry, scores = read(base, geometry_path.name), read(base, scores_path.name)
    full = read(base, 'full_analysis.json')
    assert geometry['lock_sha256'] == scores['stamp']['lock_sha256'] == full['lock_sha256'] == digest(lock)
    assert full['geometry_sha256'] == scores['stamp']['geometry_sha256'] == digest(geometry_path)
    assert full['scores_sha256'] == digest(scores_path)
    cases, scored = index(geometry['rows']), index(scores['rows'])
    records = {r['scene']: r for r in read(base, 'batch_inputs.json')['records']}
    assert len(records) == 144
    assert len(set(r['group'] for r in records.values())) == 144
    assert len(set(r['source_sha256'] for r in records.values())) == 144
    assert Counter(r['magnitude'] for r in records.values()) == {0.5: 48, 2: 48, 6: 48}
    assert set(cases) == set(scored) == set(product(records, CONDITIONS))
    all_pairs = 0
    for scene, record in records.items():
        direction = record['direction']
        magnitude = record['magnitude']
        hd = [value * magnitude for value in direction]
        # The frozen direction order assigns SR the other axis with the same sign.
        sd = hd[::-1]
        scene_pairs = {}
        for condition in CONDITIONS:
            row, score = cases[scene, condition], scored[scene, condition]
            zero = cases[scene, 'zero']
            assert row['lr_pixel_sha256'] == zero['lr_pixel_sha256'] == record['lr_pixel_sha256']
            expected_hr = hd if condition in ('hr_global', 'both_global') else [0, 0]
            if condition == LOCAL[1]:
                expected_hr = [2 * value for value in direction]
            assert row['hr_added'] == score['hr_added'] == expected_hr
            assert score['sr_added'] == (sd if condition in ('sr_global', 'both_global') else [0, 0])
            actions, ids, values = score['actions'], score['pair_ids'], score['scores']
            assert set(row['actions']) == set(METHODS)
            assert set(actions) == set(ids) == set(values) == SCORED
            assert all(actions[m] == row['actions'][m] for m in METHODS)
            assert actions['raw'] == [0, 0] and actions['oracle'] == [-value for value in expected_hr]
            assert len({pair.split(':')[0] for pair in ids.values()}) == 1
            for a, b in product(SCORED, repeat=2):
                if actions[a] == actions[b]:
                    assert ids[a] == ids[b] and values[a] == values[b]
            for method, pair in ids.items():
                assert scene_pairs.setdefault(pair, values[method]) == values[method]
            if condition not in ('sr_global', 'both_global'):
                assert ids['raw'].split(':')[0] == scored[scene, 'zero']['pair_ids']['raw'].split(':')[0]
        for first, second in (('zero', 'sr_global'), ('hr_global', 'both_global')):
            a, b = (cases[scene, c] for c in (first, second))
            assert a['hr_pixel_sha256'] == b['hr_pixel_sha256'] and a['actions'] == b['actions']
        all_pairs += len(scene_pairs)
    assert all_pairs == scores['unique_pairs'] == full['unique_pairs']

    for name, result in full['strata'].items():
        condition = 'hr_global' if name.startswith('hr_global_') else name
        magnitude = float(name.removeprefix('hr_global_').removesuffix('px')) if name != condition else None
        subset = [r for r in cases.values() if r['condition'] == condition
                  and (magnitude is None or r['magnitude'] == magnitude)]
        endpoint = 'correct' if max_abs(subset[0]['hr_added']) else 'false'
        assert result['endpoint'] == endpoint
        check_summary(subset, result, lambda r, m: event(r, m, endpoint))
        for method, rate in result['estimates'].items():
            low, high = rate['binomial_ci95']
            assert 0 <= low <= rate['value'] <= high <= 1
            assert rate['unavailability'] == sum(r['tracks'] < 8 if method == 'median_lk'
                                                 else method in r['invalid'] for r in subset)
    for i, method in enumerate(('median_lk', 'tile_trimmed')):
        false, correct, clean = [full['strata'][c]['paired_differences'][i] for c in (*LOCAL, 'hr_global')]
        verdict = full['primary'][method]
        assert (false, correct, clean) == (verdict['false_reduction'], verdict['correct_improvement'],
                                          verdict['clean_difference'])
        assert verdict['meets_all'] == (false['value'] >= .05 and false['ci95'][0] > 0
                                       and correct['value'] >= .05 and correct['ci95'][0] > 0
                                       and clean['ci95'][0] >= -.05)

    for condition, report in full['shafe'].items():
        subset = [r for r in scored.values() if condition in (r['condition'], 'all')]

        def signed(r, method):
            return r['scores'][method] - r['scores']['oracle']

        check_summary(subset, report['signed'], signed)
        check_summary(subset, report['absolute'], lambda r, m: abs(signed(r, m)))
        for method, count in report['scoring_below_oracle'].items():
            assert count == sum(signed(r, method) < -1e-6 for r in subset)
    return cases, dict(scenes=len(records), cases=len(cases), unique_score_pairs=all_pairs,
                       shafe_strata=len(full['shafe']), geometry_strata=len(full['strata']),
                       scores_sha256=digest(scores_path))


def check_taper(base, cases):
    folder = base / 'sensitivity'
    taper = read(folder, 'coco_taper_analysis.json')
    assert taper['lock_sha256'] == digest(folder / 'lock.json')
    assert taper['geometry_sha256'] == digest(base / 'geometry.json')
    rows = index(taper['rows'])
    assert rows.keys() == {key for key in cases if key[1] in LOCAL}
    for key, row in rows.items():
        reference = cases[key]
        assert row['overlap_pixels'] == reference['patch']['overlap_pixels']
        assert row['hr_added'] == reference['hr_added']
    for label, report in taper['strata'].items():
        condition, scope = label.split('/')
        subset = [r for r in rows.values() if r['condition'] == condition
                  and (scope == 'all' or bool(r['overlap_pixels']) == (scope == 'overlap'))]
        endpoint = 'false' if condition == LOCAL[0] else 'correct'

        def outcome(row, name):
            mode, method = name.split('/')
            case = cases[row['scene'], condition] if mode == 'hard' else row
            return event(case, method, endpoint)

        check_summary(subset, report, outcome)
    return dict(cases=len(rows), strata=len(taper['strata']))


def check_pixel_replay(root):
    base = root / 'coco_confirm'
    replay = read(root, 'coco_confirmation_pixel_replay.json')
    assert replay['completed_utc']
    provenance = replay['provenance']
    files = {
        'replay_sha256': root / 'replay_coco_confirmation.py',
        'lock_sha256': base / 'lock.json',
        'geometry_sha256': base / 'geometry.json',
        'scores_sha256': base / 'scores.json',
        'manifest_sha256': base / 'batch_inputs.json',
    }
    for key, path in files.items():
        assert provenance[key] == digest(path)
    assert provenance['weight_sha256'] == read(base, 'lock.json')['shafe']['weight_sha256']
    assert provenance['score_tolerance'] == 1e-6
    scores = read(base, 'scores.json')
    assert provenance['device'] == scores['stamp']['device']
    assert provenance['batch_size'] == scores['stamp']['batch_size']
    pair_ids = {}
    for row in scores['rows']:
        pair_ids.setdefault(row['scene'], set()).update(row['pair_ids'].values())
    checks = replay['replayed']
    assert checks.keys() == pair_ids.keys()
    for scene, check in checks.items():
        assert check['conditions'] == len(CONDITIONS)
        assert check['pairs'] == len(pair_ids[scene])
        assert 0 <= check['max_score_error'] <= provenance['score_tolerance']
    summary = dict(scenes=len(checks), conditions=len(scores['rows']), pairs=scores['unique_pairs'],
                   max_score_error=max(r['max_score_error'] for r in checks.values()))
    assert replay['summary'] == summary
    return summary


def check_realsr(base, camera):
    folder = base / 'sensitivity'
    result = read(folder, 'realsr_lk_analysis.json')
    assert result['lock_sha256'] == digest(folder / 'lock.json')
    assert result['prior_scores_sha256'] == digest(camera / 'scores.json')
    prior = read(camera, 'scores.json')
    assert prior['geometry_sha256'] == digest(camera / 'geometry.json')
    before, after = index(prior['rows']), index(result['rows'])
    scenes = {key[0] for key in before}
    assert len(scenes) == 32 and before.keys() == after.keys() == set(product(scenes, ('zero', 'shift2', *LOCAL)))
    for key, row in after.items():
        old = before[key]
        assert all(row[k] == old[k] for k in ('hr_pixel_sha256', 'sr_pixel_sha256', 'hr_added'))
        assert all(row['scores'][m] == value for m, value in old['scores'].items())
        assert all(row['actions'][m] == action for m, action in old['actions'].items())
        baseline = after[row['scene'], 'zero']['actions']
        for m, action in row['actions'].items():
            remaining = max_abs([a - b + h for a, b, h in zip(action, baseline[m], row['hr_added'], strict=True)])
            assert isclose(remaining, row['incremental_error'][m], abs_tol=1e-12)
        known = dict(row['actions'], oracle=[-x for x in row['hr_added']])
        for m, vector in known.items():
            if vector == known['median_lk']:
                equal(row['scores']['median_lk'], row['scores'][m])
    for condition, report in result['strata'].items():
        subset = [r for r in after.values() if r['condition'] == condition]
        check_summary(subset, report['error_gt_1px'], lambda r, m: r['incremental_error'][m] > 1)
        check_summary(subset, report['shafe_mae'], lambda r, m: abs(r['scores'][m] - r['scores']['oracle']))
        assert report['lk_fewer_than_eight_tracks'] == sum(r['tracks'] < 8 for r in subset)
    return dict(scenes=len(scenes), cases=len(after), conditions_checked=len(result['strata']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit_root', type=Path)
    parser.add_argument('--pixel-replay', action='store_true', help='also validate the confirmation replay records')
    args = parser.parse_args()
    root = args.audit_root
    coco = root / 'coco_confirm'
    cases, confirmation = check_coco(coco)
    report = dict(coco=confirmation, taper=check_taper(coco, cases),
                  realsr=check_realsr(coco, root / 'holdout' / 'camera_local'))
    if args.pixel_replay:
        report['pixel_replay'] = check_pixel_replay(root)
    print(json.dumps(report, indent=2))
