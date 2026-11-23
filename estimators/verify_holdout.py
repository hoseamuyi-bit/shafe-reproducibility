"""Recompute the follow-up's geometry and SHAFE checks from frozen case rows."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import comb, isclose, isfinite
from pathlib import Path


N = 96
RATES = (
    ('replacement_256', 'simple_lr', 'tile_trimmed', 'false'),
    ('shift2_replacement_256', 'tile_trimmed', 'simple_lr', 'correct'),
    ('hr_global', 'tile_trimmed', 'simple_lr', 'correct'),
)
METHODS = {'raw', 'direct', 'simple_lr', 'single', 'dual', 'tile_mean', 'tile_trimmed', 'oracle'}


def read(folder, name):
    return json.loads((folder / name).read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def case_key(row):
    return row['scene'], row['degradation'], row['condition']


def max_abs(vector):
    return max(map(abs, vector))


def event(row, method, endpoint):
    action = row['actions'][method]
    if endpoint == 'false':
        return int(max_abs(action) > .25)
    residual = [a + b for a, b in zip(action, row['hr_added'], strict=True)]
    return int(max_abs(action) > .25 and max_abs(residual) <= .25)


def exact_empirical_bootstrap(frequencies):
    """Enumerate the resampled {-1, 0, 1} paired-difference counts."""
    n = sum(frequencies.values())
    p0, p_negative, p_positive = (frequencies[k] / n for k in (0, -1, 1))
    masses = Counter()
    for positives in range(n + 1):
        for negatives in range(n - positives + 1):
            zeros = n - positives - negatives
            masses[positives - negatives] += (comb(n, positives) * comb(n - positives, negatives)
                                            * p_positive**positives * p_negative**negatives * p0**zeros)
    assert isclose(sum(masses.values()), 1, abs_tol=1e-12)
    limits = []
    for q in (.025, .975):
        cumulative = 0.
        for difference, probability in sorted(masses.items()):
            cumulative += probability
            if cumulative >= q:
                limits.append(difference / n)
                break
    return limits


def verify(base):
    folder = base / 'synthetic'
    geometry = read(folder, 'geometry.json')
    scores = read(folder, 'scores.json')
    full_analysis = read(folder, 'full_analysis.json')
    assert geometry['lock_sha256'] == scores['stamp']['lock_sha256'] == digest(base / 'lock.json')
    assert full_analysis['geometry_sha256'] == scores['stamp']['geometry_sha256'] == digest(folder / 'geometry.json')
    assert full_analysis['scores_sha256'] == digest(folder / 'scores.json')
    assert len(geometry['rows']) == len(scores['rows']) == 2304
    cases = {case_key(row): row for row in geometry['rows']}
    measured = {case_key(row): row for row in scores['rows']}
    assert cases.keys() == measured.keys() and len(cases) == 2304
    scenes = {key[0] for key in cases}
    conditions = {'zero', 'hr_global', 'sr_global', 'both_global', 'sr_local', 'hr2_sr_local',
                  *(f'{p}replacement_{size}' for p in ('', 'shift2_') for size in (128, 256, 320))}
    assert len(scenes) == N
    assert cases.keys() == set(product(scenes, ('normal', 'strong'), conditions))
    for scene, degradation in product(scenes, ('normal', 'strong')):
        zero = cases[scene, degradation, 'zero']
        assert all(cases[scene, degradation, c]['lr_pixel_sha256'] == zero['lr_pixel_sha256'] for c in conditions)
        for a, b in [('zero', 'sr_global'), ('zero', 'sr_local'), ('hr_global', 'both_global')]:
            first, second = (cases[scene, degradation, c] for c in (a, b))
            assert first['hr_pixel_sha256'] == second['hr_pixel_sha256']
            assert all(first['actions'][m] == second['actions'][m]
                       for m in ('simple_lr', 'single', 'tile_mean', 'tile_trimmed'))
    for key, row in cases.items():
        assert row['actions']['oracle'] == [-value for value in row['hr_added']]
        scored = measured[key]
        actions, values, ids = row['actions'], scored['scores'], scored['pair_ids']
        assert set(actions) == set(values) == set(ids) == METHODS
        assert len({pair.split(':')[0] for pair in ids.values()}) == 1
        for a, b in product(METHODS, repeat=2):
            if actions[a] == actions[b]:
                assert ids[a] == ids[b] and values[a] == values[b]
        assert all(isfinite(value) for value in values.values())

    for key, metrics in full_analysis['shafe'].items():
        degradation, condition = key.split('/')
        rows = [row['scores'] for (_, d, c), row in measured.items()
                if d == degradation and (condition == 'all' or c == condition)]
        assert len(rows) == metrics['absolute']['n']
        for method, reported in metrics['absolute']['estimates'].items():
            mae = sum(abs(r[method] - r['oracle']) for r in rows) / len(rows)
            assert isclose(mae, reported['value'], rel_tol=0, abs_tol=1e-12)

    assessments = {}
    for (condition, a, b, endpoint), contrast_key in zip(
            RATES, ('false_reduction', 'correct_improvement', 'clean_global_difference'), strict=True):
        pairs = [(event(cases[s, 'strong', condition], a, endpoint),
                  event(cases[s, 'strong', condition], b, endpoint)) for s in sorted(scenes)]
        frequencies = Counter(x - y for x, y in pairs)
        delta = (frequencies[1] - frequencies[-1]) / N
        published = full_analysis['primary'][contrast_key]
        assert (a, b) == (published['first'], published['second'])
        assert isclose(delta, published['value'], rel_tol=0, abs_tol=1e-12)
        assessments[contrast_key] = dict(condition=condition, endpoint=endpoint,
            rates={a: sum(x for x, _ in pairs) / N, b: sum(y for _, y in pairs) / N},
            paired_positive=frequencies[1], paired_negative=frequencies[-1], effect=delta,
            frozen_bootstrap_ci95=published['ci95'], exact_empirical_bootstrap_ci95=exact_empirical_bootstrap(frequencies))
    return dict(n_cases=len(cases), n_scenes=N, score_strata_checked=len(full_analysis['shafe']),
                geometry_sha256=digest(folder / 'geometry.json'), scores_sha256=digest(folder / 'scores.json'),
                strong_primary=assessments, meets_frozen_criteria=full_analysis['primary']['meets_frozen_criteria'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.base), indent=2))
