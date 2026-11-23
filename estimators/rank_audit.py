"""Pair the formal VOSR and Real-ESRGAN exports to assess SHAFE ranking stability."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ('raw', 'direct', 'simple_lr', 'single', 'dual', 'legacy', 'oracle')
MARGINS = (0.0, .0001, .001)
DRAWS = 5000
SEED = 20261126


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def pair(vosr, real_esrgan):
    def index(rows):
        selected = [r for r in rows if r['degradation'] == 'strong']
        result = {(r['scene'], r['condition']): r for r in selected}
        if len(result) != len(selected):
            raise ValueError('Duplicate scene and condition')
        return result

    vosr, real_esrgan = index(vosr), index(real_esrgan)
    if set(vosr) != set(real_esrgan) or len(vosr) != 150 * 8:
        raise ValueError('Expected 150 matched scenes with eight conditions each')
    columns = ('scene', 'dataset', 'group', 'condition', 'lr_pixel_sha256', 'hr_pixel_sha256',
               'hr_added_dx', 'hr_added_dy', 'sr_added_dx', 'sr_added_dy', 'oracle_dx', 'oracle_dy')
    rows = []
    for key in sorted(vosr):
        a, b = vosr[key], real_esrgan[key]
        if any(a[c] != b[c] for c in columns):
            raise ValueError(f'Inconsistent LR, HR or intervention at {key}')
        values = {m: float(a[f'{m}_shafe']) - float(b[f'{m}_shafe']) for m in METHODS}
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f'Nonfinite SHAFE score at {key}')
        actions = {m: [[float(r[f'{m}_{axis}']) for axis in ('dx', 'dy')] for r in (a, b)] for m in METHODS}
        rows.append(dict(**{c: a[c] for c in columns[:4]}, gap=values, actions=actions))
    counts = defaultdict(int)
    for row in rows:
        counts[row['dataset'], row['group']] += 1
    if len(counts) != 150 or set(counts.values()) != {8}:
        raise ValueError('Expected eight complete cases per independent scene')
    return rows


def bootstrap(units):
    strata = defaultdict(list)
    for i, (dataset, _) in enumerate(units):
        strata[dataset].append(i)
    rng = np.random.default_rng(SEED)
    weights = np.zeros((DRAWS, len(units)), dtype=np.int16)
    for indices in strata.values():
        n = len(indices)
        weights[:, indices] = rng.multinomial(n, np.full(n, 1 / n), size=DRAWS)
    return weights


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['group']].append(row)
    units = sorted(groups)
    if len({len(groups[u]) for u in units}) != 1:
        raise ValueError('Expected the same number of paired conditions per source scene')
    gaps = np.array([[[r['gap'][m] for m in METHODS] for r in groups[u]] for u in units])
    n = gaps.shape[0] * gaps.shape[1]
    weights = bootstrap(units)

    def paired_ci(values):
        return np.quantile((weights @ values.sum(axis=1)) / n, [.025, .975], axis=0).T.tolist()

    means = gaps.mean(axis=(0, 1))
    gap_ci = paired_ci(gaps)
    absolute_gap_error = np.abs(gaps - gaps[:, :, -1:])
    error_ci = paired_ci(absolute_gap_error)
    report = dict(n=n, scenes=len(units), models={m: dict(
        mean_gap=float(means[j]), mean_gap_ci95=gap_ci[j],
        absolute_gap_error=float(absolute_gap_error[:, :, j].mean()), absolute_gap_error_ci95=error_ci[j])
        for j, m in enumerate(METHODS)}, margins={})
    for margin in MARGINS:
        ranks = np.where(gaps > margin, 1, np.where(gaps < -margin, -1, 0))
        oracle = ranks[:, :, -1:]
        flips = (ranks * oracle == -1).astype(int)
        delta_ci = paired_ci(flips[:, :, 1:2] - flips)
        report['margins'][str(margin)] = dict(oracle_decisive_n=int(np.count_nonzero(oracle)), methods={m: dict(
            reversed_n=int(flips[:, :, j].sum()), delta_reversal_rate_direct_minus_method_ci95=delta_ci[j],
            changed_to_tie_n=int(np.count_nonzero((ranks[:, :, j:j+1] == 0) & (oracle != 0))),
            changed_from_tie_n=int(np.count_nonzero((ranks[:, :, j:j+1] != 0) & (oracle == 0))))
            for j, m in enumerate(METHODS)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vosr', type=Path)
    parser.add_argument('real_esrgan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = pair(read_csv(args.vosr), read_csv(args.real_esrgan))
    conditions = sorted({r['condition'] for r in rows})
    assert len(conditions) == 8
    summaries = {name: summarize([r for r in rows if r['condition'] == name]) for name in conditions}
    summaries['all'] = summarize(rows)
    report = dict(
        scope='Post hoc ranking analysis of paired strong-degradation cases with both restoration models.',
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.vosr, args.real_esrgan)},
        gap='SHAFE(VOSR) minus SHAFE(Real-ESRGAN). Lower SHAFE is preferred by the metric. Oracle '
            'keeps each SR fixed and reverses the added HR translation, including in the SR-only variants.',
        flips='A flip requires opposite SHAFE order with both gaps outside the specified tie margin. '
              'The denominator is all paired cases. This evaluates metric ranking relative to the known '
              'intervention, not agreement with human visual preference.',
        bootstrap=dict(unit='source scene, stratified by dataset', draws=DRAWS, seed=SEED),
        strata=summaries, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, data in summaries.items():
        counts = data['margins']['0.001']['methods']
        print(name, data['n'], {m: counts[m]['reversed_n'] for m in METHODS})


if __name__ == '__main__':
    main()
