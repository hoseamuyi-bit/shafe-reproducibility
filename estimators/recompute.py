"""Recompute the manuscript's primary comparison from the exported cases (NumPy only)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TARGET = 0.8
METHODS = ('raw', 'direct', 'simple_lr', 'single', 'dual', 'legacy', 'oracle')


def load(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def correction(row, method):
    action = [float(row[f'{method}_{axis}']) for axis in ('dx', 'dy')]
    moving = max(map(abs, action)) > .25
    need = row['needs_correction'] == '1'
    residual = max(abs(value + float(row[f'hr_added_{axis}']))
                   for value, axis in zip(action, ('dx', 'dy')))
    return [need, not need, need and moving and residual <= .25,
            not need and moving, need and moving and residual > .25, not moving]


def crossing(grouped, weights, denominator):
    pos, neg = 0, 0
    target_count = TARGET * denominator[0]
    for positive, false in grouped:
        old_pos, old_neg = pos, neg
        pos += positive @ weights
        neg += false @ weights
        if pos >= target_count:
            return (old_neg + (target_count-old_pos)/(pos-old_pos)*(neg-old_neg)) / denominator[1]
    return None


def primary(rows):
    rows = [r for r in rows if r['degradation'] == 'strong']
    units = sorted({(r['dataset'], r['group']) for r in rows})
    index = {unit:i for i,unit in enumerate(units)}
    n = len(units)
    methods = ('single', 'dual')
    dataset_indices = {ds:[i for i, u in enumerate(units) if u[0] == ds] for ds, _ in units}
    denominators = np.zeros((n, 2), dtype=int)
    totals = {method:np.zeros((n, 6), dtype=int) for method in METHODS}
    groups = {method:{} for method in methods}
    for row in rows:
        i = index[(row['dataset'], row['group'])]
        need = row['needs_correction'] == '1'
        denominators[i, int(not need)] += 1
        for method in METHODS:
            totals[method][i] += correction(row, method)
        for method in methods:
            value = row[f'{method}_confidence']
            if value:
                count = groups[method].setdefault(float(value), np.zeros((2,n), dtype=int))
                count[0,i] += row['candidate_correct'] == '1'
                count[1,i] += not need
    weights = np.ones(n, dtype=int)
    bootstrap = np.zeros((5000,n), dtype=int)
    rng = np.random.default_rng(20261123)
    for ds in sorted(dataset_indices):
        indices = dataset_indices[ds]
        size = len(indices)
        bootstrap[:, indices] = rng.multinomial(size,np.full(size,1/size),size=len(bootstrap))

    overall_denominators = denominators.sum(axis=0)
    bootstrap_denominators = bootstrap @ denominators
    cuts = {}
    for method in methods:
        # Use integer cumulative event counts, independent of the frozen evaluator.
        factors = np.stack([groups[method][score] for score in sorted(groups[method], reverse=True)])
        cumulative = np.cumsum(factors, axis=0)
        correct, false = cumulative[:,0], cumulative[:,1]
        exact = crossing(factors, weights, overall_denominators)
        by_boot = []
        for w, (m,h) in zip(bootstrap, bootstrap_denominators):
            x = correct @ w
            first = int(np.searchsorted(x, TARGET*m))
            if first == len(x):
                by_boot.append(np.nan)
                continue
            p0 = x[first-1] if first else 0
            f0 = false[first-1] @ w if first else 0
            f1 = false[first] @ w
            by_boot.append((f0 + (TARGET*m - p0)/(x[first] - p0)*(f1 - f0))/h)
        cuts[method] = (exact, np.array(by_boot))
    a, b = (cuts[m][0] for m in methods)
    delta = cuts['single'][1] - cuts['dual'][1]
    paired = delta[np.isfinite(delta)]
    return dict(scenes=n, cases=len(rows),
                false_at_80=dict(single=a, dual=b), delta=a-b if a is not None and b is not None else None,
                delta_ci=np.quantile(paired,[.025,.975]).tolist() if len(paired) else None,
                unreachable_draws=int(len(delta)-len(paired)),
                fixed={m:v.sum(axis=0).tolist() for m,v in totals.items()},
                fixed_columns=['move_n','hold_n','correct_n','false_n','wrong_n','no_action_n'])


def shafe_mae(rows):
    labeled = {}
    for row in rows:
        for name in ('all', row['condition']):
            labeled.setdefault(f"{row['degradation']}/{name}", []).append(row)
    return {key:{m:float(np.mean([abs(float(r[f'{m}_shafe'])-float(r['oracle_shafe'])) for r in values]))
                 for m in METHODS} for key,values in sorted(labeled.items())}


def sr_only(rows):
    selected = [row for row in rows
                if row['degradation'] == 'strong' and row['condition'] == 'sr_global']
    return dict(n=len(selected),
                moved={m:sum(correction(row, m)[3] for row in selected) for m in METHODS},
                direct_score_decreased=sum(float(row['direct_shafe']) < float(row['oracle_shafe'])
                                           for row in selected))


def camera_response(rows):
    """Subtract each scene's zero-injection action before comparing with the injection."""
    lookup = {(r['scene'], r['condition']): r for r in rows}
    ap = lambda row, method: np.array([float(row[f'{method}_{axis}']) for axis in ('dx', 'dy')])
    result = []
    for condition in ('hr_global', 'sr_global', 'both_global'):
        for camera in ('All', 'Canon', 'Nikon'):
            subset = [r for r in rows if r['condition'] == condition and
                      (camera == 'All' or r['dataset'] == camera)]
            for method in ('direct', 'simple_lr', 'single', 'dual', 'legacy', 'hr_fit', 'sr_fit'):
                errors, changes = [], []
                selected_arm = 'sr' if method == 'sr_fit' else 'hr'
                for row in subset:
                    zero = lookup[(row['scene'], 'zero')]
                    if method in ('hr_fit', 'sr_fit') and any(
                        r[f'{selected_arm}_valid'] != '1' for r in (row, zero)
                    ):
                        continue
                    value = ap(row, method) - ap(zero, method)
                    truth = ap(row, f'{selected_arm}_added')
                    changes.append(float(np.max(np.abs(value))))
                    errors.append(float(np.max(np.abs(value + truth))))
                result.append(dict(condition=condition, camera=camera, method=method,
                    n=len(subset), valid=len(errors),
                    within_025=sum(x <= .25 for x in errors),
                    within_1=sum(x <= 1 for x in errors),
                    median_error=float(np.median(errors)) if errors else None,
                    changed=sum(x > .25 for x in changes)))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('formal', type=Path)
    cli.add_argument('--external', type=Path)
    cli.add_argument('--transfer', type=Path)
    args = cli.parse_args()
    formal = load(args.formal)
    report = dict(primary=primary(formal), formal_shafe_mae=shafe_mae(formal),
                  formal_sr_only=sr_only(formal))
    if args.external:
        external = load(args.external)
        report['external_shafe_mae'] = shafe_mae(external)
        report['external_response'] = camera_response(external)
    if args.transfer:
        transfer = load(args.transfer)
        report['transfer'] = dict(primary=primary(transfer), shafe_mae=shafe_mae(transfer),
                                  sr_only=sr_only(transfer),
                                  camera_response=camera_response(
                                      [r for r in transfer if r['degradation'] == 'camera']))
    print(json.dumps(report,indent=2,allow_nan=False))
