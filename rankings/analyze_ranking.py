"""Analyze paired rank reversal against an HR-only geometric correction."""
from collections import Counter
import itertools

import numpy as np

from ranking_common import CONDITIONS, METRICS, METHODS, MODELS, ROOT, atomic_json, load, verify_lock

PAIRS = tuple(itertools.combinations(range(len(MODELS)), 2))
PAIR_NAMES = tuple(f'{MODELS[a]} vs {MODELS[b]}' for a, b in PAIRS)
SIGNS = np.array([1, 1, -1, -1])
EPSILONS = (1e-6, 1e-7, 1e-6, 1e-6)
NEAR_TIE = (.05, .0005, .0005, .0005)
BOOT_SEED = 20261130
BOOT_REPS = 5000
MIN_PP = 2


def sign(delta, tolerance):
    return np.where(delta > tolerance, 1, np.where(delta < -tolerance, -1, 0))


def pair_delta(values):
    return np.stack([values[..., a, :] - values[..., b, :] for a, b in PAIRS], axis=-2)


def bootstrap_weights(strata, rng):
    groups = {}
    for i, key in enumerate(strata):
        groups.setdefault(key, []).append(i)
    weights = np.zeros((BOOT_REPS, len(strata)), dtype=float)
    for indices in groups.values():
        weights[:, indices] = rng.multinomial(len(indices), [1. / len(indices)] * len(indices),
                                              size=BOOT_REPS) / len(strata)
    return weights


def effect(values, weights, scale=1):
    mean = scale * float(np.mean(values))
    ci = np.quantile(scale * (weights @ values), [.025, .975]).tolist()
    return dict(estimate=mean, ci95=ci)


def ranking(values, tolerance):
    ordered = sorted(zip(MODELS, values.tolist()), key=lambda item: -item[1])
    groups = []
    first = None
    for model, score in ordered:
        if first is None or first - score > tolerance:
            groups.append([])
            first = score
        groups[-1].append(model)
    return groups


def view(values, weights, tolerance):
    # Axes: scene, alignment method, model, metric.
    oracle_idx = METHODS.index('oracle')
    lk_idx = METHODS.index('lk_lr')
    direct_idx = METHODS.index('direct')
    with np.errstate(invalid='ignore'):
        delta = pair_delta(values)
    # Two exact PSNR matches tie. None are expected with the locked SR models.
    pair_sign = sign(delta, tolerance)
    oracle_sign = pair_sign[:, oracle_idx:oracle_idx + 1]
    flips = pair_sign * oracle_sign == -1
    n = flips.shape[0] * len(PAIRS)
    to_bootstrap = (flips[:, direct_idx].astype(float) - flips[:, lk_idx]).mean(axis=1)
    summary = {}
    for q, name in enumerate(METRICS):
        true_non_tie = oracle_sign[:, 0, :, q] != 0
        reference_means = delta[:, oracle_idx, :, q]
        reference_order = sign(reference_means.mean(axis=0), tolerance[q])
        bootstrap_reference = sign(weights @ reference_means, tolerance[q])
        per_method = {}
        for m, method in enumerate(METHODS):
            score = values[:, m, :, q].mean(axis=0)
            aggregate_diff = delta[:, m, :, q].mean(axis=0)
            mean_replicates = weights @ delta[:, m, :, q]
            intervals = np.quantile(mean_replicates, [.025, .975], axis=0).T
            aggregate_flips = sign(aggregate_diff, tolerance[q]) * reference_order == -1
            bootstrap_flips = sign(mean_replicates, tolerance[q]) * bootstrap_reference == -1
            methodsign, refsign = pair_sign[:, m, :, q], oracle_sign[:, 0, :, q]
            table = Counter(zip(refsign.ravel().tolist(), methodsign.ravel().tolist()))
            per_method[method] = dict(
                flips=int(flips[:, m, :, q].sum()), total=n,
                jointly_ordered=int((true_non_tie & (methodsign != 0)).sum()),
                oracle_tied=int((~true_non_tie).sum()), method_tied=int((methodsign == 0).sum()),
                agreement={f'{a},{b}': table[a, b] for a, b in itertools.product((-1, 0, 1), repeat=2)},
                flips_by_model_pair=dict(zip(PAIR_NAMES, flips[:, m, :, q].sum(axis=0).tolist())),
                mean_score=dict(zip(MODELS, (score * SIGNS[q]).tolist())),
                mean_ranking=ranking(score, tolerance[q]),
                mean_ranking_reversed_pairs=[pair for pair, flipped in zip(PAIR_NAMES, aggregate_flips)
                                             if flipped],
                mean_ranking_reversal_bootstrap_fraction=dict(zip(PAIR_NAMES,
                                                                  bootstrap_flips.mean(axis=0).tolist())),
                oriented_mean_difference={pair: dict(estimate=float(aggregate_diff[k]),
                                                      ci95=intervals[k].tolist())
                                          for k, pair in enumerate(PAIR_NAMES)})
        summary[name] = dict(direct_minus_lk_pp=effect(to_bootstrap[:, q], weights, 100),
                             methods=per_method)
    return summary


def read_scene_cube(records):
    _, lock_sha = verify_lock()
    cube = np.empty((len(records), len(CONDITIONS), len(METHODS), len(MODELS), len(METRICS)))
    results = []
    for i, record in enumerate(records):
        obj = load(ROOT / 'formal/scores' / f'{record["scene"]}.json')
        if obj['checks']['config'] != lock_sha or obj['lr_pixel_sha256'] != record['lr_pixel_sha256']:
            raise ValueError('Scene score provenance mismatch')
        results.append(obj)
        cases = {(r['condition'], r['model']): r for r in obj['cases']}
        if len(cases) != 16 or len(obj['cases']) != 16:
            raise ValueError('Incomplete or nonunique scene conditions')
        for j, c in enumerate(CONDITIONS):
            for m, model in enumerate(MODELS):
                for k, method in enumerate(METHODS):
                    scores = cases[c, model]['scores'][method]
                    cube[i, j, k, m] = [float(scores[name]) for name in METRICS]
    if not np.isfinite(cube).all():
        raise ValueError('Infinite PSNR requires a separate aggregate-score treatment')
    return cube, results


def analyze():
    records = load(ROOT / 'inputs/manifest.json')['records']
    cube, objects = read_scene_cube(records)
    oriented = cube * SIGNS
    split = np.array([r['split'] for r in records])
    amplitude = np.array([r['magnitude'] for r in records])
    strata = list(zip(split, amplitude))
    rng = np.random.default_rng(BOOT_SEED)
    all_weights = bootstrap_weights(strata, rng)
    pooled = {cond: view(oriented[:, k], all_weights, EPSILONS) for k, cond in enumerate(CONDITIONS)}
    sr = CONDITIONS.index('sr_global')
    zero = CONDITIONS.index('zero')
    sensitivity = view(oriented[:, sr], all_weights, NEAR_TIE)
    splits = {}
    for name, mask in [(str(s), split == s) for s in sorted(set(split))] + [
            (f'{a:g} px', amplitude == a) for a in sorted(set(amplitude))]:
        weights = bootstrap_weights([k for k, use in zip(strata, mask, strict=True) if use], rng)
        splits[name] = view(oriented[mask, sr], weights, EPSILONS)
    one_sr = {}
    for m, model in enumerate(MODELS):
        one_shifted = oriented[:, zero].copy()
        one_shifted[:, :, m] = oriented[:, sr, :, m]
        one_sr[model] = view(one_shifted, all_weights, EPSILONS)
    primary = pooled['sr_global']['SHAFE']['direct_minus_lk_pp']
    geometry = {}
    for cond in CONDITIONS:
        eligible = [r for o in objects for r in o['cases'] if r['condition'] == cond]
        hr_cases = [r for r in eligible if r['model'] == MODELS[0]]
        geometry[cond] = dict(by_method={
            m: dict(n=len(cases),
                    moves=sum(any(r['actions'][m]) for r in cases),
                    residual_over_quarter_px=sum(max(abs(x+y) for x, y in
                                                    zip(r['actions'][m], r['hr_added'])) > .25
                                                  for r in cases))
            for m, cases in (('direct', eligible), ('global_lr', hr_cases), ('lk_lr', hr_cases))},
            global_unavailable=sum(not r['global_valid'] for r in hr_cases),
            lk_unavailable=sum(r['lk_tracks'] < 8 for r in hr_cases))
    fixed = METHODS.index('fixed')
    direct = METHODS.index('direct')
    favorable = (oriented[:, sr, direct] - oriented[:, sr, fixed]).mean(axis=1)
    oracle = METHODS.index('oracle')
    clean = METHODS.index('clean_hr')
    resampling = np.abs(cube[:, :, oracle] - cube[:, :, clean])
    report = dict(primary=primary, success=primary['estimate'] >= MIN_PP and primary['ci95'][0] > 0,
                  scenes=len(records), unique_pairs=sum(o['n_unique_pairs'] for o in objects),
                  finite_scores=int(np.isfinite(cube).sum()), results=pooled,
                  practical_ties_sr=sensitivity, sr_by_split_and_amplitude=splits, one_sr=one_sr,
                  sr_direct_vs_fixed_score_gain={m: effect(favorable[:, q], all_weights)
                                                for q, m in enumerate(METRICS)},
                  oracle_resampling_mae={f'{cond} at {a:g} px':
                                          dict(zip(METRICS, resampling[amplitude == a, c]
                                                   .mean(axis=(0, 1)).tolist()))
                                          for c, cond in enumerate(CONDITIONS) for a in sorted(set(amplitude))},
                  geometry=geometry)
    atomic_json(ROOT / 'formal/analysis.json', report)
    print(f'Primary effect in percentage points: {primary}. Pass: {report["success"]}', flush=True)


if __name__ == '__main__':
    analyze()
