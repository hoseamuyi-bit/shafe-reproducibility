"""Audit model-pair identity and paired temporal controls on archived scores."""

from argparse import ArgumentParser
from itertools import combinations
import json
from math import isfinite
from pathlib import Path

import numpy as np

from audit_arkitscenes_holdout_endpoints import DRAWS, MODELS, SEED
from audit_arkitscenes_reference_offsets import (
    HOLDOUT, MARGIN, OFFSETS, POSES, RADII, REPLAY_ATOL, ROOT, ZERO, frozen_scores, load,
    model_scores, pair_rows, sha, verify_and_summarize,
)


TUM = ROOT / 'tum_rgbd'
GAP_MARGINS = (MARGIN, 0.001, 0.002, 0.005)
MODEL_PAIRS = set(combinations(sorted(MODELS), 2))
OUTPUT = ROOT / 'temporal_reference_control_comparison.json'


def reversed_pair(sync_gap, test_gap, margin):
    return min(abs(sync_gap), abs(test_gap)) > margin and (sync_gap < 0) != (test_gap < 0)


def gaps(ranking):
    rows = {}
    for row in ranking['pairs']:
        left, right = row['models']
        pair = tuple(sorted((left, right)))
        sign = 1 if pair == (left, right) else -1
        prior, test = row['sync_gap'] * sign, row['control_gap'] * sign
        if (pair in rows or not (isfinite(prior) and isfinite(test))
                or reversed_pair(prior, test, MARGIN) != row['reversal']):
            raise ValueError('Invalid archived model-pair comparison')
        rows[pair] = prior, test
    if rows.keys() != MODEL_PAIRS or sum(
            reversed_pair(*row, MARGIN) for row in rows.values()) != ranking['reversals']:
        raise ValueError('Ranking must include all six model pairs')
    return rows


def score_gaps(sync, test):
    if set(sync) != set(test) or set(sync) != set(MODELS):
        raise ValueError('Incomplete score comparison')
    result = {(a, b): (sync[a] - sync[b], test[a] - test[b]) for a, b in sorted(MODEL_PAIRS)}
    if not all(isfinite(value) for pair in result.values() for value in pair):
        raise ValueError('Non-finite score')
    return result


def persistent(pair_offsets, margin):
    return {pair for pair in MODEL_PAIRS
            if all(reversed_pair(*row[pair], margin) for row in pair_offsets)}


def paired_summary(windows, first, second, margin, bootstrap):
    units = {}
    for unit, _, a, b in windows:
        if a is None or b is None:
            continue
        counts = units.setdefault(unit, [0, 0, 0, 0])
        for pair in MODEL_PAIRS:
            prior, x = a[pair]
            other_prior, y = b[pair]
            if prior != other_prior:
                raise ValueError('Paired references must use the same synchronous baseline')
            counts[2 * reversed_pair(prior, x, margin) + reversed_pair(prior, y, margin)] += 1
    if len(units) < 2:
        raise ValueError('Paired analysis requires multiple independent units')
    by_unit, rates = {}, []
    for unit, (neither, second_only, first_only, both) in sorted(units.items()):
        n = neither + second_only + first_only + both
        rates.append((first_only - second_only) / n)
        by_unit[unit] = dict(pairs=n, neither=neither, first_only=first_only, second_only=second_only,
                             both=both, difference=rates[-1])
    rates = np.array(rates)
    removed = (rates.sum() - rates) / (len(rates) - 1)
    interval = None
    if bootstrap:
        index = np.random.default_rng(SEED).integers(len(rates), size=(DRAWS, len(rates)))
        interval = np.quantile(rates[index].mean(axis=1), [0.025, 0.975]).tolist()
    return dict(first=first, second=second, pairs=sum(r['pairs'] for r in by_unit.values()),
                first_reversals=sum(r['first_only'] + r['both'] for r in by_unit.values()),
                second_reversals=sum(r['second_only'] + r['both'] for r in by_unit.values()),
                equal_unit_difference=float(rates.mean()), bootstrap_95=interval,
                positive_units=int((rates > 0).sum()), tied_units=int((rates == 0).sum()),
                negative_units=int((rates < 0).sum()),
                leave_one_unit_out_range=[float(removed.min()), float(removed.max())], units=by_unit)


def summarize_intersection(sets):
    shared = set.intersection(*sets)
    return dict(count=len(shared), units=len({unit for unit, _, _ in shared}),
                members=[dict(unit=unit, window=window, models=list(pair))
                         for unit, window, pair in sorted(shared)])


def summarize(dataset, rows, radii, offset_radii):
    bootstrap = dataset == 'Apple ARKitScenes'
    results = {}
    for margin in GAP_MARGINS:
        contrasts, sets = {}, {}
        for radius in radii:
            contrasts[radius] = {}
            for pose in POSES:
                pairs = [(unit, window, record[radius][pose]['moved'], record[radius][pose]['same_frame'])
                         for unit, window, record in rows]
                controls = paired_summary(pairs, 'moved', 'same_frame', margin, bootstrap)
                common = [(unit, window, a, b) for (unit, window, a, b), (_, _, record)
                          in zip(pairs, rows, strict=True)
                          if all(record[radius][name]['same_frame'] is not None for name in POSES)]
                contrasts[radius][pose] = dict(pose_specific_window_support=controls,
                                              common_to_both_poses=paired_summary(
                                                  common, 'moved', 'same_frame', margin, bootstrap))
                if radius in offset_radii:
                    sets[radius, pose] = {(unit, window, pair) for unit, window, record in rows
                                          for pair in persistent(record[radius][pose]['offsets'], margin)}
            pairs = [(unit, window, record[radius]['direct'], record[radius]['lk_lr'])
                     for unit, window, record in rows]
            contrasts[radius]['direct_vs_lk'] = paired_summary(pairs, 'direct', 'lk_lr', margin, bootstrap)
        intersections = {radius: dict(
            counts_by_pose={pose: len(sets[radius, pose]) for pose in POSES},
            both_poses=summarize_intersection([sets[radius, pose] for pose in POSES]))
            for radius in offset_radii}
        intersections['all_tested'] = summarize_intersection(list(sets.values()))
        results[str(margin)] = dict(contrasts=contrasts, offset_intersections=intersections)
    return results


def apple_rows():
    paths = [HOLDOUT / name for name in ('reference_offset_sensitivity.json', 'roundtrip_cutoff_scores.json')]
    offset, roundtrip = map(load, paths)
    hashes, paper, normalized, selected, archived = frozen_scores()
    receipt = ROOT / 'holdout_online_replay_fresh' / 'verification.json'
    if (offset['frozen_sha256'] != hashes or roundtrip['frozen_sha256'] != hashes
            or offset['independent_receipts_sha256'] != sha(receipt)
            or roundtrip['independent_receipts_sha256'] != sha(receipt)
            or roundtrip['script_sha256'] != sha(ROOT / 'score_arkitscenes_holdout_roundtrip.py')
            or offset['offsets_hr_xy'] != [list(point) for point in OFFSETS]
            or offset['summary'] != verify_and_summarize(offset['visits'], paper, normalized, selected, archived)):
        raise ValueError('Apple replays differ from the frozen measurements')
    roundtrips = {row['visit']: row for row in roundtrip['visits']}
    if roundtrips.keys() != paper.keys() or len(roundtrips) != len(roundtrip['visits']):
        raise ValueError('Apple same-frame replay is incomplete')
    rows = []
    for row in offset['visits']:
        visit, window = row['visit'], row['video']
        if roundtrips[visit]['video'] != window:
            raise ValueError(f'Same-frame replay has a different window: {visit}')
        record = {}
        for radius, old in zip(RADII, (paper[visit]['scores'], normalized[visit][0]['scores']), strict=True):
            sync = model_scores(old, 'sync', 'fixed')
            record[radius] = {method: score_gaps(sync, model_scores(old, 'moved', method))
                              for method in ('direct', 'lk_lr')}
            for pose in POSES:
                offsets = [score_gaps(sync, s) for s in row['poses'][pose]['scores'][radius]]
                moved = score_gaps(sync, model_scores(old, 'moved', 'depth_' + pose))
                replay = roundtrips[visit]['poses'][pose]
                control = archived[visit][pose]['same_frame']
                if (control is None) != (replay is None):
                    raise ValueError(f'Same-frame replay support changed: {visit}/{pose}')
                same = None
                if control is not None:
                    paper_control = gaps(control['ranking'])
                    reproduced = score_gaps(model_scores(paper[visit]['scores'], 'sync', 'fixed'),
                                            replay['scores']['paper'])
                    error = max(abs(paper_control[pair][1] - reproduced[pair][1]) for pair in MODEL_PAIRS)
                    if (paper_control.keys() != reproduced.keys() or error > 1e-6
                            or error != replay['max_paper_gap_difference']):
                        raise ValueError(f'Same-frame replay differs: {visit}/{pose}')
                    same = paper_control if radius == 'paper' else score_gaps(sync, replay['scores'][radius])
                if radius == 'paper' and moved != gaps(archived[visit][pose]['moved_rank']):
                    raise ValueError(f'Moved-frame ranking differs: {visit}/{pose}')
                record[radius][pose] = dict(moved=moved, same_frame=same, offsets=offsets)
        rows.append((visit, window, record))
    return paths + [receipt], rows


def tum_rows():
    paths = [TUM / 'static_full_control_recheck.json', TUM / 'sequence_endpoint_comparison.json',
             ROOT / 'temporal_shafe_radius_sensitivity.json', TUM / 'normalized_reference_offsets.json']
    frozen, endpoint, normalized, offset_archive = map(load, paths)
    digest = sha(paths[0])
    if (endpoint['verification_sha256'] != digest
            or normalized['input_sha256']['static_full_control_verified.json'] != digest
            or offset_archive['verification_sha256'] != digest
            or offset_archive['normalized_scores_sha256'] != sha(paths[2])
            or offset_archive['radius'] != normalized['radius']['matched']
            or offset_archive['practical_tie'] != MARGIN
            or offset_archive['offsets_hr_xy'] != [list(point) for point in OFFSETS]):
        raise ValueError('TUM evaluations use different window sets')
    windows = normalized['cases']['tum']
    expected = {name for name, item in frozen['sequences'].items() if item['eligible_windows']}
    if not (windows.keys() == offset_archive['sequences'].keys() == expected):
        raise ValueError('Incomplete TUM cutoff re-evaluation')
    rows = []
    for unit, variants in windows.items():
        frozen_unit = frozen['sequences'][unit]
        normal_path = TUM / unit / 'pilot_scores_full_controls.json'
        path = normal_path if normal_path.is_file() else TUM / f'{unit}_scored' / normal_path.name
        paths.append(path)
        if sha(path) != frozen_unit['score_sha256']:
            raise ValueError(f'TUM scores changed: {unit}')
        old_cases = load(path)['cases']
        residuals = frozen_unit['pixel_replay']['residuals']
        offset_unit = offset_archive['sequences'][unit]
        if (any(offset_unit[key] != frozen_unit[key] for key in ('geometry_sha256', 'score_sha256', 'source'))
                or [case['video'] for case in old_cases] != [row['window'] for row in offset_unit['windows']]
                or [case['video'] for case in old_cases] != [row['window'] for row in endpoint['sequences'][unit]['windows']]
                or len(old_cases) != len(residuals) or len(old_cases) != frozen_unit['eligible_windows']):
            raise ValueError(f'TUM window order or support changed: {unit}')
        for old, new, replayed in zip(old_cases, variants, offset_unit['windows'], strict=True):
            window = old['video']
            paper = {method: gaps(old['rank'][method]) for method in ('direct', 'lk_lr')}
            sync = model_scores(new['scores'], 'sync', 'fixed')
            variant = {method: score_gaps(sync, model_scores(new['scores'], 'moved', method))
                       for method in ('direct', 'lk_lr')}
            for pose in POSES:
                replay = residuals[window][pose]
                if [r['offset_xy'] for r in replay['offset_rankings']] != [list(point) for point in OFFSETS]:
                    raise ValueError(f'TUM offset lattice differs: {unit}/{window}/{pose}')
                offsets = [gaps(r) for r in replay['offset_rankings']]
                moved = gaps(old['rank']['depth_' + pose])
                expected_stable = {tuple(sorted(pair)) for pair in replay['stable_reversals']}
                if (offsets[ZERO] != moved or any(
                            value[0] != moved[key][0] for points in offsets for key, value in points.items())
                        or persistent(offsets, MARGIN) != expected_stable):
                    raise ValueError(f'TUM offset rankings differ: {unit}/{window}/{pose}')
                paper[pose] = dict(moved=moved, same_frame=gaps(old['same_frame'][pose]), offsets=offsets)
                control = {m: s['SHAFE'] for m, s in new['controls']['same_frame_' + pose].items()}
                baseline = model_scores(new['scores'], 'moved', 'depth_' + pose)
                offset_pose = replayed['poses'][pose]
                readings = offset_pose['scores']
                rankings = pair_rows(sync, readings)
                if (len(readings) != len(OFFSETS) or rankings != offset_pose['rankings']
                        or max(abs(readings[ZERO][m] - baseline[m]) for m in MODELS) > REPLAY_ATOL
                        or [r['offset_reversals'][ZERO] for r in rankings] != [
                            r['offset_reversals'][0] for r in pair_rows(sync, [baseline])]):
                    raise ValueError(f'TUM normalized offset rankings differ: {unit}/{window}/{pose}')
                variant[pose] = dict(moved=score_gaps(sync, baseline), same_frame=score_gaps(sync, control),
                                     offsets=[score_gaps(sync, scores) for scores in readings])
            rows.append((unit, window, dict(paper=paper, normalized=variant)))
    return paths, rows


def run():
    apple_inputs, apple = apple_rows()
    tum_inputs, tum = tum_rows()
    result = dict(
        scope='Post hoc analysis. Apple units are visits, TUM units are sequences.',
        script_sha256=sha(Path(__file__)),
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in apple_inputs + tum_inputs},
        bootstrap=dict(seed=SEED, draws=DRAWS),
        datasets=dict(apple=summarize('Apple ARKitScenes', apple, RADII, RADII),
                      tum=summarize('TUM RGB-D', tum, RADII, RADII)))
    existing = load(ROOT / 'temporal_shafe_radius_sensitivity.json')['results']['tum'][str(MARGIN)]
    actual = result['datasets']['tum'][str(MARGIN)]['contrasts']['normalized']
    for key, contrast in [('direct - lk_lr', actual['direct_vs_lk']),
                          *((f'depth_{p} - same_frame_{p}', actual[p]['pose_specific_window_support'])
                            for p in POSES)]:
        if any(contrast[k] != existing[key][k] for k in ('pairs', 'first_reversals', 'second_reversals')):
            raise ValueError('TUM cutoff re-analysis differs from the archived counts')
    return result


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify the stored statistics without writing files.')
    result = run()
    if parser.parse_args().verify:
        if load(OUTPUT) != result:
            raise ValueError('Stored comparison differs from offline recalculation')
    else:
        temporary = OUTPUT.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(OUTPUT)
    for name, dataset in result['datasets'].items():
        print(name, dataset[str(MARGIN)]['offset_intersections'], flush=True)
