"""Independently recalculate the frozen holdout endpoints from the saved measurements."""

import ast
from collections import Counter
import csv
from hashlib import file_digest
from itertools import combinations
import json
from math import isclose, isfinite
from pathlib import Path
from random import Random

import numpy as np


ROOT = Path(__file__).resolve().parent
HOLDOUT = ROOT / 'arkitscenes_temporal_holdout'
PROTOCOL_SHA = '8466eaf330a1ef8c8c344eaaa8564b03183635fc4a69327e66fe86a0ecf7678d'
MODELS = ('Bicubic', 'SwinIR-S', 'Real-ESRGAN', 'VOSR')
METHODS = ('fixed', 'direct', 'global_lr', 'lk_lr', 'depth_linear', 'depth_spline')
MARGIN, DRAWS, SEED = 0.0005, 10000, 20261122
METRIC_MARGINS = {'PSNR': (0., 0.05), 'SSIM': (0., 0.0005), 'ST-LPIPS': (0., 0.0005)}
APPLE_RAW = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training'


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def verify_roster(directory):
    plan_path = directory / 'protocol.json'
    sample_path = directory / 'sampling_protocol.json'
    plan, sample = load(plan_path), load(sample_path)
    index = directory / 'apple_official_split.csv'
    previous_draw = ROOT / 'arkitscenes_temporal_training' / 'sampling_protocol.json'
    if (sha(plan_path) != PROTOCOL_SHA or sha(sample_path) != plan['sampling_protocol_sha256']
            or sha(index) != sample['index_sha256'] or sample['draw'] != 240
            or sample['sample_seed'] != 20261124):
        raise ValueError('Frozen visit files or sampling settings changed')
    excluded = {row['visit_id'] for row in load(previous_draw)['candidates']}
    for label, expected in sample['earlier_input_sha256'].items():
        path = ROOT / label / 'inputs.json'
        if sha(path) != expected:
            raise ValueError(f'Prior visit manifest changed: {label}')
        excluded.update(row['visit'] for row in load(path)['visits'])
    with index.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    excluded.update(row['visit_id'] for row in rows if row['fold'] == 'Validation')
    available = {}
    for row in sorted(rows, key=lambda item: item['video_id']):
        if row['fold'] == 'Training' and row['visit_id'] not in excluded:
            available.setdefault(row['visit_id'], row)
    drawn = [available[visit] for visit in Random(20261124).sample(sorted(available), 240)]
    membership = {(row['visit_id'], row['video_id']) for row in drawn}
    ranked = {(row['visit_id'], row['video_id']) for row in plan['ranking']}
    candidate_rows = [row for row in plan['ranking'] if row['priority_pairs'] > 0]
    if (len(available) != 554 or drawn != sample['sample'] or len(ranked) != 240
            or len(plan['ranking']) != 240 or membership != ranked
            or candidate_rows != plan['candidates'] or len(candidate_rows) != 194):
        raise ValueError('Could not reconstruct the official disjoint draw and candidate list')
    return plan


def check_acquisition_flow(directory, plan, inputs):
    downloader = directory / 'apple_download_data.py'
    assignment, = (node for node in ast.parse(downloader.read_text(encoding='utf-8')).body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == 'missing_3dod_assets_video_ids'
                       for target in node.targets))
    missing = set(ast.literal_eval(assignment.value))
    unavailable = [row for row in plan['ranking'] if row.get('availability')]
    if ({row['video_id'] for row in unavailable} != {row['video_id'] for row in plan['ranking']} & missing
            or any(row['official_roster_sha256'] != sha(downloader) for row in unavailable)):
        raise ValueError('Trajectory exclusions differ from the Apple download script')
    flow = Counter()
    frames = {}
    ranges = Counter()
    for video, row in inputs.items():
        candidates = row['candidates']
        windows = [window for candidate in candidates for window in candidate['rgb_windows']]
        flow['depth_pairs_examined'] += len(candidates)
        flow['rgb_pairs_examined'] += sum('rgb_source' in pair for pair in candidates)
        flow['visits_with_depth_windows'] += any(pair['depth_counts']['passed'] for pair in candidates)
        flow['visits_with_rgb_downloads'] += any('rgb_source' in pair for pair in candidates)
        flow['visits_passing_sync'] += any(window['sync']['pass'] for window in windows)
        flow['visits_passing_rgb_motion'] += any('optical_check' in window for window in windows)
        flow['visits_passing_optical_agreement'] += any('appearance' in window for window in windows)
        flow['visits_passing_all_checks'] += any(window.get('passed', False) for window in windows)
        if (any(window.get('passed', False) for window in windows) != bool(row['selected'])
                or sum('rgb_source' in pair for pair in candidates) != row.get('rgb_pairs_downloaded', 0)):
            raise ValueError(f'Inconsistent visit screening history: {video}')
        for pair in candidates:
            sources = {'highres_depth': pair['depth_source']}
            if 'rgb_source' in pair:
                sources.update(wide=pair['rgb_source']['wide'], lowres_wide=pair['rgb_source']['low'])
            times = [pair['anchor_time'], pair['shifted_time']]
            for asset, source in sources.items():
                url = f'{APPLE_RAW}/{video}/{asset}.zip'
                if (source['url'] != url or not source['etag'] or len(source['zip_tail_sha256']) != 64
                        or [frame['timestamp'] for frame in source['frames']]
                        != (times[:1] if asset == 'lowres_wide' else times)):
                    raise ValueError(f'Unexpected frame source or timestamps: {video}/{asset}')
                for frame in source['frames']:
                    identity = (url, frame['member'])
                    receipt = (source['etag'], source['zip_tail_sha256'], frame)
                    if identity in frames:
                        if frames[identity] != receipt:
                            raise ValueError(f'Conflicting ZIP member receipts: {identity}')
                        continue
                    start, stop = frame['range_start'], frame['range_stop_inclusive']
                    if (not 0 <= start <= stop < source.get('total_zip_bytes', float('inf'))
                            or not frame['member'].startswith(f'{asset}/{video}_')
                            or len(frame['crc32']) != 8 or len(frame['sha256']) != 64):
                        raise ValueError(f'Invalid frame receipt: {identity}')
                    frames[identity] = receipt
                    ranges[asset] += stop - start + 1
    return dict(flow, official_trajectory_missing=len(unavailable), archived_zip_members=len(frames),
                distinct_member_range_bytes=dict(ranges))


def paired_interval(values, draws):
    return np.percentile(np.asarray(values)[draws].mean(axis=1), [2.5, 97.5]).tolist()


def strict_reversals(scores, method, metric='SHAFE', margin=MARGIN, condition='moved', baseline='fixed'):
    counts = Counter({key: 0 for key in ('reversed', 'same', 'sync_tied', 'moved_tied')})
    for a, b in combinations(MODELS, 2):
        sync = scores['sync'][a][baseline][metric] - scores['sync'][b][baseline][metric]
        moved = scores[condition][a][method][metric] - scores[condition][b][method][metric]
        if not isfinite(sync) or not isfinite(moved):
            raise ValueError(f'Non-finite {metric} score difference')
        if abs(sync) <= margin:
            counts['sync_tied'] += 1
        elif abs(moved) <= margin:
            counts['moved_tied'] += 1
        else:
            counts['reversed' if (sync < 0) != (moved < 0) else 'same'] += 1
    return dict(counts)


def check_numbers(observed, expected, label):
    if len(observed) != len(expected) or not all(
            isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(observed, expected, strict=True)):
        raise ValueError(f'Different independent and original values for {label}: {observed} vs {expected}')


def unique_videos(rows):
    videos = {row['video']: row for row in rows}
    if len(videos) != len(rows) or len({row['visit'] for row in rows}) != len(rows):
        raise ValueError('Repeated video or visit identifier')
    return videos


def verify_sensitivity(rows, metric, margin, draws, published):
    counts = np.array([[strict_reversals(scores, method, metric, margin)['reversed']
                        for method in ('direct', 'lk_lr')] for scores in rows])
    direct, lk = counts.sum(axis=0)
    differences = (counts[:, 0] - counts[:, 1]) / 6
    pairs = len(rows) * 6
    difference = float(differences.mean())
    interval = paired_interval(differences, draws)
    check_numbers([direct, lk, pairs, difference, *interval],
                  [published['first_reversals'], published['second_reversals'], published['pairs'],
                   published['difference'], *published['bootstrap_95']], f'{metric} at margin {margin}')
    return dict(direct=int(direct), lk=int(lk), pairs=pairs, difference=difference, bootstrap_95=interval)


def reference_swap_controls(rows, draws):
    moved = np.array([[strict_reversals(row, method)['reversed'] for method in ('direct', 'lk_lr')]
                      for row in rows])
    controls = {
        'synchronous': np.array([[strict_reversals(row, method, condition='sync')['reversed']
                                  for method in ('direct', 'lk_lr')] for row in rows]),
        'own_synchronous_baseline': np.array([[strict_reversals(row, method, baseline=method)['reversed']
                                               for method in ('direct', 'lk_lr')] for row in rows]),
    }
    result = {}
    for name, counts in controls.items():
        delta = (counts[:, 0] - counts[:, 1]) / 6
        direct, lk = counts.sum(axis=0)
        result[name] = dict(direct=int(direct), lk=int(lk), difference=float(delta.mean()),
                            bootstrap_95=paired_interval(delta, draws))
    adjusted = ((moved - controls['synchronous']) @ np.array([1, -1])) / 6
    result['change_from_synchronous'] = dict(difference=float(adjusted.mean()),
                                             bootstrap_95=paired_interval(adjusted, draws))
    return result


def run(directory=HOLDOUT):
    plan = verify_roster(directory)
    summary_path = directory / 'holdout_endpoint_summary.json'
    summary = load(summary_path)
    evidence = {}
    for label in ('inputs', 'scores', 'ranking_reanalysis', 'acquisition_replay',
                  'reference_control_replay_v2', 'normalized_cutoff_scores'):
        path = directory / f'{label}.json'
        if sha(path) != summary['artifact_sha256'][label]:
            raise ValueError(f'Endpoint provenance changed for {label}')
        evidence[label] = load(path)
    inputs = unique_videos(evidence['inputs']['visits'])
    scores = unique_videos(evidence['scores']['visits'])
    acquisition = unique_videos(evidence['acquisition_replay']['cases'])
    controls = unique_videos(evidence['reference_control_replay_v2']['visits'])
    if ([(row['video_id'], row['visit_id']) for row in plan['candidates']]
            != [(video, row['visit']) for video, row in inputs.items()] or any(row.get('error') for row in inputs.values())):
        raise ValueError('Screened cohort differs from the frozen candidates')
    flow = check_acquisition_flow(directory, plan, inputs)
    selected = {video: row['visit'] for video, row in inputs.items() if row['selected']}
    if not selected or any({video: row['visit'] for video, row in rows.items()} != selected
                           for rows in (scores, acquisition, controls)):
        raise ValueError('Selected cohort differs from the raw measurements')

    visits = []
    for video, row in scores.items():
        if row != load(directory / 'cases' / f'{video}.json'):
            raise ValueError(f'Per-visit score differs from its aggregate: {video}')
        visits.append(dict(visit=selected[video], video=video, lag_ms=acquisition[video]['lag_ms'],
                           transitions={method: strict_reversals(row['scores'], method) for method in METHODS},
                           p90_direct_minus_lk_px={
                               pose: sum(geometry['translation_residual_px']['direct'][model]['p90']
                                         - geometry['translation_residual_px']['lk_lr'][model]['p90']
                                         for model in MODELS) / len(MODELS)
                               for pose, geometry in controls[video]['poses'].items()}))
    draw_indices = np.random.default_rng(SEED).integers(len(visits), size=(DRAWS, len(visits)))
    pair_total = len(visits) * 6
    transitions = {}
    for method in METHODS:
        by_visit = [row['transitions'][method]['reversed'] for row in visits]
        totals = Counter()
        for row in visits:
            totals.update(row['transitions'][method])
        transitions[method] = dict(totals)
        published = summary['primary']['methods'][method]
        check_numbers([totals['reversed'], totals['reversed'] + totals['same'], pair_total,
                       *paired_interval(np.asarray(by_visit) / 6, draw_indices)],
                      [published['reversals'], published['strict_pairs'], published['denominator'],
                       *published['bootstrap_95']], method)
    deltas = np.array([row['transitions']['direct']['reversed'] - row['transitions']['lk_lr']['reversed']
                       for row in visits]) / 6
    contrast = dict(fraction=float(deltas.mean()), bootstrap_95=paired_interval(deltas, draw_indices))
    published = summary['primary']['paired_minus_lk']['direct']
    check_numbers([contrast['fraction'], *contrast['bootstrap_95']],
                  [published['delta_fraction'], *published['bootstrap_95']], 'primary direct minus LK')
    normalized = evidence['normalized_cutoff_scores']
    if (normalized['scores_sha256'] != summary['artifact_sha256']['scores']
            or normalized['radius'] != plan['endpoint']['sensitivity']['cutoff_radius']
            or set(normalized['visits']) != set(selected.values())):
        raise ValueError('Normalized cutoff scores disagree with the frozen analysis')
    normalized_rows = [normalized['visits'][row['visit']][0]['scores'] for row in visits]
    sensitivity = dict(normalized_cutoff=verify_sensitivity(
        normalized_rows, 'SHAFE', MARGIN, draw_indices, summary['normalized_cutoff']))
    base_rows = [row['scores'] for row in scores.values()]
    if len(deltas) > 1:
        leave_one_out = (deltas.sum() - deltas) / (len(deltas) - 1)
        contrast['leave_one_visit_out'] = [float(leave_one_out.min()), float(leave_one_out.max())]
    for metric, margins in METRIC_MARGINS.items():
        sensitivity[metric] = {str(margin): verify_sensitivity(
            base_rows, metric, margin, draw_indices, summary['additional_metrics'][metric][str(margin)])
            for margin in margins}
    residuals = {}
    for pose in ('linear', 'spline'):
        values = np.asarray([row['p90_direct_minus_lk_px'][pose] for row in visits])
        result = dict(mean=float(values.mean()), bootstrap_95=paired_interval(values, draw_indices))
        published = summary['residuals'][pose]['p90']['paired_comparator_minus_lk']['direct']
        check_numbers([result['mean'], *result['bootstrap_95']],
                      [published['paired_mean_hr_px'], *published['bootstrap_95_hr_px']], f'{pose} residual')
        residuals[pose] = result

    result = dict(protocol_sha256=PROTOCOL_SHA, endpoint_sha256=sha(summary_path),
                  audit_sha256=sha(Path(__file__)), screened_visits=len(inputs), scored_visits=len(visits),
                  acquisition_flow=flow, pairs=pair_total, transitions=transitions, rank_difference=contrast,
                  reference_swap_controls=reference_swap_controls(base_rows, draw_indices),
                  p90_direct_minus_lk_px=residuals, sensitivity=sensitivity, visits=visits)
    target = directory / 'independent_endpoint_recalculation.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)
    print(f'{len(inputs)} screened; {len(visits)} scored; {pair_total} pairs independently audited')


if __name__ == '__main__':
    run()
