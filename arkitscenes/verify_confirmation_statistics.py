"""Independently verify the frozen sample, acquisition flow, and paired SHAFE statistics."""

from argparse import ArgumentParser
import ast
from collections import Counter
import csv
from hashlib import file_digest
from itertools import combinations
import json
from math import isclose, isfinite
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(directory, name):
    return json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))


def require(condition, label):
    if not condition:
        raise ValueError(label)


def check_frozen_sources(directory, sample):
    source = directory.parent
    index_path = source / 'arkitscenes_upsampling_split.csv'
    parent_path = source / 'arkitscenes_temporal_holdout' / 'protocol.json'
    missing_path = parent_path.with_name('apple_download_data.py')
    require(digest(index_path) == sample['index_sha256'], 'Official split digest')
    require(digest(parent_path) == sample['parent_sha256'], 'Parent protocol digest')
    require(digest(missing_path) == sample['official_missing_script_sha256'], 'Official missing-asset digest')
    for path, expected in {**sample['history_sha256'], **sample['hashes']}.items():
        require(digest(source / path) == expected, f'Frozen source digest for {path}')

    with index_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    excluded = sample['excluded_visits']
    require(all('official Validation' in excluded.get(row['visit_id'], []) for row in rows
                if row['fold'] == 'Validation'), 'All Validation visits excluded')
    training = {}
    for row in sorted(rows, key=lambda row: row['video_id']):
        if row['fold'] == 'Training' and row['visit_id'] not in excluded:
            training.setdefault(row['visit_id'], row)
    require(list(training.values()) == sample['sample'], 'All remaining Training visits and lowest video IDs')

    missing_lists = [node.value for node in ast.parse(missing_path.read_text(encoding='utf-8')).body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'missing_3dod_assets_video_ids'
                             for target in node.targets)]
    require(len(missing_lists) == 1 and sorted(ast.literal_eval(missing_lists[0])) ==
            sample['official_missing_video_ids'], 'Official missing-asset roster')


def check_acquisition(directory, report, sample, plan):
    inputs = read(directory, 'inputs')
    replay = read(directory, 'acquisition_replay')
    original = read(directory, 'sampling_protocol_pre_smoke_fix')
    require(sample['pre_scoring_amendment']['original_sampling_sha256'] ==
            digest(directory / 'sampling_protocol_pre_smoke_fix.json'), 'Original protocol hash')
    require(all(original[key] == sample[key] for key in ('sample', 'endpoint', 'settings', 'expansion')),
            'Pre-scoring amendment changed the sample or design')
    frozen = {(row['visit_id'], row['video_id']) for row in sample['sample']}
    require(len(frozen) == len(sample['sample']) == len({visit for visit, _ in frozen}), 'Unique sample visits')
    require(not {visit for visit, _ in frozen} & sample['excluded_visits'].keys(), 'Historical visit overlap')
    require(frozen == {(row['visit_id'], row['video_id']) for row in plan['ranking']}, 'Pose-screened roster')
    missing = set(sample['official_missing_video_ids'])
    require(all(bool(row.get('availability')) == (row['video_id'] in missing) for row in plan['ranking']),
            'Missing official trajectories accounted for')
    expected = [(row['visit_id'], row['video_id']) for row in plan['ranking'] if row['priority_pairs']]
    require(expected == [(row['visit'], row['video']) for row in inputs['visits']], 'Frame-screened roster')
    require(not any(row.get('error') for row in inputs['visits']), 'Unresolved frame-screening error')
    for row in inputs['visits']:
        receipt = read(directory / 'screened', row['video'])
        require(receipt == row, f'Frame-screening receipt for {row["video"]}')
    selected = {row['visit']: row['video'] for row in inputs['visits'] if row['selected']}
    require(selected == {row['visit']: row['video'] for row in replay['cases']}, 'Reconstructed scoring roster')
    require(replay['screened_visits'] == len(expected)
            and all(row['verified_input_images'] == 5 for row in replay['cases']), 'Reconstruction coverage')
    if selected:
        scores = read(directory, 'scores')
        require(selected == {row['visit']: row['video'] for row in scores['visits']}, 'Scored visit roster')
        require(scores['inputs_sha256'] == report['files_sha256']['inputs'], 'Score input digest')
        for row in scores['visits']:
            data = row['scores']
            for pose, conditions in report['pair_gaps'][row['visit']]['50'].items():
                for left, right, sync, moved in conditions['moved']:
                    expected_sync = data['sync'][left]['fixed']['SHAFE'] - data['sync'][right]['fixed']['SHAFE']
                    expected_moved = (data['moved'][left]['depth_' + pose]['SHAFE'] -
                                      data['moved'][right]['depth_' + pose]['SHAFE'])
                    require(sync == expected_sync and moved == expected_moved,
                            f'Radius-50 gaps agree with saved scores for {row["video"]}')
    require(report['frozen_visits'] == len(frozen)
            and report['missing_official_pose'] == sum(bool(row.get('availability')) for row in plan['ranking'])
            and report['pose_eligible'] == len(expected)
            and report['rgb_downloaded'] == sum(bool(row.get('rgb_pairs_downloaded')) for row in inputs['visits'])
            and report['scored_visits'] == len(selected)
            and selected.keys() == report['pair_gaps'].keys(), 'Acquisition flow totals')


def check_online_replay(directory, sample):
    online = read(directory, 'online_replay')
    local = read(directory, 'acquisition_replay')
    require(online['archive'] == sample['official_archive_root']
            and online['inputs_sha256'] == local['inputs_sha256'] == digest(directory / 'inputs.json'),
            'Official source replay input hash')
    require(len(online['cases']) == len(local['cases']), 'Official source replay coverage')
    for row, archived in zip(online['cases'], local['cases'], strict=True):
        video = archived['video']
        require(all(row[key] == value for key, value in archived.items())
                and row['source_frames_verified'] == 5 and row['geometry_assets_verified'] == 3
                and row['screen_sha256'] == digest(directory / 'screened' / f'{video}.json'),
                f'Official source replay record for {video}')
        pixels = row['pixel_hashes']
        require(set(pixels) == {'lr', 'sync', 'moved', 'depth_linear', 'depth_spline'}
                and all(record['dtype'] == 'uint8' and record['shape'][-1] == 3
                        and len(record['sha256']) == 64 for record in pixels.values()),
                f'Official source replay pixel receipt for {video}')
    print(f'Verified official source replay records for {len(online["cases"])} visits.')


def count_case(rows, margin, models):
    expected = set(combinations(sorted(models), 2))
    require(len(rows) == len(expected) and {tuple(row[:2]) for row in rows} == expected, 'Six unique model pairs')
    flags = {}
    for left, right, sync, score in rows:
        require(isfinite(sync) and isfinite(score), 'Finite model gaps')
        flags[left, right] = (sync, sync * score < 0 and min(abs(sync), abs(score)) > margin)
    return flags


def check_contrast(record, windows, margin, endpoint, labels):
    require((record['first'], record['second']) == labels, 'Contrast labels')
    counts = []
    for visit, row in sorted(windows.items()):
        moved, control = (count_case(row[label], margin, endpoint['models']) for label in labels)
        require(all(moved[pair][0] == control[pair][0] for pair in moved), 'Paired synchronous gap')
        bins = Counter((moved[pair][1], control[pair][1]) for pair in moved)
        values = [bins[flags] for flags in ((False, False), (True, False), (False, True), (True, True))]
        counts.append(values)
        saved = record['units'][visit]
        require([saved[key] for key in ('neither', 'first_only', 'second_only', 'both')] == values,
                f'Paired counts for {visit}')
        require(saved['pairs'] == 6 and isclose(saved['difference'], (values[1] - values[2]) / 6),
                f'Visit effect for {visit}')
    neither, moved_only, control_only, both = np.array(counts).T
    differences = (moved_only - control_only) / 6
    require(record['units'].keys() == windows.keys()
            and record['pairs'] == 6 * len(windows)
            and record['first_reversals'] == int((moved_only + both).sum())
            and record['second_reversals'] == int((control_only + both).sum())
            and isclose(record['equal_unit_difference'], float(differences.mean()), abs_tol=1e-15),
            'Total reversals or paired point estimate')
    settings = endpoint['uncertainty']
    generator = np.random.default_rng(settings['seed'])
    draws = generator.integers(len(windows), size=(settings['draws'], len(windows)))
    interval = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])
    require(np.allclose(interval, record['bootstrap_95'], rtol=0, atol=1e-15), 'Bootstrap interval')
    leave_one_out = (differences.sum() - differences) / (len(windows) - 1)
    require(record['positive_units'] == int((differences > 0).sum())
            and record['tied_units'] == int((differences == 0).sum())
            and record['negative_units'] == int((differences < 0).sum())
            and np.allclose(record['leave_one_unit_out_range'], [leave_one_out.min(), leave_one_out.max()],
                            rtol=0, atol=1e-15), 'Visit influence statistics')
    return moved_only - control_only


def exact_probability(differences):
    total = sum(differences)
    probabilities = np.array([1], dtype=object)
    reach = 0
    for amount in differences:
        weight = abs(int(amount))
        polynomial = np.zeros(2 * weight + 1, dtype=object)
        polynomial[0] += 1
        polynomial[-1] += 1
        probabilities = np.convolve(probabilities, polynomial)
        reach += weight
    return sum(count for index, count in enumerate(probabilities) if abs(index - reach) >= abs(total)) / 2**len(differences)


def check_secondary(directory, primary, endpoint):
    report = read(directory, 'secondary_results')
    source = directory.parent
    registration = source / 'confirmation_secondary_protocol.md'
    script = source / 'check_arkitscenes_confirmation_secondary.py'
    require(digest(registration) == '07cb69c8ff159edac896dbfe807f8519e7e140309ea909bb7f67e00f797d0bac',
            'Supplementary analysis registration digest')
    require(report['confirmation_sha256'] == digest(directory / 'confirmation_results.json')
            and report['analysis_source_sha256'] == digest(script)
            and all(value in registration.read_text(encoding='utf-8') for value in
                    (primary['sampling_sha256'], report['analysis_source_sha256'])), 'Registered analysis and input digests')
    alignment = report['alignment_pair_gaps']
    references = report['resampling_pair_gaps']
    require(alignment.keys() == primary['pair_gaps'].keys()
            and references.keys() == set(primary['fully_supported_visits']), 'Supplementary analysis rosters')
    scores = read(directory, 'scores')['visits'] if alignment else []
    for row in scores:
        data = row['scores']
        for method, gaps in alignment[row['visit']]['50'].items():
            for left, right, sync, moved in gaps:
                require(sync == data['sync'][left]['fixed']['SHAFE'] - data['sync'][right]['fixed']['SHAFE']
                        and moved == data['moved'][left][method]['SHAFE'] - data['moved'][right][method]['SHAFE'],
                        f'Alignment gaps agree with saved scores for {row["visit"]}/{method}')
    settings = endpoint['sensitivity']
    for radius in map(str, settings['radii']):
        direct = {visit: row[radius] for visit, row in alignment.items()}
        for margin in settings['margins'] if len(direct) >= 2 else ():
            check_contrast(report['alignment'][radius][str(margin)], direct, margin, endpoint, ('direct', 'lk_lr'))
        for pose in settings['poses']:
            cases = {}
            for visit, row in references.items():
                control = row[pose]
                expected = primary['pair_gaps'][visit][radius][pose]
                require(control['reference_pixel_sha256']['two'] == expected['same_frame_pixel_sha256'],
                        f'Two-step reference pixel digest for {visit}/{pose}')
                gaps = control['pair_gaps'][radius]
                cases[visit] = {label: [[*value['models'], value['sync'], value[label]] for value in gaps]
                                for label in ('moved', 'one', 'two')}
                require(cases[visit]['moved'] == expected['moved']
                        and cases[visit]['two'] == expected['same_frame'], f'Frozen reference gaps for {visit}/{pose}')
            for margin in settings['margins'] if len(cases) >= 2 else ():
                comparisons = report['resampling'][radius][pose][str(margin)]
                for labels in (('moved', 'one'), ('moved', 'two'), ('two', 'one')):
                    check_contrast(comparisons['_vs_'.join(labels)], cases, margin, endpoint, labels)
    print(f'Verified supplementary statistics from {len(alignment)} alignment and {len(references)} resampling visits.')


def check_metrics(directory, primary, endpoint):
    report = read(directory, 'confirmation_metrics')
    registration = directory.parent / 'confirmation_metrics_protocol.md'
    source = directory.parent / 'check_arkitscenes_confirmation_metrics.py'
    require(digest(registration) == '4079f0574e3695b5041fb70bb0ce76e61fc3e2158c3f71797ed1f592bac08c40',
            'Cross-metric registration digest')
    require(report['primary_sha256'] == digest(directory / 'confirmation_results.json')
            and report['source_sha256'] == digest(source)
            and all(value in registration.read_text(encoding='utf-8') for value in
                    (primary['sampling_sha256'], report['source_sha256'])), 'Cross-metric analysis and input digests')
    controls = report['controls']
    require(controls.keys() == set(primary['fully_supported_visits']), 'Cross-metric common support roster')
    scores = {row['visit']: row['scores'] for row in read(directory, 'scores')['visits']} if controls else {}
    pairs = list(combinations(sorted(endpoint['models']), 2))
    margins = {'PSNR': [0., 0.05], 'SSIM': [0., 0.0005], 'SHAFE': [0., 0.0005], 'ST-LPIPS': [0., 0.0005]}
    require(report['margins'] == margins, 'Cross-metric margins')
    for pose in endpoint['sensitivity']['poses']:
        for visit, row in controls.items():
            require(row[pose]['pixel_sha256'] == primary['pair_gaps'][visit]['50'][pose]['same_frame_pixel_sha256'],
                    f'Cross-metric reference pixels for {visit}/{pose}')
        for metric, thresholds in margins.items():
            cases = {}
            for visit, row in controls.items():
                sync, moved = (scores[visit][condition] for condition in ('sync', 'moved'))
                targets = {
                    'moved': {model: values['depth_' + pose] for model, values in moved.items()},
                    'same_frame': row[pose]['model_scores'],
                }
                cases[visit] = {label: [[a, b, sync[a]['fixed'][metric] - sync[b]['fixed'][metric],
                                        models[a][metric] - models[b][metric]] for a, b in pairs]
                                for label, models in targets.items()}
            for margin in thresholds if len(cases) >= 2 else ():
                check_contrast(report['results'][pose][metric][str(margin)], cases, margin, endpoint,
                               ('moved', 'same_frame'))
            if metric == 'SHAFE' and len(cases) >= 2:
                require(report['results'][pose][metric]['0.0005'] == primary['results']['0.0005']['50'][pose],
                        f'Cross-metric SHAFE agrees with main statistics for {pose}')
    print(f'Verified cross-metric contrasts for {len(controls)} visits.')


def main(directory, verify_secondary=False, verify_metrics=False, verify_online=False):
    report = read(directory, 'confirmation_results')
    sample = read(directory, 'sampling_protocol')
    plan = read(directory, 'protocol')
    require(report['sampling_sha256'] == plan['sampling_protocol_sha256'] ==
            digest(directory / 'sampling_protocol.json'), 'Frozen sampling digest')
    check_frozen_sources(directory, sample)
    for name, expected in report['files_sha256'].items():
        require(digest(directory / f'{name}.json') == expected, f'File digest for {name}')
    check_acquisition(directory, report, sample, plan)
    if verify_online:
        check_online_replay(directory, sample)
    endpoint = sample['endpoint']
    require(endpoint == plan['endpoint'] == report['endpoint'], 'Frozen endpoint')
    sensitivities = endpoint['sensitivity']
    radii = [str(radius) for radius in sensitivities['radii']]
    poses = sensitivities['poses']
    windows = {}
    for visit, row in report['pair_gaps'].items():
        require(set(row) == set(radii), 'Scoring radii')
        if row[radii[0]]:
            require(all(set(row[radius]) == set(poses) for radius in radii), 'Common pose support')
            windows[visit] = row
        else:
            require(not any(row.values()), 'Common support inconsistent across radii')
    require(len(windows) == report['common_support_visits'] and set(windows) == set(report['fully_supported_visits']),
            'Final common support count')
    if len(windows) < 2:
        require(not report['results'] and report['primary_exact_sign_flip_p'] is None
                and not report['primary_passed'], 'Endpoint with insufficient common support')
        print('Verified acquisition. Insufficient common support for inference.')
        if verify_secondary:
            check_secondary(directory, report, endpoint)
        if verify_metrics:
            check_metrics(directory, report, endpoint)
        return
    primary, probability = endpoint['primary'], None
    for margin in sensitivities['margins']:
        for radius in radii:
            for pose in poses:
                contrast = report['results'][str(margin)][radius][pose]
                cases = {visit: row[radius][pose] for visit, row in windows.items()}
                differences = check_contrast(contrast, cases, margin, endpoint, ('moved', 'same_frame'))
                if (margin, radius, pose) == (primary['practical_tie'], str(primary['radius']), primary['pose']):
                    probability = exact_probability(differences)
                    require(isclose(probability, report['primary_exact_sign_flip_p'], abs_tol=1e-15)
                            and report['primary_passed'] == (contrast['bootstrap_95'][0] > 0 and probability < 0.05),
                            'Primary sign-flip and success criterion')
    require(probability is not None, 'Primary result missing')
    print(f'Verified {len(windows)} paired visits, all sensitivity contrasts, and exact p={probability:.9g}.')
    if verify_secondary:
        check_secondary(directory, report, endpoint)
    if verify_metrics:
        check_metrics(directory, report, endpoint)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--secondary', action='store_true', help='Recompute the registered supplementary statistics')
    parser.add_argument('--metrics', action='store_true', help='Recompute the registered cross-metric statistics')
    parser.add_argument('--online', action='store_true', help='Verify the fresh official source replay records')
    args = parser.parse_args()
    main(args.directory, args.secondary, args.metrics, args.online)
