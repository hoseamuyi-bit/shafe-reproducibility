"""Verify the completed frozen image replay and its linkage to the portable CSVs."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


COHORTS = {'formal': (150, 300, 2400), 'external': (32, 32, 128)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Hash changed: {path}')
    return path


def case_key(row):
    return tuple(row[name] for name in ('scene', 'degradation', 'condition'))


def by_case(rows):
    result = {case_key(row): row for row in rows}
    require(len(result) == len(rows), 'Duplicate case')
    return result


def check_csv(path, geometry, decisions, scores):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = by_case(list(csv.DictReader(stream)))
    require(rows.keys() == geometry.keys() == decisions.keys() == scores.keys(), f'Case sets differ: {path}')
    pairs = {}
    for key, row in rows.items():
        measured, decision, score = geometry[key], decisions[key], scores[key]
        require(all(decision[k] == value for k, value in measured.items()), f'{key}: geometry differs')
        hashes = ('lr_pixel_sha256', 'hr_pixel_sha256', 'sr_pixel_sha256')
        require(all(row[name] == measured[name] for name in hashes), f'{key}: pixel hashes differ')
        require(score['actions'] == decision['actions'], f'{key}: actions differ')
        for method, vector in decision['actions'].items():
            require(vector == [float(row[f'{method}_{axis}']) for axis in ('dx', 'dy')],
                    f'{key} {method}: CSV actions differ')
        for method, value in score['scores'].items():
            require(value == float(row[f'{method}_shafe']), f'{key} {method}: CSV scores differ')
        pairs.setdefault(f'{key[0]}__{key[1]}', set()).update(score['score_pair_ids'].values())
    return pairs


def full_source_checks(audit, lock, provenance, pilot):
    full = read(audit / 'full_vosr_replay.json')
    require(bool(full.get('completed_utc')), 'Full VOSR replay has not completed')
    recorded = full['provenance']
    checked(audit / 'replay_all_vosr.py', recorded['script_sha256'])
    require(recorded['staging_sha256'] == pilot['runner_sha256'], 'VOSR staging code differs')
    require(recorded['lock_sha256'] == provenance['formal_lock_sha256'], 'VOSR replay lock differs')
    require(recorded['inputs'] == provenance['inputs'], 'VOSR replay inputs differ')
    require(recorded['runner_sha256'] == pilot['vosr_runner_sha256'], 'VOSR runner differs')
    require(recorded['weights'] == pilot['vosr_weights'], 'VOSR replay weights differ')
    require(recorded['args'] == lock['vosr_args'], 'VOSR replay command differs')

    expected_batches = {f'{split}_{start:03d}': (split, min(100, n - start))
                        for split, (_, n, _) in COHORTS.items() for start in range(0, n, 100)}
    require(full['batches'].keys() == expected_batches.keys(), 'VOSR replay batches differ')
    checks = {}
    for label, batch in full['batches'].items():
        split, size = expected_batches[label]
        run = batch['run']
        require(batch['split'] == split and len(batch['checks']) == run['input_count'] == size,
                f'{label}: replay batch size differs')
        for field in ('upscale', 'precision', 'seed_per_image', 'steps', 'batch', 'tile', 'color_fix',
                      'vae_posterior', 'vae_precision', 'torch_version'):
            require(run[field] == pilot['run'][field], f'{label}: VOSR replay setting {field} differs')
        expected_names = (f"{row['scene']}__{row['degradation']}.png" for row in batch['checks'])
        require(sorted(sample['filename'] for sample in run['samples']) == sorted(expected_names),
                f'{label}: VOSR output names differ')
        for row in batch['checks']:
            key = (split, row['scene'], row['degradation'])
            require(key not in checks, f'{key}: duplicate source replay case')
            checks[key] = row
    return checks


def verify(root):
    audit = root / 'experiment_audit'
    replay = read(audit / 'frozen_full_pixel_replay.json')
    require(bool(replay.get('completed_utc')), 'Frozen pixel replay has not completed')
    provenance = replay['provenance']
    lock = read(checked(root / 'formal_lock.json', provenance['formal_lock_sha256']))
    checked(root / 'calibration/gates.json', provenance['gates_sha256'])
    checked(audit / 'replay_frozen.py', provenance['replay_sha256'])
    require(provenance['weight_sha256'] == lock['weights']['shafe']['sha256'], 'SHAFE weight differs')
    for field in ('device', 'batch_size'):
        require(provenance[field] == lock['scoring'][field], f'Scoring {field} changed')

    source = read(audit / 'source_replay.json')
    require(bool(source.get('completed_utc')), 'Source replay has not completed')
    checked(audit / 'replay_from_source.py', source['runner_sha256'])
    for field in ('formal_lock_sha256', 'inputs'):
        require(source[field] == provenance[field], f'Source replay {field} differs')
    environment = read(root / 'environment_verified.json')
    require(source['vosr_runner_sha256'] in environment['code_and_records'].values(),
            'VOSR runner fingerprint changed')
    require(source['vosr_weights'] == {row['path']: row['sha256'] for row in environment['vosr_weights']},
            'VOSR weights differ')
    run = source['run']
    require((run['upscale'], run['precision'], run['seed_per_image'], run['input_count']) == (4, 'fp32', 42, 6),
            'VOSR source replay settings differ')
    source_checks = {(row['split'], row['scene'], row['degradation']): row for row in source['checks']}
    require(len(source_checks) == len(source['checks']) == run['input_count'], 'Source replay cases differ')
    expected_source_checks = set()
    full_checks = full_source_checks(audit, lock, provenance, source)

    report = {}
    for split, (scenes, scene_degradations, conditions) in COHORTS.items():
        directory = root / split
        archived = {name: read(checked(directory / f'{name}.json', provenance['inputs'][split][name]))
                    for name in ('batch_inputs', 'geometry', 'decisions', 'scores')}
        records = archived['batch_inputs']['records']
        require(len(records) == scene_degradations and len({r['scene'] for r in records}) == scenes,
                f'{split}: unexpected input coverage')
        by_dataset = defaultdict(set)
        for record in records:
            by_dataset[record['dataset']].add(record['scene'])
        selected = {sorted(ids)[len(ids) // 2] for ids in by_dataset.values()}
        geometry = by_case(archived['geometry']['rows'])
        pair_ids = check_csv(root / 'results' / f'{split}_cases.csv',
                             geometry, *(by_case(archived[name]['rows']) for name in ('decisions', 'scores')))
        checks = replay['replayed'][split]
        require(checks.keys() == pair_ids.keys() and len(checks) == scene_degradations,
                f'{split}: replay coverage differs')
        rows_per_record = conditions // scene_degradations
        for record in records:
            scene, degradation = record['scene'], record['degradation']
            key = f'{scene}__{degradation}'
            row = checks[key]
            require((row['scene'], row['degradation']) == (scene, degradation), f'{key}: record ID differs')
            require(row['conditions'] == rows_per_record and row['pairs'] == len(pair_ids[key]),
                    f'{key}: replay condition or pair counts differ')
            require(0 <= row['max_score_error'] <= provenance['score_tolerance'], f'{key}: SHAFE differs')
            require(0 <= row['max_confidence_error'] <= provenance['confidence_tolerance'],
                    f'{key}: confidence differs')
            source_key = (split, scene, degradation)
            regenerated = full_checks.pop(source_key)
            for field in ('dataset', 'source_sha256', 'lr_sha256', 'hr_sha256'):
                require(regenerated[field] == record[field], f'{key}: full source replay {field} differs')
            require(regenerated['zero_sr_pixel_sha256'] == geometry[scene, degradation, 'zero']['sr_pixel_sha256'],
                    f'{key}: full source replay does not match the SR input to geometry')
            if scene in selected:
                expected_source_checks.add(source_key)
                fresh = source_checks[source_key]
                require(fresh['sr_pixel_sha256'] == regenerated['sr_pixel_sha256'],
                        f'{key}: full and sampled VOSR outputs differ')
                for field in ('dataset', 'lr_sha256', 'hr_sha256'):
                    require(fresh[field] == record[field], f'{key}: source replay {field} differs')
                for field in ('conditions', 'pairs'):
                    require(fresh[field] == row[field], f'{key}: source replay {field} differs')
                require(0 <= fresh['max_score_error'] <= provenance['score_tolerance'],
                        f'{key}: source replay SHAFE differs')
                require(0 <= fresh['max_confidence_error'] <= provenance['confidence_tolerance'],
                        f'{key}: source replay confidence differs')
        report[split] = dict(scenes=scenes, scene_degradations=len(checks),
                             conditions=sum(r['conditions'] for r in checks.values()),
                             pairs=sum(r['pairs'] for r in checks.values()),
                             max_confidence_error=max(r['max_confidence_error'] for r in checks.values()),
                             max_score_error=max(r['max_score_error'] for r in checks.values()))
        require(report[split] == replay['summaries'][split] and report[split]['conditions'] == conditions
                and report[split]['pairs'] == archived['scores']['unique_pairs'], f'{split}: replay totals differ')
    require(source_checks.keys() == expected_source_checks, 'Source replay selection differs')
    require(not full_checks, 'Unrecognized VOSR source replay cases')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))
