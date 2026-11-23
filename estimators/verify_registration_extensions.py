"""Check archived registration extensions without image assets or model weights."""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from verify_confirmation import check_summary, equal, index
from verify_holdout import digest, event, max_abs, read


METHODS = ('raw', 'simple_lr', 'tile_trimmed', 'median_lk', 'ecc', 'phase_correlation', 'lk_consensus')
NEW = METHODS[-3:]
LOCAL = ('replacement_256', 'shift2_replacement_256')
SYNTHETIC = ('zero', 'hr_global', 'sr_global', 'both_global', *LOCAL)
CAMERA = ('zero', 'shift2', *LOCAL)


def check_lock(base, hashes):
    for name, expected in hashes.items():
        path = base.parent / name
        if not path.is_file() and Path(name).parent == Path('.'):
            path = base.parent / 'frozen_implementation' / name
        assert digest(path) == expected, name


def check_actions(row, methods):
    assert set(row['actions']) == set(methods)
    assert row['actions']['raw'] == [0, 0]
    for vector in row['actions'].values():
        assert len(vector) == 2 and max_abs(vector) <= 8
        assert all(4 * v == round(4 * v) for v in vector)
        assert max_abs(vector) > .25 or vector == [0, 0]


def check_stratum(summary, cases, geometry):
    check_summary(cases, summary['shafe_mae'], lambda r, m: abs(r['scores'][m] - r['scores']['oracle']))
    if geometry is not None:
        published = summary['geometry']
        check_summary(cases, published, geometry)
        assert published['counts'] == {m: sum(geometry(row, m) for row in cases) for m in METHODS}


def check_benchmarks(base, cohort):
    camera = cohort == 'realsr'
    folder = base / 'registration_benchmarks'
    result = read(folder, f'{cohort}_results.json')
    plan_path = folder / 'lock.json'
    plan = read(folder, plan_path.name)
    assert result['lock_sha256'] == digest(plan_path)
    check_lock(base, plan['hashes'])
    relative = 'coco_confirm/sensitivity/realsr_lk_analysis.json' if camera else 'coco_confirm/scores.json'
    assert plan['hashes'][f'experiment_audit/{relative}'] == digest(base / relative)
    sources = index(read(base, relative)['rows'])
    rows = index(result['rows'])
    scenes = {scene for scene, _ in rows}
    conditions = CAMERA if camera else SYNTHETIC
    assert len(scenes) == (32 if camera else 144)
    assert rows.keys() == sources.keys() == set(product(scenes, conditions))

    for key, row in rows.items():
        source = sources[key]
        check_actions(row, METHODS)
        assert row['hr_added'] == source['hr_added']
        assert all(row['actions'][m] == source['actions'][m] for m in METHODS if m not in NEW)
        assert all(row['scores'][m] == source['scores'][m] for m in (*METHODS, 'oracle') if m not in NEW)
        for method, fit in row['benchmark_fits'].items():
            assert fit['action'] == row['actions'][method]
            if fit['valid']:
                vector = np.rint(-16 * np.asarray(fit['displacement_lr'])) / 4
                expected = vector.tolist() if max_abs(vector) > .25 else [0, 0]
                assert max_abs(fit['displacement_lr']) <= 2 and fit['action'] == expected
            else:
                assert fit['action'] == [0, 0]
            reuse = row['score_reused'][method]
            if reuse is not None:
                assert source['scores'][reuse] == row['scores'][method]
                expected = source['actions'].get(reuse, [-x for x in row['hr_added']])
                assert expected == row['actions'][method]
        if not camera:
            assert source['pair_ids']['oracle'].split(':')[0] == row['sr_crop_sha256']
            twin = {'sr_global': 'zero', 'both_global': 'hr_global'}.get(row['condition'])
            if twin is not None:
                other = rows[row['scene'], twin]
                assert all(row[k] == other[k] for k in ('hr_pixel_sha256', 'benchmark_fits', 'actions'))

    expected_strata = {*conditions, 'all'} if camera else {*conditions, 'all', *(f'hr_global/{m}px' for m in (0.5, 2, 6))}
    assert set(result['strata']) == expected_strata
    for stratum, summary in result['strata'].items():
        name, _, magnitude = stratum.partition('/')
        cases = [row for row in rows.values() if name in ('all', row['condition']) and
                 (not magnitude or row['magnitude'] == float(magnitude.removesuffix('px')))]
        geometry = None
        if camera and name not in ('all', 'zero'):
            def geometry(row, method):
                baseline = rows[row['scene'], 'zero']['actions'][method]
                residual = np.asarray(row['actions'][method]) - baseline + row['hr_added']
                return max_abs(residual) > 1
        elif not camera and name != 'all':
            endpoint = 'correct' if max_abs(cases[0]['hr_added']) else 'false'

            def geometry(row, method):
                return event(row, method, endpoint)

        assert summary['unavailable'] == {m: sum(not r['benchmark_fits'][m]['valid'] for r in cases) for m in NEW}
        check_stratum(summary, cases, geometry)
    print(f'{cohort} benchmarks: verified {len(rows)} cases, {len(result["strata"])} strata')


def check_pixel_replay(base, name, folder, manifest, unique_pairs, result):
    replay = read(base, 'deformation_full_pixel_replay.json')
    assert replay['completed_utc']
    provenance = replay['provenance']
    assert provenance['score_tolerance'] == 1e-6
    assert provenance['replay_sha256'] == digest(base / 'replay_warp_cohorts.py')
    assert provenance['pilot_sha256'] == digest(base / 'deformation_pixel_replay.json')
    assert provenance['cohorts'][name] == {
        'lock_sha256': digest(folder / 'lock.json'), 'manifest_sha256': digest(manifest),
        'results_sha256': digest(folder / 'results.json'),
    }
    pilot = read(base, 'deformation_pixel_replay.json')
    assert provenance['weight_sha256'] == pilot['shafe_weight_sha256']
    prior, = (r for r in pilot['checks'] if r['cohort'] == name)
    records = replay['replayed'][name]
    assert records[prior['scene']] == dict(prior, source='prior_pixel_replay')
    pair_counts = Counter(scene for scene, _ in unique_pairs)
    assert records.keys() == pair_counts.keys()
    for scene, row in records.items():
        assert row['conditions'] == 8 and row['pairs'] == pair_counts[scene]
        assert 0 <= row['max_score_error'] <= provenance['score_tolerance']
        assert row['source'] == ('prior_pixel_replay' if scene == prior['scene'] else 'full_cohort_replay')
    assert replay['summaries'][name] == dict(
        scenes=len(pair_counts), conditions=len(result['rows']), pairs=len(unique_pairs),
        fresh_scenes=len(pair_counts) - 1, max_score_error=max(r['max_score_error'] for r in records.values()),
    )
    print(f'{name} pixel replay record verified for {len(pair_counts)} scenes')


def check_deformation(base, camera=False, pixel_replay=False):
    folder = base / ('camera_deformation' if camera else 'local_deformation')
    plan, result = read(folder, 'lock.json'), read(folder, 'results.json')
    assert result['lock_sha256'] == digest(folder / 'lock.json')
    check_lock(base, plan['hashes'])
    if camera:
        manifest = base.parent / 'external/batch_inputs.json'
        original_path = base / 'registration_benchmarks/realsr_results.json'
        assert plan['hashes']['experiment_audit/local_deformation/lock.json'] == digest(base / 'local_deformation/lock.json')
        assert plan['hashes']['experiment_audit/registration_benchmarks/realsr_results.json'] == digest(original_path)
        direction_field, expected_scenes = 'hr_direction', 32
    else:
        manifest = base / 'coco_confirm/batch_inputs.json'
        original_path = base / 'coco_confirm/scores.json'
        assert plan['revision']['previous_sha256'] == digest(folder / 'lock_v1_before_dtype_fix.json')
        assert plan['parameters'] == read(folder, 'lock_v1_before_dtype_fix.json')['parameters']
        direction_field, expected_scenes = 'direction', 144
    assert plan['hashes'][manifest.relative_to(base.parent).as_posix()] == digest(manifest)
    records = {r['scene']: r for r in read(manifest.parent, manifest.name)['records']}
    original = index(read(original_path.parent, original_path.name)['rows'])
    rows = index(result['rows'])
    design = {f'diameter{diameter}_local{amplitude}_global{translation}': (diameter, amplitude, translation)
              for diameter, amplitude, translation in product((256, 384), (4, 16), (0, 2))}
    assert len(records) == expected_scenes and set(records) == set(plan['parameters'])
    assert rows.keys() == set(product(records, design))
    assert result['strata'].keys() == design.keys()
    unique_pairs = set()
    for scene, record in records.items():
        scores = {}
        zero = original[scene, 'zero']
        sr_hash = zero['sr_crop_sha256'] if camera else zero['pair_ids']['oracle'].split(':')[0]
        for name, (diameter, amplitude, translation) in design.items():
            row = rows[scene, name]
            check_actions(row, (*METHODS, 'oracle'))
            assert row['lr_pixel_sha256'] == record['lr_pixel_sha256']
            assert (row['diameter'], row['local_peak'], row['global_magnitude']) == (diameter, amplitude, translation)
            assert row['hr_added'] == [translation * d for d in record[direction_field]]
            assert row['actions']['oracle'] == [-v for v in row['hr_added']]
            if camera:
                residuals = {m: max_abs([a - b + h for a, b, h in zip(row['actions'][m], zero['actions'][m],
                                                                   row['hr_added'], strict=True)]) for m in METHODS}
                assert row['incremental_error'] == residuals
            assert set(row['valid']) == set(METHODS) and set(row['scores']) == set(row['pair_ids']) == set(row['actions'])
            assert row['valid']['median_lk'] == (row['n_tracks'] >= 8)
            for method, valid in row['valid'].items():
                if not valid:
                    assert row['actions'][method] == [0, 0]
            for method, pair in row['pair_ids'].items():
                assert pair.split(':')[0] == sr_hash
                equal(scores.setdefault(pair, row['scores'][method]), row['scores'][method])
            for a, b in product(row['actions'], repeat=2):
                if row['actions'][a] == row['actions'][b]:
                    assert row['pair_ids'][a] == row['pair_ids'][b]
        for diameter, amplitude in product((256, 384), (4, 16)):
            prefix = f'diameter{diameter}_local{amplitude}_global'
            no_shift, shifted = rows[scene, prefix + '0'], rows[scene, prefix + '2']
            assert no_shift['pair_ids']['oracle'] == shifted['pair_ids']['oracle']
        unique_pairs.update((scene, pair) for pair in scores)
    assert len(unique_pairs) == result['unique_pairs']
    for name, summary in result['strata'].items():
        cases = [rows[scene, name] for scene in records]
        endpoint = 'correct' if design[name][-1] else 'false'
        assert summary['unavailable'] == {m: sum(not r['valid'][m] for r in cases) for m in METHODS}
        if camera:
            check_stratum(summary, cases, lambda r, m: r['incremental_error'][m] > 1)
            check_summary(cases, summary['mean_residual'], lambda r, m: r['incremental_error'][m])
        else:
            check_stratum(summary, cases, lambda r, m: event(r, m, endpoint))
    name = 'RealSR' if camera else 'COCO'
    diagnostics = read(base, 'deformation_tracks.json')['cohorts'][name]
    assert diagnostics['lock_sha256'] == result['lock_sha256']
    tracks = index(diagnostics['rows'])
    assert tracks.keys() == rows.keys()
    for key, counts in tracks.items():
        assert rows[key]['n_tracks'] == counts['accepted']
        assert 0 <= counts['in_disk_accepted'] <= counts['accepted'] <= counts['forward_backward'] <= counts['detected']
        assert counts['in_disk_accepted'] <= counts['in_disk_forward_backward'] <= counts['forward_backward']
    for condition, means in diagnostics['mean_tracks_per_scene'].items():
        cases = [counts for (_, key), counts in tracks.items() if key == condition]
        for field, average in means.items():
            equal(average, np.mean([counts[field] for counts in cases]))
    print(f'{name} local deformation: verified {len(rows)} cases, {len(unique_pairs)} SHAFE pairs')
    if pixel_replay:
        check_pixel_replay(base, name, folder, manifest, unique_pairs, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--cohort', choices=('all', 'coco', 'realsr', 'deformation', 'camera-deformation'), default='all')
    parser.add_argument('--pixel-replay', action='store_true', help='also validate the full replay records')
    args = parser.parse_args()
    for cohort in ('coco', 'realsr'):
        if args.cohort in ('all', cohort):
            check_benchmarks(args.archive, cohort)
    if args.cohort in ('all', 'deformation'):
        check_deformation(args.archive, pixel_replay=args.pixel_replay)
    if args.cohort in ('all', 'camera-deformation'):
        check_deformation(args.archive, camera=True, pixel_replay=args.pixel_replay)
