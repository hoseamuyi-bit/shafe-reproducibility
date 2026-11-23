"""Replay the remaining frozen smooth-warp scenes without using scoring caches."""

from datetime import datetime, timezone
import math
import sys

import cv2

import camera_deformation as camera
import local_deformation as warp
from holdout_run import OLD, ROOT, atomic_dump, load, sha
from score import BATCH_SIZE, DEVICE


HERE = warp.DEST.parent
PILOT = HERE / 'deformation_pixel_replay.json'
OUTPUT = HERE / 'deformation_full_pixel_replay.json'
TOLERANCE = 1e-6


def cohort_inputs():
    camera_plan = camera.verify_inputs()
    coco_plan = load(warp.LOCK)
    for path, expected in coco_plan['hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Frozen COCO input changed: {path}')
    return (
        ('RealSR', camera.DEST, camera.INPUTS, camera_plan, 32),
        ('COCO', warp.DEST, warp.INPUTS, coco_plan, 144),
    )


def group_rows(rows):
    scenes = {}
    for row in rows:
        conditions = scenes.setdefault(row['scene'], {})
        if row['condition'] in conditions:
            raise ValueError(f'Duplicate case: {row["scene"]} {row["condition"]}')
        conditions[row['condition']] = row
    return scenes


def compare(scene, observed, archived, pairs):
    if len(observed) != 8 or set(archived) != {row['condition'] for row in observed}:
        raise ValueError(f'Incomplete or duplicate conditions for {scene}')
    if pairs != len({p for row in archived.values() for p in row['pair_ids'].values()}):
        raise ValueError(f'Unique scoring-pair mismatch for {scene}')
    max_error = 0.0
    for row in observed:
        prior = archived[row['condition']]
        for field, value in row.items():
            if field != 'scores' and value != prior[field]:
                raise ValueError(f'{scene} {row["condition"]}: {field} differs')
        if row['scores'].keys() != prior['scores'].keys():
            raise ValueError(f'{scene} {row["condition"]}: scoring methods differ')
        for method, value in row['scores'].items():
            error = abs(value - prior['scores'][method])
            if not math.isfinite(error) or error > TOLERANCE:
                raise ValueError(f'{scene} {row["condition"]} {method}: SHAFE difference {error}')
            max_error = max(max_error, error)
    return max_error


def replay():
    cohorts = cohort_inputs()
    warp.self_test()
    cv2.setNumThreads(1)
    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    weight = sha(DEFAULT_WEIGHT)
    expected_weight = load(warp.benchmarks.coco_confirm.PLAN)['shafe']['weight_sha256']
    pilot = load(PILOT)
    if weight != expected_weight or weight != pilot['shafe_weight_sha256']:
        raise ValueError('SHAFE weight mismatch')
    provenance = dict(
        replay_sha256=sha(__file__), pilot_sha256=sha(PILOT), weight_sha256=weight,
        batch_size=BATCH_SIZE, device=DEVICE, score_tolerance=TOLERANCE,
        cohorts={label: dict(lock_sha256=sha(folder / 'lock.json'), manifest_sha256=sha(manifest),
                             results_sha256=sha(folder / 'results.json'))
                 for label, folder, manifest, _, _ in cohorts},
    )
    result = load(OUTPUT) if OUTPUT.exists() else dict(provenance=provenance, replayed={}, summaries={})
    if result['provenance'] != provenance:
        raise ValueError('Replay inputs or implementation changed since checkpoint')
    pilot_checks = {(r['cohort'], r['scene']): r for r in pilot['checks']}

    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)

    def evaluate(pairs):
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    for label, folder, manifest, plan, n in cohorts:
        archived = load(folder / 'results.json')
        lock_hash = provenance['cohorts'][label]['lock_sha256']
        records = load(manifest)['records']
        prior = group_rows(archived['rows'])
        if (archived['lock_sha256'] != lock_hash or len(records) != n or len(prior) != n
                or len(archived['rows']) != 8 * n
                or set(prior) != {record['scene'] for record in records}):
            raise ValueError(f'{label} archived cohort does not match the frozen inputs')
        checks = result['replayed'].setdefault(label, {})
        for index, record in enumerate(records, 1):
            scene = record['scene']
            first = pilot_checks.get((label, scene))
            if first:
                checkpoint = load(folder / 'scenes' / f'{scene}.json')
                if (first['lock_sha256'] != lock_hash or first['conditions'] != 8
                        or first['pairs'] != checkpoint['unique_pairs']
                        or group_rows(checkpoint['rows'])[scene] != prior[scene]
                        or not 0 <= first['max_score_error'] <= TOLERANCE):
                    raise ValueError(f'Prior pixel replay could not be linked to {scene}')
                checks[scene] = dict(first, source='prior_pixel_replay')
            elif scene not in checks:
                if label == 'RealSR':
                    record = dict(record, direction=record['hr_direction'])
                rows, pairs = warp.process(record, plan['parameters'][scene], evaluate)
                error = compare(scene, rows, prior[scene], pairs)
                checks[scene] = dict(conditions=len(rows), pairs=pairs, max_score_error=error,
                                     source='full_cohort_replay')
                atomic_dump(OUTPUT, result)
            if index % 8 == 0 or index == n:
                print(f'{label} {index}/{n} scenes match pixels, actions, and SHAFE scores', flush=True)
        summary = dict(scenes=len(checks), conditions=sum(r['conditions'] for r in checks.values()),
                       pairs=sum(r['pairs'] for r in checks.values()),
                       max_score_error=max(r['max_score_error'] for r in checks.values()),
                       fresh_scenes=sum(r['source'] == 'full_cohort_replay' for r in checks.values()))
        if (checks.keys() != prior.keys() or summary['conditions'] != len(archived['rows'])
                or summary['pairs'] != archived['unique_pairs']):
            raise ValueError(f'{label} replay totals do not match archived totals')
        result['summaries'][label] = summary
        atomic_dump(OUTPUT, result)
        print(f'{label} complete: {summary}', flush=True)
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_dump(OUTPUT, result)


if __name__ == '__main__':
    replay()
