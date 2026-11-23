"""Recompute the frozen COCO confirmation from images in a fresh directory."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import coco_confirm as confirmation
from holdout_run import OLD, atomic_dump, load, sha
from score import BATCH_SIZE, DEVICE


TOLERANCE = 1e-6


def by_scene(rows):
    scenes = {}
    for row in rows:
        cases = scenes.setdefault(row['scene'], {})
        if row['condition'] in cases:
            raise ValueError(f"Duplicate condition for {row['scene']}: {row['condition']}")
        cases[row['condition']] = row
    return scenes


def compare_scores(scene, rows, expected, pair_count):
    actual = by_scene(rows)[scene]
    if actual.keys() != expected.keys():
        raise ValueError(f'{scene}: scoring condition coverage changed')
    if pair_count != len({pair for row in expected.values() for pair in row['pair_ids'].values()}):
        raise ValueError(f'{scene}: distinct scoring-pair count changed')
    errors = []
    for condition, row in actual.items():
        prior = expected[condition]
        if row.keys() != prior.keys() or any(row[k] != prior[k] for k in row if k != 'scores'):
            raise ValueError(f'{scene} {condition}: scoring inputs or actions changed')
        if row['scores'].keys() != prior['scores'].keys():
            raise ValueError(f'{scene} {condition}: scoring methods changed')
        for method, value in row['scores'].items():
            error = abs(value - prior['scores'][method])
            if not math.isfinite(error) or error > TOLERANCE:
                raise ValueError(f'{scene} {condition} {method}: score difference {error}')
            errors.append(error)
    return max(errors)


def replay(output):
    plan = confirmation.verify()
    original_folder = confirmation.FOLDER
    geometry_path = original_folder / 'geometry.json'
    scores_path = original_folder / 'scores.json'
    geometry, scores = load(geometry_path), load(scores_path)
    lock_hash = sha(confirmation.PLAN)
    if (geometry['lock_sha256'] != lock_hash or scores['stamp']['lock_sha256'] != lock_hash
            or scores['stamp']['geometry_sha256'] != sha(geometry_path)):
        raise ValueError('COCO confirmation results no longer match their frozen inputs')
    records = load(confirmation.INPUTS)['records']
    original_geometry = by_scene(geometry['rows'])
    original_scores = by_scene(scores['rows'])
    expected_cases = confirmation.N * len(confirmation.CONDITIONS)
    if (len(records) != confirmation.N or original_geometry.keys() != original_scores.keys()
            or set(original_geometry) != {r['scene'] for r in records}
            or len(geometry['rows']) != expected_cases or len(scores['rows']) != expected_cases):
        raise ValueError('Incomplete confirmation cohort')

    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    weight_hash = sha(DEFAULT_WEIGHT)
    if weight_hash != plan['shafe']['weight_sha256']:
        raise ValueError('SHAFE weight changed')
    provenance = dict(
        replay_sha256=sha(__file__), lock_sha256=lock_hash, geometry_sha256=sha(geometry_path),
        scores_sha256=sha(scores_path), manifest_sha256=sha(confirmation.INPUTS), weight_sha256=weight_hash,
        score_tolerance=TOLERANCE, device=DEVICE, batch_size=BATCH_SIZE,
    )
    result = load(output) if output.exists() else dict(provenance=provenance, replayed={})
    if result['provenance'] != provenance:
        raise ValueError('COCO replay inputs or implementation changed since checkpoint')
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    cases = result['replayed']
    with TemporaryDirectory(prefix='coco_replay_', dir=output.parent) as scratch:
        # The frozen geometry entry point writes scene files under FOLDER.
        confirmation.FOLDER = Path(scratch)
        try:
            for index, record in enumerate(records, 1):
                scene = record['scene']
                if scene not in cases:
                    rows = confirmation.geometry_scene((record, lock_hash))
                    measured = by_scene(rows)[scene]
                    if measured != original_geometry[scene]:
                        raise ValueError(f'{scene}: geometry or pixel hashes changed')
                    scored, count = confirmation.score_input(record, measured, metric, paper_scores)
                    error = compare_scores(scene, scored, original_scores[scene], count)
                    cases[scene] = dict(conditions=len(rows), pairs=count, max_score_error=error)
                    atomic_dump(output, result)
                if index % 12 == 0:
                    print(f'COCO confirmation replay: {index}/{len(records)} scenes agree', flush=True)
        finally:
            confirmation.FOLDER = original_folder
    summary = dict(
        scenes=len(cases), conditions=sum(c['conditions'] for c in cases.values()),
        pairs=sum(c['pairs'] for c in cases.values()),
        max_score_error=max(c['max_score_error'] for c in cases.values()),
    )
    if (cases.keys() != original_geometry.keys() or summary['conditions'] != len(geometry['rows'])
            or summary['pairs'] != scores['unique_pairs']):
        raise ValueError('COCO confirmation pixel replay totals changed')
    result.update(summary=summary, completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_dump(output, result)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?',
                        default=Path(__file__).with_name('coco_confirmation_pixel_replay.json'))
    args = parser.parse_args()
    replay(args.output.resolve())
