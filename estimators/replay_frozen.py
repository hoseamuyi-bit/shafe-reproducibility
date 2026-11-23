"""Replay the frozen formal and RealSR injection evaluations from archived images."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev_pipeline import OLD, dump, sha
from experiment import CONDITIONS, load
from freeze import GATES, LOCK, verify, verify_inputs
from gate import actions
from measure import measure_record
from score import BATCH_SIZE, DEVICE, score_record


COHORTS = {'formal': (300, CONDITIONS), 'external': (32, CONDITIONS[:4])}
CONFIDENCE_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-5


def group(rows):
    grouped = {}
    for row in rows:
        cases = grouped.setdefault((row['scene'], row['degradation']), {})
        if row['condition'] in cases:
            raise ValueError(f'Duplicate condition: {row["scene"]} {row["condition"]}')
        cases[row['condition']] = row
    return grouped


def read_cohort(split, frozen):
    base = ROOT / split
    paths = {name: base / f'{name}.json' for name in ('batch_inputs', 'geometry', 'decisions', 'scores')}
    manifest, geometry, decisions, scores = (load(path) for path in paths.values())
    hashes = {name: sha(path) for name, path in paths.items()}
    if split == 'formal':
        verify_inputs(manifest, frozen)
    elif manifest['formal_lock_sha256'] != sha(LOCK):
        raise ValueError('RealSR input manifest differs from the formal lock')

    settings = geometry['settings']
    scoring = scores['scoring']
    if (settings['sources_sha256'] != hashes['batch_inputs'] or settings['reference']
            or not settings['include_direct'] or scoring['geometry_sha256'] != hashes['geometry']
            or scoring['gates_sha256'] != sha(GATES) or decisions['gates_sha256'] != sha(GATES)
            or scoring['weight_sha256'] != frozen['weights']['shafe']['sha256']
            or any(scoring[name] != frozen['scoring'][name] for name in frozen['scoring'])):
        raise ValueError(f'{split}: archived run provenance differs')

    count, conditions = COHORTS[split]
    records = manifest['records']
    keys = {(row['scene'], row['degradation']) for row in records}
    archived = [group(data['rows']) for data in (geometry, decisions, scores)]
    if (len(records) != count or len(keys) != count
            or any(table.keys() != keys or any(tuple(cases) != conditions for cases in table.values())
                   for table in archived)):
        raise ValueError(f'{split}: incomplete or inconsistent case coverage')
    return records, archived, hashes, scores['unique_pairs']


def measure(record):
    cv2.setNumThreads(2)
    return measure_record(record, reference=False, include_direct=True)


def difference(observed, archived, tolerance, label):
    if observed is None and archived is None:
        return 0.0
    if observed is None or archived is None:
        raise ValueError(f'{label}: missing score differs')
    error = abs(observed - archived)
    if not math.isfinite(error) or error > tolerance:
        raise ValueError(f'{label}: difference {error!r} exceeds {tolerance}')
    return error


def compare_row(observed, archived, label, ignore=()):
    if observed.keys() != archived.keys():
        raise ValueError(f'{label}: row fields differ')
    for field, value in observed.items():
        if field not in ignore and value != archived[field]:
            raise ValueError(f'{label}: {field} differs')


def compare_record(record, measured, archived, gates, evaluate):
    key = record['scene'], record['degradation']
    original_geometry, original_decisions, original_scores = (table[key] for table in archived)
    packed = {row['condition']: row for row in measured}
    if tuple(packed) != tuple(original_geometry) or len(packed) != len(measured):
        raise ValueError(f'{key}: measured conditions differ')
    confidence_errors = []
    for condition, observed in packed.items():
        label = f'{key} {condition}'
        compare_row(observed, original_geometry[condition], label)
        vectors, confidence, reasons, attribution, statuses = actions(observed, gates)
        decision = dict(observed, actions=vectors, confidence=confidence, reasons=reasons,
                        attribution=attribution, legacy_status=statuses)
        prior = original_decisions[condition]
        compare_row(decision, prior, label, ignore=('confidence',))
        if confidence.keys() != prior['confidence'].keys():
            raise ValueError(f'{label}: confidence methods differ')
        confidence_errors.extend(difference(value, prior['confidence'][method], CONFIDENCE_TOLERANCE,
                                            f'{label} {method} confidence')
                                 for method, value in confidence.items())
    rows, pairs = score_record(record, packed, gates, evaluate)
    distinct_pairs = {pair for row in original_scores.values() for pair in row['score_pair_ids'].values()}
    if tuple(row['condition'] for row in rows) != tuple(original_scores) or pairs != len(distinct_pairs):
        raise ValueError(f'{key}: scoring coverage differs')
    score_errors = []
    for row in rows:
        label = f'{key} {row["condition"]}'
        prior = original_scores[row['condition']]
        compare_row(row, prior, label, ignore=('scores',))
        if row['scores'].keys() != prior['scores'].keys():
            raise ValueError(f'{label}: SHAFE methods differ')
        score_errors.extend(difference(value, prior['scores'][method], SCORE_TOLERANCE,
                                      f'{label} {method} SHAFE')
                            for method, value in row['scores'].items())
    return dict(scene=key[0], degradation=key[1], conditions=len(rows), pairs=pairs,
                max_confidence_error=max(confidence_errors), max_score_error=max(score_errors))


def checkpoint(output, result):
    temporary = output.with_suffix('.tmp')
    dump(temporary, result)
    temporary.replace(output)


def replay(output, workers):
    import torch

    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    frozen = verify()
    weight_hash = sha(DEFAULT_WEIGHT)
    if weight_hash != frozen['weights']['shafe']['sha256']:
        raise ValueError('SHAFE weight differs from the formal lock')
    cohorts = {split: read_cohort(split, frozen) for split in COHORTS}
    provenance = dict(
        replay_sha256=sha(__file__), formal_lock_sha256=sha(LOCK), gates_sha256=sha(GATES),
        weight_sha256=weight_hash, device=DEVICE, gpu=torch.cuda.get_device_name(),
        batch_size=BATCH_SIZE, confidence_tolerance=CONFIDENCE_TOLERANCE,
        score_tolerance=SCORE_TOLERANCE, inputs={split: data[2] for split, data in cohorts.items()},
        input_scope='archived LR, HR, and VOSR SR; no new VOSR inference',
    )
    result = load(output) if output.exists() else dict(provenance=provenance, replayed={}, summaries={})
    if result['provenance'] != provenance:
        raise ValueError('Replay implementation or inputs differ from the checkpoint')
    gates = load(GATES)
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)

    def evaluate(pairs):
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, (records, archived, _, expected_pairs) in cohorts.items():
            checks = result['replayed'].setdefault(split, {})
            names = [f'{record["scene"]}__{record["degradation"]}' for record in records]
            pending = [(name, record) for name, record in zip(names, records, strict=True) if name not in checks]
            if pending:
                print(f'{split}: recomputing {len(pending)}/{len(records)} scene-degradations', flush=True)
            for (name, record), measured in zip(pending, pool.map(measure, (r for _, r in pending)), strict=True):
                checks[name] = compare_record(record, measured, archived, gates, evaluate)
                checkpoint(output, result)
                if len(checks) % 10 == 0 or len(checks) == len(records):
                    print(f'{split}: {len(checks)}/{len(records)} scene-degradations match', flush=True)
            summary = dict(scenes=len({row['scene'] for row in checks.values()}), scene_degradations=len(checks),
                           conditions=sum(row['conditions'] for row in checks.values()),
                           pairs=sum(row['pairs'] for row in checks.values()),
                           max_confidence_error=max(row['max_confidence_error'] for row in checks.values()),
                           max_score_error=max(row['max_score_error'] for row in checks.values()))
            if (checks.keys() != set(names) or summary['pairs'] != expected_pairs
                    or summary['conditions'] != len(records) * len(COHORTS[split][1])):
                raise ValueError(f'{split}: replay totals differ from the archive')
            result['summaries'][split] = summary
            print(f'{split}: {summary}', flush=True)
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--geometry-workers', type=int, default=3)
    args = parser.parse_args()
    replay(args.output, args.geometry_workers)
