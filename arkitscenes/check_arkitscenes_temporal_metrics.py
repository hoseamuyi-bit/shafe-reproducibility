"""Recompute four-metric scores for the frozen ARKitScenes same-frame controls."""

from pathlib import Path

import audit_arkitscenes_reference_controls as audit
from check_temporal_metrics import METRIC_MARGINS, MODELS, POSES, ROOT, digest, load, summarize, verify_counts
from score_arkitscenes_temporal import atomic_json, verify_lock


CONTROL_NAMES = ('local_translation', 'same_frame', 'rgb_bias_corrected')


class RecordedScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = []

    def evaluate(self, pairs):
        scores = self.scorer.evaluate(pairs)
        reference = audit.pixel_sha(pairs[0][1])
        if len(scores) != len(MODELS) + 1 or any(audit.pixel_sha(b) != reference for _, b in pairs):
            raise ValueError('A control call must score the same reference against the baseline and four outputs')
        self.calls.append((scores, reference))
        return scores


def replay(directory, batch, scorer):
    sources = {name: directory / f'{name}.json' for name in batch['artifact_sha256']}
    for name, expected in batch['artifact_sha256'].items():
        if digest(sources[name]) != expected:
            raise ValueError(f'Frozen record changed: {directory.name}/{name}')
    protocol = digest(directory / 'protocol.json')
    if protocol != batch['protocol_sha256']:
        raise ValueError(f'Frozen protocol changed: {directory.name}')
    inputs, scores, archive = (load(sources[name]) for name in ('inputs', 'scores', 'reference_control_replay_v2'))
    if (archive['script_sha256'] != digest(Path(audit.__file__)) or
            archive['scores_sha256'] != digest(sources['scores']) or
            archive['margin'] != audit.MARGIN or
            any(row['inputs_sha256'] != digest(sources['inputs']) or row['protocol_sha256'] != protocol
                for row in (scores, archive))):
        raise ValueError(f'Frozen replay provenance differs: {directory.name}')
    selected = {row['visit']: row for row in inputs['visits'] if row['selected']}
    archived = {row['visit']: row for row in archive['visits']}
    if len(selected) != batch['scored'] or selected.keys() != archived.keys():
        raise ValueError(f'Frozen visit membership differs: {directory.name}')

    visits = {}
    for score in scores['visits']:
        visit = score['visit']
        recorded = RecordedScorer(scorer)
        result = audit.run_case(directory, selected[visit], score, recorded)
        if result != archived[visit]:
            raise ValueError(f'Frozen ARKitScenes geometry or SHAFE replay differs: {visit}')
        expected_calls = [(pose, name) for pose in POSES for name in CONTROL_NAMES
                          if result['poses'][pose].get(name) is not None]
        baseline = {m: score['scores']['sync'][m]['fixed'] for m in MODELS}
        controls, hashes = {}, {}
        for (pose, name), (values, reference) in zip(expected_calls, recorded.calls, strict=True):
            models = dict(zip(MODELS, values[1:], strict=True))
            expected = result['poses'][pose][name]
            if values[0] != expected['over_sync'] or audit.ranks(baseline, models) != expected['ranking']:
                raise ValueError(f'Control score attribution differs: {visit}/{pose}/{name}')
            if name == 'same_frame':
                label = f'{name}_{pose}'
                controls[label] = models
                hashes[label] = reference
        visits[visit] = dict(scores=score['scores'], controls=controls, control_pixel_sha256=hashes)
        print(directory.name, visit, 'verified', len(controls), 'supported round trips', flush=True)
    if visits.keys() != selected.keys():
        raise ValueError(f'Incomplete visit replay: {directory.name}')
    return visits


def main():
    audit.cv2.setNumThreads(1)
    audit.check_roundtrip()
    _, config = verify_lock()
    scorer = audit.Scorer()
    cohort_path = ROOT / 'temporal_complete_draw_summary.json'
    metric_path = ROOT / 'temporal_metric_sensitivity.json'
    totals_path = ROOT / 'temporal_rank_totals_checked.json'
    cohort, metrics, totals = map(load, (cohort_path, metric_path, totals_path))
    if metrics['scoring_lock_sha256'] != config:
        raise ValueError('Scoring locks differ between metric replays')
    splits = dict(validation={}, training={})
    for name, batch in cohort['batches'].items():
        if batch['scored']:
            split = 'training' if name.startswith('arkitscenes_temporal_training') else 'validation'
            visits = replay(ROOT / name, batch, scorer)
            if splits[split].keys() & visits.keys():
                raise ValueError(f'Duplicate visits in {name}')
            splits[split].update(visits)
    results = {}
    for split, visits in splits.items():
        old_visits = metrics['arkitscenes'][split]['visits']
        if {v: c['scores'] for v, c in visits.items()} != {v: c['scores'] for v, c in old_visits.items()}:
            raise ValueError(f'Stored frame scores differ in {split}')
        summary = summarize(visits)
        verify_counts(summary, totals[split]['results']['0.0005']['reversals'])
        coverage = {pose: [v for v, c in visits.items() if f'same_frame_{pose}' in c['controls']] for pose in POSES}
        results[split] = dict(summary=summary, visits=visits, same_frame_visit_ids=coverage)
    record = dict(scope='Post hoc metrics for frozen same-frame round-trip controls.', models=MODELS,
                  margins=METRIC_MARGINS, scoring_lock_sha256=config, script_sha256=digest(Path(__file__)),
                  input_sha256={p.name: digest(p) for p in (cohort_path, metric_path, totals_path)},
                  arkitscenes=results)
    atomic_json(ROOT / 'arkitscenes_temporal_control_metrics.json', record)


if __name__ == '__main__':
    main()
