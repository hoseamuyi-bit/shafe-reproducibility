"""Replay practical rank reversals for four full-reference measures on the same visits."""

from pathlib import Path

import audit_arkitscenes_reference_controls as audit
from check_arkitscenes_temporal_metrics import CONTROL_NAMES, RecordedScorer
from compare_temporal_reference_controls import paired_summary, score_gaps
from run_arkitscenes_confirmation import OUT
from score_arkitscenes_temporal import atomic_json
from verify_confirmation_statistics import digest, main as verify, read


MARGINS = {'PSNR': (0., 0.05), 'SSIM': (0., 0.0005),
           'SHAFE': (0., 0.0005), 'ST-LPIPS': (0., 0.0005)}


def replay(directory, source, score, scorer, expected):
    recorded = RecordedScorer(scorer)
    replayed = audit.run_case(directory, source, score, recorded)['poses']
    labels = [(pose, label) for pose in audit.cohort.POSE_NAMES for label in CONTROL_NAMES
              if replayed[pose].get(label) is not None]
    results = {}
    for (pose, label), (scores, pixel_hash) in zip(labels, recorded.calls, strict=True):
        if label != 'same_frame':
            continue
        if pixel_hash != expected[pose]['same_frame_pixel_sha256']:
            raise ValueError(f'Round-trip pixels changed for {score["visit"]}/{pose}')
        model_scores = dict(zip(audit.MODELS, scores[1:], strict=True))
        results[pose] = dict(pixel_sha256=pixel_hash, model_scores=model_scores)
    if set(results) != set(audit.cohort.POSE_NAMES):
        raise ValueError(f'Incomplete round-trip support for {score["visit"]}')
    return results


def summarize(scores, controls, primary):
    data = {}
    for pose in audit.cohort.POSE_NAMES:
        data[pose] = {}
        for metric, margins in MARGINS.items():
            cases = []
            for row in scores:
                visit = row['visit']
                if visit not in controls:
                    continue
                sync = {model: row['scores']['sync'][model]['fixed'][metric] for model in audit.MODELS}
                moved = {model: row['scores']['moved'][model]['depth_' + pose][metric] for model in audit.MODELS}
                control = {model: controls[visit][pose]['model_scores'][model][metric] for model in audit.MODELS}
                cases.append((visit, row['video'], score_gaps(sync, moved), score_gaps(sync, control)))
            results = {str(margin): paired_summary(cases, 'moved', 'same_frame', margin, True)
                       for margin in margins if len(cases) >= 2}
            if results and metric == 'SHAFE' and results['0.0005'] != primary['results']['0.0005']['50'][pose]:
                raise ValueError(f'Replayed SHAFE ranks differ for {pose}')
            data[pose][metric] = results
    return data


def main(directory):
    verify(directory)
    primary = read(directory, 'confirmation_results')
    inputs = {row['visit']: row for row in read(directory, 'inputs')['visits'] if row['selected']}
    scores = read(directory, 'scores')['visits'] if inputs else []
    supported = set(primary['fully_supported_visits'])
    controls = {}
    if supported:
        audit.cv2.setNumThreads(1)
        audit.check_roundtrip()
        scorer = audit.Scorer()
        for row in scores:
            visit = row['visit']
            if visit in supported:
                controls[visit] = replay(directory, inputs[visit], row, scorer, primary['pair_gaps'][visit]['50'])
                print(f'Cross-metric controls replayed {visit}', flush=True)
        if controls.keys() != supported:
            raise ValueError('Incomplete cross-metric replay')
    atomic_json(directory / 'confirmation_metrics.json', dict(
        primary_sha256=digest(directory / 'confirmation_results.json'), source_sha256=digest(Path(__file__)),
        margins=MARGINS, controls=controls, results=summarize(scores, controls, primary)))


if __name__ == '__main__':
    main(OUT)
