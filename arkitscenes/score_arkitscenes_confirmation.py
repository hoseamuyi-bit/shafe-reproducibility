"""Rebuild verified inputs and evaluate the locked, visit-paired confirmation."""

from pathlib import Path
from unittest.mock import patch

import score_arkitscenes_temporal as scoring
import audit_arkitscenes_reference_controls as controls
from check_arkitscenes_temporal_metrics import CONTROL_NAMES
from check_temporal_shafe_radius import ControlRecorder, MatchedCutoffScorer, rescore
from compare_temporal_reference_controls import DRAWS, SEED, gaps, paired_summary, score_gaps
import ranking_models
from run_arkitscenes_confirmation import BOOTSTRAP, MARGINS, OUT, RADIUS_MATCHED, ROOT, screen
from verify_arkitscenes_temporal_acquisition import verify


def shafe(rows, condition, method):
    return {model: rows[condition][model][method]['SHAFE'] for model in scoring.MODELS}


def score_controls(rows, selected, visits, paper_scorer):
    matched_scorer = MatchedCutoffScorer(paper_scorer)
    controls.check_roundtrip()
    for row in rows:
        visit = row['visit']
        source = selected[visit]
        recorder = ControlRecorder(matched_scorer, paper_scorer)
        result = controls.run_case(OUT, source, row, recorder)['poses']
        labels = [(pose, label) for pose in screen.POSE_NAMES for label in CONTROL_NAMES
                  if result[pose].get(label) is not None]
        by_label = dict(zip(labels, recorder.calls, strict=True))
        if all(result[name]['same_frame'] is not None for name in screen.POSE_NAMES):
            matched = rescore(OUT, row, source['selected'], matched_scorer)
            sync = shafe(matched, 'sync', 'fixed')
            for pose in screen.POSE_NAMES:
                sha, values = by_label[pose, 'same_frame']
                same = dict(zip(scoring.MODELS, [value['SHAFE'] for value in values[1:]], strict=True))
                visits[visit][str(RADIUS_MATCHED)][pose] = dict(
                    same_frame=score_gaps(sync, same),
                    moved=score_gaps(sync, shafe(matched, 'moved', 'depth_' + pose)),
                    same_frame_pixel_sha256=sha)
                visits[visit]['50'][pose] = dict(
                    same_frame=gaps(result[pose]['same_frame']['ranking']),
                    moved=gaps(result[pose]['moved_rank']), same_frame_pixel_sha256=sha)
        print(f'Replayed reference controls for {visit}', flush=True)


def sign_flip(units):
    counts = [row['first_only'] - row['second_only'] for row in units.values()]
    # Integer bin counts give the exact randomization distribution of the sum.
    distribution = {0: 1}
    for count in counts:
        new = {}
        for value, weight in distribution.items():
            for signed in (value - count, value + count):
                new[signed] = new.get(signed, 0) + weight
        distribution = new
    return sum(weight for value, weight in distribution.items()
               if abs(value) >= abs(sum(counts))) / 2**len(counts)


def run(sampling_path):
    files = {name: OUT / f'{name}.json' for name in
             ('protocol', 'inputs', 'acquisition_replay', 'scores')}
    sample = screen.read(sampling_path)
    plan = screen.read(files['protocol'])
    expected = sorted((row['video_id'], row['visit_id']) for row in sample['sample'])
    actual = sorted((row['video_id'], row['visit_id']) for row in plan['ranking'])
    if (plan['sampling_protocol_sha256'] != screen.digest(sampling_path)
            or plan['hashes'] != sample['hashes'] or plan['endpoint'] != sample['endpoint']
            or plan['candidates'] != [row for row in plan['ranking'] if row['priority_pairs']]
            or actual != expected or len({row[1] for row in actual}) != len(actual)
            or any(screen.digest(ROOT / name) != digest for name, digest in sample['hashes'].items())
            or (DRAWS, SEED) != (BOOTSTRAP['draws'], BOOTSTRAP['seed'])):
        raise ValueError('The acquired roster or analysis differs from the frozen protocol')

    verify(OUT)
    inputs = screen.read(files['inputs'])
    selected = {row['visit']: row for row in inputs['visits'] if row['selected']}
    visits = {visit: {'50': {}, str(RADIUS_MATCHED): {}} for visit in selected}
    if selected:
        vosr = Path(__file__).with_name('run_vosr_pread.py').resolve()
        with patch.object(scoring, 'OUT', OUT), patch.object(ranking_models, 'VOSR', vosr):
            scoring.run()
        scores = screen.read(files['scores'])
        score_controls(scores['visits'], selected, visits, scoring.Scorer())
    else:
        del files['scores']
    common = [visit for visit, values in visits.items() if values['50']]

    results = {}
    if len(common) >= 2:
        for margin in MARGINS:
            results[str(margin)] = {}
            for radius in ('50', str(RADIUS_MATCHED)):
                by_pose = {}
                for pose in screen.POSE_NAMES:
                    windows = [(visit, selected[visit]['video'],
                                visits[visit][radius][pose]['moved'], visits[visit][radius][pose]['same_frame'])
                               for visit in common]
                    by_pose[pose] = paired_summary(windows, 'moved', 'same_frame', margin, True)
                results[str(margin)][radius] = by_pose
    primary = results[str(MARGINS[0])]['50']['linear'] if results else None
    p = sign_flip(primary['units']) if primary else None

    def make_serializable(values):
        return {radius: {pose: dict(
            data, same_frame=[[*pair, *gap] for pair, gap in sorted(data['same_frame'].items())],
            moved=[[*pair, *gap] for pair, gap in sorted(data['moved'].items())])
            for pose, data in poses.items()} for radius, poses in values.items()}

    report = dict(sampling_sha256=screen.digest(sampling_path),
                  files_sha256={name: screen.digest(path) for name, path in files.items()},
                  endpoint=plan['endpoint'], frozen_visits=len(actual),
                  missing_official_pose=sum(bool(row.get('availability')) for row in plan['ranking']),
                  pose_eligible=len(inputs['visits']),
                  rgb_downloaded=sum(bool(row.get('rgb_pairs_downloaded')) for row in inputs['visits']),
                  scored_visits=len(selected), common_support_visits=len(common),
                  fully_supported_visits=common, primary_exact_sign_flip_p=p,
                  primary_passed=bool(primary and primary['bootstrap_95'][0] > 0 and p < 0.05),
                  results=results, pair_gaps={visit: make_serializable(values)
                                              for visit, values in visits.items()})
    path = OUT / 'confirmation_results.json'
    screen.write(path, report)
    print(f'Confirmation complete with {len(common)} common-support visits. Results at {path}', flush=True)
