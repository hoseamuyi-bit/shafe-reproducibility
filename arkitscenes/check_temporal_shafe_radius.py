"""Re-score the temporal conditions at the normalized cutoff of the 448px experiment."""

from pathlib import Path
from unittest.mock import patch

import audit_arkitscenes_reference_controls as audit
import score_arkitscenes_temporal as scoring
from check_arkitscenes_temporal_metrics import CONTROL_NAMES
from check_temporal_metrics import DATA, MODELS, POSES, ROOT, digest, load, replay_controls
from temporal_cluster_sensitivity import CONTRASTS, analyze


FORMAL_SIDE = 448
WINDOW_SIDE = 88
PAPER_RADIUS = 50
MATCHED_RADIUS = PAPER_RADIUS * WINDOW_SIDE / FORMAL_SIDE


class MatchedCutoffScorer:
    def __init__(self, scorer):
        from ranking_scores import BATCH, gray

        self.scorer = scorer
        self.batch = BATCH
        self.gray = gray
        self.formal_settings = scorer.paper_scores.__globals__['CONFIGURATIONS']
        if self.formal_settings['paper_shafe']['radius'] != PAPER_RADIUS:
            raise ValueError('Unexpected paper cutoff')

    def evaluate(self, pairs):
        if any(image.shape != (WINDOW_SIDE, WINDOW_SIDE, 3) for pair in pairs for image in pair):
            raise ValueError('Unexpected normalized cutoff support')
        gray = [(self.gray(a), self.gray(b)) for a, b in pairs]
        settings = dict(self.formal_settings['paper_shafe'], radius=MATCHED_RADIUS)
        with patch.dict(self.formal_settings, {'paper_shafe': settings}):
            values = self.scorer.paper_scores(self.scorer.shafe, gray, 'cuda', self.batch)
        return [{'SHAFE': value} for value in values]


class ControlRecorder:
    def __init__(self, scorer, reference_scorer=None):
        self.scorer = scorer
        self.reference_scorer = reference_scorer
        self.calls = []

    def evaluate(self, pairs):
        values = self.scorer.evaluate(pairs)
        self.calls.append((audit.pixel_sha(pairs[0][1]), values))
        if self.reference_scorer is not None:
            return self.reference_scorer.evaluate(pairs)
        return values


def rescore(directory, old, selected, scorer):
    if len(old['sr_pixel_sha256']) != len(MODELS):
        raise ValueError('Incomplete restoration manifest')
    paths = {m: scoring.artifact(directory, m, old['lr_sha256']) for m in MODELS}
    with patch.object(scoring, 'OUT', directory):
        new = scoring.score_visit(selected, paths, scorer)
    keys = ('lr_sha256', 'sr_pixel_sha256', 'score_xyxy', 'actions', 'fit_quality')
    if any(new[key] != old[key] for key in keys):
        raise ValueError(f'Score pixels or alignment changed: {directory.name}/{old["video"]}')
    return new['scores']


def arkit_groups(cohort, recorded, scorer):
    splits = dict(training={}, validation={})
    for name, batch in cohort['batches'].items():
        if not batch['scored']:
            continue
        split = 'training' if name.startswith('arkitscenes_temporal_training') else 'validation'
        directory = ROOT / name
        for key in ('inputs', 'scores', 'reference_control_replay_v2'):
            if digest(directory / f'{key}.json') != batch['artifact_sha256'][key]:
                raise ValueError(f'ARKitScenes frozen input differs: {name}/{key}')
        inputs, old, archive = (load(directory / f'{key}.json')['visits']
                                for key in ('inputs', 'scores', 'reference_control_replay_v2'))
        selected = {row['visit']: row for row in inputs if row['selected']}
        archived = {row['visit']: row['poses'] for row in archive}
        for row in old:
            visit = row['visit']
            expected = recorded[split]['visits'][visit]
            scores = rescore(directory, row, selected[visit]['selected'], scorer)
            controls = ControlRecorder(scorer)
            actual = audit.run_case(directory, selected[visit], row, controls)['poses']
            labels = [(pose, label) for pose in POSES for label in CONTROL_NAMES
                      if archived[visit][pose].get(label) is not None]
            if labels != [(pose, label) for pose in POSES for label in CONTROL_NAMES
                          if actual[pose].get(label) is not None]:
                raise ValueError(f'ARKitScenes control eligibility differs: {visit}')
            output = {}
            for (pose, label), (sha, values) in zip(labels, controls.calls, strict=True):
                if label == 'same_frame':
                    key = f'{label}_{pose}'
                    if sha != expected['control_pixel_sha256'][key]:
                        raise ValueError(f'ARKitScenes control pixels differ: {visit}/{pose}')
                    output[key] = dict(zip(MODELS, values[1:], strict=True))
            if output.keys() != expected['controls'].keys():
                raise ValueError(f'ARKitScenes controls differ: {visit}')
            splits[split][visit] = [dict(scores=scores, controls=output)]
            print(name, visit, 're-scored', flush=True)
    if any(splits[s].keys() != recorded[s]['visits'].keys() for s in splits):
        raise ValueError('Incomplete ARKitScenes normalized cutoff analysis')
    return splits


def tum_groups(tum, recorded, scorer, paper_scorer, config):
    sequences = {}
    for name, row in tum['sequences'].items():
        if not row['eligible_windows']:
            continue
        controls = ControlRecorder(scorer, paper_scorer)
        if replay_controls(name, row, controls, config) != recorded['sequences'][name]['windows']:
            raise ValueError(f'TUM control replay differs: {name}')
        directory = DATA / name
        if directory.with_suffix('.tgz').is_file():
            directory = DATA / f'{name}_scored'
        archive = load(directory / 'pilot_scores_full_controls.json')['cases']
        calls = iter(controls.calls)
        sequences[name] = []
        for old in archive:
            window = old['video']
            expected = recorded['sequences'][name]['windows'][window]
            files = {}
            for label in ('lr', 'sync', 'moved', *(f'depth_{pose}' for pose in POSES)):
                path = directory / window / f'{label}.png'
                files[label] = dict(path=path.relative_to(directory).as_posix(), sha256=digest(path))
            x, y, _, _ = old['score_xyxy']
            selected = dict(files=files, context_origin_lr=[0, 0], score_xyxy=old['score_xyxy'],
                            candidate=dict(origin_lr=[(x - scoring.INSET) // scoring.OUTPUT_SCALE,
                                                      (y - scoring.INSET) // scoring.OUTPUT_SCALE]),
                            video=window, visit=name)
            values = rescore(directory, old, selected, scorer)
            output = {}
            for pose in POSES:
                label = f'same_frame_{pose}'
                sha, scores = next(calls)
                if sha != expected['control_pixel_sha256'][label]:
                    raise ValueError(f'TUM control pixels differ: {name}/{window}/{pose}')
                output[label] = dict(zip(MODELS, scores, strict=True))
            sequences[name].append(dict(scores=values, controls=output))
        if next(calls, None) is not None or len(sequences[name]) != row['eligible_windows']:
            raise ValueError(f'Incomplete TUM normalized cutoff analysis: {name}')
        print(name, len(sequences[name]), 'windows re-scored', flush=True)
    return sequences


def main():
    audit.cv2.setNumThreads(1)
    audit.check_roundtrip()
    _, config = scoring.verify_lock()
    paths = (ROOT / 'temporal_complete_draw_summary.json', ROOT / 'arkitscenes_temporal_control_metrics.json',
             DATA / 'static_full_control_verified.json', ROOT / 'temporal_metric_sensitivity.json')
    cohort, arkit, tum, temporal = map(load, paths)
    if arkit['scoring_lock_sha256'] != config or temporal['scoring_lock_sha256'] != config:
        raise ValueError('Lock differs from fixed-radius result')
    base = scoring.Scorer()
    matched = MatchedCutoffScorer(base)
    groups = arkit_groups(cohort, arkit['arkitscenes'], matched)
    groups['tum'] = tum_groups(tum, temporal['tum'], matched, base, config)
    margins = temporal['margins']['SHAFE']
    results = {name: {str(margin): {f'{a} - {b}': analyze(clusters, margin, 'SHAFE', a, b, name != 'tum')
                                  for a, b in CONTRASTS} for margin in margins}
               for name, clusters in groups.items()}
    report = dict(scope='Post hoc cutoff sensitivity on frozen scoring grids and restoration outputs.',
                  radius=dict(paper=PAPER_RADIUS, paper_side=FORMAL_SIDE, matched=MATCHED_RADIUS, side=WINDOW_SIDE),
                  scoring_lock_sha256=config, script_sha256=digest(Path(__file__)),
                  input_sha256={path.name: digest(path) for path in paths}, results=results, cases=groups)
    scoring.atomic_json(ROOT / 'temporal_shafe_radius_sensitivity.json', report)


if __name__ == '__main__':
    main()
