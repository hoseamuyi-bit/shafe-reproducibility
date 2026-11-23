"""Compare the prior alignment methods and emulate one RGB interpolation on the new captures."""

from pathlib import Path

import cv2
import numpy as np

import audit_arkitscenes_reference_controls as audit
from check_temporal_shafe_radius import MatchedCutoffScorer, rescore
from compare_temporal_reference_controls import GAP_MARGINS, paired_summary, score_gaps
from compare_temporal_resampling_paths import check_maps, score_paths, score_sr
from run_arkitscenes_confirmation import OUT, RADIUS_MATCHED
from score_arkitscenes_temporal import atomic_json
from verify_confirmation_statistics import digest, main as verify, read


RADIUS_KEYS = ('50', str(RADIUS_MATCHED))


def score_same_frame(directory, source, row, gaps, scorers):
    selected = source['selected']
    t, u = (selected['candidate'][key] for key in ('anchor_time', 'shifted_time'))
    record = next(pair for pair in source['candidates']
                  if (pair['anchor_time'], pair['shifted_time']) == (t, u))
    site = directory / f'arkit_{row["video"]}'
    for name, entry in source['assets']['assets'].items():
        if digest(site / name) != entry['sha256']:
            raise ValueError(f'Geometry asset differs: {site.name}/{name}')
    low, wide = (audit.intrinsics(site, asset) for asset in ('lowres_wide', 'wide'))
    cameras = wide[t], wide[u]
    poses = audit.prepare_poses(np.loadtxt(site / 'lowres_wide.traj'), (t, u))
    anchor, delayed = (audit.frame(site, 'wide', entry, time, camera)
                       for entry, time, camera in zip(record['rgb_source']['wide']['frames'], (t, u), cameras, strict=True))
    depths = tuple(audit.frame(site, 'highres_depth', entry, time, camera)
                   for entry, time, camera in zip(record['depth_source']['frames'], (t, u), cameras, strict=True))
    x, y, ex, ey = selected['score_xyxy']
    core = np.s_[y:ey, x:ex]
    grid, _ = audit.near_pixels(audit.make_grid(selected['context_origin_lr'], audit.cohort.LR_CONTEXT)
                                [core], low[t], wide[t])
    sr = score_sr(directory, row, core)
    archived = {radius: {pose: {label: {(a, b): (sync, alternate) for a, b, sync, alternate in case[label]}
                                for label in ('moved', 'same_frame')}
                         for pose, case in gaps[radius].items()} for radius in RADIUS_KEYS}
    results = {}
    for pose, first, second in zip(audit.cohort.POSE_NAMES, poses[t][0], poses[u][0], strict=True):
        forward, masks = audit.project_depth(grid, *depths, *cameras, first, second)

        def inverse(target):
            return audit.project_depth(target, *depths[::-1], *cameras[::-1], second, first)

        detail = score_paths(scorers, sr, anchor, delayed, grid, forward, masks['visible'], inverse, archived, pose)
        if detail['reference_pixel_sha256']['two'] != gaps['50'][pose]['same_frame_pixel_sha256']:
            raise ValueError(f'Two-step reference pixels changed: {row["visit"]}/{pose}')
        results[pose] = detail
    return results


def direct_vs_lk(scores):
    models = audit.MODELS
    sync = {model: scores['sync'][model]['fixed']['SHAFE'] for model in models}
    return {method: score_gaps(sync, {model: scores['moved'][model][method]['SHAFE'] for model in models})
            for method in ('direct', 'lk_lr')}


def align_summary(cases):
    summary = {}
    for radius in RADIUS_KEYS:
        comparisons = [(visit, visit, scores[radius]['direct'], scores[radius]['lk_lr'])
                       for visit, scores in cases.items()]
        summary[radius] = {str(margin): paired_summary(comparisons, 'direct', 'lk_lr', margin, True)
                           for margin in GAP_MARGINS} if len(comparisons) >= 2 else {}
    return summary


def same_frame_summary(cases, primary):
    results = {}
    for radius in RADIUS_KEYS:
        results[radius] = {}
        for pose in audit.cohort.POSE_NAMES:
            rows = [(visit, visit, case[pose]['pair_gaps'][radius]) for visit, case in cases.items()]
            results[radius][pose] = {}
            for margin in GAP_MARGINS if len(rows) >= 2 else ():
                comparisons = {}
                for a, b in (('moved', 'one'), ('moved', 'two'), ('two', 'one')):
                    paired = [(visit, video, {pair: (row['sync'], row[a]) for pair, row in gaps.items()},
                               {pair: (row['sync'], row[b]) for pair, row in gaps.items()})
                              for visit, video, gaps in rows]
                    comparisons[f'{a}_vs_{b}'] = paired_summary(paired, a, b, margin, True)
                expected = primary['results'][str(margin)][radius][pose]
                if comparisons['moved_vs_two']['units'] != expected['units']:
                    raise ValueError(f'Two-step primary counts changed: {radius}/{pose}/{margin}')
                results[radius][pose][str(margin)] = comparisons
    return results


def run(directory):
    verify(directory)
    cv2.setNumThreads(1)
    check_maps()
    primary = read(directory, 'confirmation_results')
    selected = {row['visit']: row for row in read(directory, 'inputs')['visits'] if row['selected']}
    rows = read(directory, 'scores')['visits'] if selected else []
    paper = audit.Scorer()
    scorers = dict(zip(RADIUS_KEYS, (paper, MatchedCutoffScorer(paper)), strict=True))
    alignments, references = {}, {}
    for row in rows:
        visit = row['visit']
        source = selected[visit]
        matched = rescore(directory, row, source['selected'], scorers[RADIUS_KEYS[1]])
        alignments[visit] = dict(zip(RADIUS_KEYS, (direct_vs_lk(row['scores']), direct_vs_lk(matched)), strict=True))
        if visit in primary['fully_supported_visits']:
            references[visit] = score_same_frame(directory, source, row, primary['pair_gaps'][visit], scorers)
        print(f'Secondary analyses replayed {visit}', flush=True)
    if set(references) != set(primary['fully_supported_visits']):
        raise ValueError('Incomplete single-interpolation analysis')
    report = dict(
        scope='Secondary analyses defined during acquisition before accessing the new SHAFE rankings.',
        confirmation_sha256=digest(directory / 'confirmation_results.json'),
        analysis_source_sha256=digest(Path(__file__)),
        alignment=align_summary(alignments),
        resampling=same_frame_summary(references, primary))
    for case in alignments.values():
        for radius, methods in case.items():
            case[radius] = {method: [[*pair, *gap] for pair, gap in sorted(rows.items())]
                            for method, rows in methods.items()}
    for case in references.values():
        for pose in case.values():
            pose['pair_gaps'] = {radius: [dict(models=list(pair), **row) for pair, row in sorted(rows.items())]
                                 for radius, rows in pose['pair_gaps'].items()}
    report.update(alignment_pair_gaps=alignments, resampling_pair_gaps=references)
    atomic_json(directory / 'secondary_results.json', report)


if __name__ == '__main__':
    run(OUT)
