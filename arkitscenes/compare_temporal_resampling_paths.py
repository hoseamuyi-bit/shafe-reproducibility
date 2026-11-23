"""Replay the historical controls with one or two cubic RGB resampling steps."""

from pathlib import Path

import cv2
import numpy as np

import audit_arkitscenes_reference_controls as audit
import pilot_tum_temporal_reference as tum
from audit_arkitscenes_reference_offsets import HOLDOUT, POSES, RADII, REPLAY_ATOL, ROOT, load, sha
from check_temporal_shafe_radius import MatchedCutoffScorer
from check_tum_static import DATA, verify_source
from compare_temporal_reference_controls import GAP_MARGINS, apple_rows, paired_summary, reversed_pair, tum_rows
from score_arkitscenes_temporal import atomic_json
from stream_tum_temporal import read_selection


OUTPUT = ROOT / 'temporal_resampling_path_comparison.json'


def check_maps():
    audit.check_roundtrip()
    image = np.random.default_rng(8).integers(0, 256, (60, 60, 3), dtype=np.uint8)
    grid = audit.pixels(np.arange(18, 42), np.arange(18, 42))
    for shift in ((2, 3), (2.25, -3.5)):
        def inverse(xy):
            return xy - shift, {'visible': np.ones(xy.shape[:2], bool)}

        two, usable, composed = audit.roundtrip(image, grid + shift, inverse)
        sync = audit.rgb_remap(image, grid)
        one = audit.rgb_remap(image, composed)
        if (not usable.all() or not np.array_equal(one, sync)
                or np.array_equal(two, one) != (shift == (2, 3))):
            raise ValueError('Known translation did not isolate the intermediate RGB resampling')


def score_paths(scorers, sr, anchor, delayed, grid, forward, visible, inverse, archived, pose):
    two, usable, composed = audit.roundtrip(anchor, forward, inverse)
    if not (usable & visible).all():
        raise ValueError('Incomplete round-trip support in a common-support window')
    sync, one, moved = (audit.rgb_remap(image, coordinates) for image, coordinates in
                        ((anchor, grid), (anchor, composed), (delayed, forward)))
    record = dict(
        reference_pixel_sha256={label: audit.pixel_sha(image) for label, image in
                                zip(('sync', 'one', 'two', 'moved'), (sync, one, two, moved), strict=True)},
        delta_rgb_mae={label: float(np.abs(image.astype(float) - sync).mean())
                       for label, image in (('one', one), ('two', two), ('moved', moved))},
        pair_gaps={})
    for radius, scorer in scorers.items():
        baseline = archived[radius][pose]['moved']
        labels = ('sync', 'one', 'two', 'moved')
        scores = scorer.evaluate([(sr[model], target) for target in (sync, one, two, moved)
                                  for model in audit.MODELS])
        values = {label: {model: row['SHAFE'] for model, row in zip(
            audit.MODELS, scores[start:start + len(audit.MODELS)], strict=True)}
                  for label, start in zip(labels, range(0, len(scores), len(audit.MODELS)), strict=True)}
        rows = {}
        for left, right in sorted(baseline):
            expected = {'sync': baseline[left, right][0], 'moved': baseline[left, right][1],
                        'two': archived[radius][pose]['same_frame'][left, right][1]}
            actual = {label: values[label][left] - values[label][right] for label in labels}
            if any(abs(actual[label] - value) > REPLAY_ATOL for label, value in expected.items()):
                raise ValueError(f'Historical gap changed: {radius}/{pose}/{left}/{right}')
            for label in ('moved', 'two'):
                if any(reversed_pair(expected['sync'], expected[label], threshold) !=
                       reversed_pair(actual['sync'], actual[label], threshold) for threshold in GAP_MARGINS):
                    raise ValueError(f'Historical reversal changed: {radius}/{pose}/{left}/{right}')
            rows[left, right] = dict(expected, one=actual['one'])
        record['pair_gaps'][radius] = rows
    return record


def score_sr(directory, old, core):
    result = {}
    for model in audit.MODELS:
        path = audit.artifact(directory, model, old['lr_sha256'])
        image = np.ascontiguousarray(audit.read_rgb(path)[core])
        if audit.pixel_sha(image) != old['sr_pixel_sha256'][model]:
            raise ValueError(f'Frozen restoration pixels changed: {directory.name}/{old["video"]}/{model}')
        result[model] = image
    return result


def replay_apple(rows, scorers):
    sources = {row['visit']: row for row in load(HOLDOUT / 'inputs.json')['visits'] if row['selected']}
    scores = {row['visit']: row for row in load(HOLDOUT / 'scores.json')['visits']}
    receipts = {row['visit']: row for row in load(ROOT / 'holdout_online_replay_fresh' / 'verification.json')}
    results = []
    for visit, window, measures in rows:
        if any(measures['paper'][pose]['same_frame'] is None for pose in POSES):
            continue
        source = sources[visit]
        selected = source['selected']
        t, u = (selected['candidate'][key] for key in ('anchor_time', 'shifted_time'))
        evidence = next(row for row in source['candidates'] if (row['anchor_time'], row['shifted_time']) == (t, u))
        site = ROOT / 'holdout_online_replay_fresh' / f'arkit_{window}'
        if any(sha(site / name) != record['sha256'] for name, record in source['assets']['assets'].items()):
            raise ValueError(f'Calibration or trajectory changed: {visit}')
        low, wide = (audit.intrinsics(site, label) for label in ('lowres_wide', 'wide'))
        cameras = wide[t], wide[u]
        poses = audit.prepare_poses(np.loadtxt(site / 'lowres_wide.traj'), (t, u))
        anchor, delayed = (audit.frame(site, 'wide', row, time, camera) for row, time, camera in
                           zip(evidence['rgb_source']['wide']['frames'], (t, u), cameras, strict=True))
        depths = tuple(audit.frame(site, 'highres_depth', row, time, camera) for row, time, camera in
                       zip(evidence['depth_source']['frames'], (t, u), cameras, strict=True))
        x, y, ex, ey = selected['score_xyxy']
        core = np.s_[y:ey, x:ex]
        grid, _ = audit.near_pixels(audit.make_grid(selected['context_origin_lr'], audit.cohort.LR_CONTEXT)
                                    [core], low[t], wide[t])
        sync_context = audit.read_rgb(HOLDOUT / selected['files']['sync']['path'])
        if (not np.array_equal(audit.rgb_remap(anchor, grid), sync_context[core])
                or audit.pixel_sha(sync_context) != receipts[visit]['pixel_replay']['pixel_hashes']['sync']['sha256']):
            raise ValueError(f'Synchronous reference changed: {visit}')
        sr = score_sr(HOLDOUT, scores[visit], core)
        by_pose = {}
        for pose, a, b in zip(POSES, poses[t][0], poses[u][0], strict=True):
            forward, masks = audit.project_depth(grid, *depths, *cameras, a, b)
            delayed_path = HOLDOUT / selected['files']['depth_' + pose]['path']
            if not np.array_equal(audit.rgb_remap(delayed, forward), audit.read_rgb(delayed_path)):
                raise ValueError(f'Delayed reference changed: {visit}/{pose}')

            def inverse(target):
                return audit.project_depth(target, *depths[::-1], *cameras[::-1], b, a)

            by_pose[pose] = score_paths(scorers, sr, anchor, delayed, grid, forward,
                                        masks['visible'], inverse, measures, pose)
        results.append((visit, window, by_pose))
        print(f'Apple {len(results)}: {visit}', flush=True)
    return results


def replay_tum(rows, scorers):
    verified = load(DATA / 'static_full_control_recheck.json')['sequences']
    results = []
    for sequence in dict.fromkeys(unit for unit, _, _ in rows):
        raw = DATA / sequence
        archive = raw.with_suffix('.tgz')
        directory = DATA / f'{sequence}_scored' if archive.is_file() else raw
        geometry_path = directory / 'geometry_screen_full_controls.json'
        score_path = directory / 'pilot_scores_full_controls.json'
        entry = verified[sequence]
        if (sha(geometry_path), sha(score_path)) != (entry['geometry_sha256'], entry['score_sha256']):
            raise ValueError(f'TUM geometry or scores changed: {sequence}')
        geometry = load(geometry_path)
        verify_source(sequence, geometry)
        if archive.is_file():
            members, positions = tum.inventory(archive)
        else:
            _, archive, members, positions = read_selection(raw)
        requests = {int(slot): [case['times']] for slot, case in geometry['cases'].items()}
        frames = tum.selected_frames(archive, members, requests)
        scores = {row['video']: row for row in load(score_path)['cases']}
        for unit, window, measures in rows:
            if unit != sequence:
                continue
            old = scores[window]
            case = geometry['cases'][str(int(window.removeprefix('window_')))]
            t, u, p, q = case['times']
            anchor, delayed, da, db = (frames[members[asset][time]] for asset, time in
                                       (('rgb', t), ('rgb', u), ('depth', p), ('depth', q)))
            x, y, ex, ey = old['score_xyxy']
            ox, oy = tum.CONTEXT_ORIGIN
            grid = audit.pixels(np.arange(x + ox, ex + ox), np.arange(y + oy, ey + oy))
            sr = score_sr(directory, old, np.s_[y:ey, x:ex])
            by_pose = {}
            for pose, (rotate, translate) in zip(POSES, tum.pose_interpolators(positions), strict=True):
                a, b = ((rotate(time - positions[0, 0]).as_matrix().T, translate(time - positions[0, 0]))
                        for time in (t, u))
                forward, masks = tum.reproject(grid, da, db, a, b)
                if not np.array_equal(audit.rgb_remap(delayed, forward),
                                      audit.read_rgb(directory / window / f'depth_{pose}.png')):
                    raise ValueError(f'Delayed reference changed: {sequence}/{window}/{pose}')

                def inverse(target):
                    return tum.reproject(target, db, da, b, a)

                by_pose[pose] = score_paths(scorers, sr, anchor, delayed, grid, forward,
                                            masks['visible'], inverse, measures, pose)
            results.append((sequence, window, by_pose))
        print(f'TUM {sequence}: {len(scores)} windows', flush=True)
    return results


def summarize(rows, bootstrap):
    results = {}
    for radius in RADII:
        results[radius] = {}
        for pose in POSES:
            records = [(unit, window, data[pose]['pair_gaps'][radius]) for unit, window, data in rows]
            by_margin = {}
            for margin in GAP_MARGINS:
                comparisons = {}
                for a, b in (('moved', 'one'), ('moved', 'two'), ('two', 'one')):
                    paired = [(unit, window, {pair: (row['sync'], row[a]) for pair, row in gaps.items()},
                               {pair: (row['sync'], row[b]) for pair, row in gaps.items()})
                              for unit, window, gaps in records]
                    comparisons[f'{a}_vs_{b}'] = paired_summary(paired, a, b, margin, bootstrap)
                by_margin[str(margin)] = comparisons
            results[radius][pose] = by_margin
    return results


def main():
    cv2.setNumThreads(1)
    check_maps()
    apple_paths, apple = apple_rows()
    tum_paths, tum_windows = tum_rows()
    paper = audit.Scorer()
    scorers = dict(zip(RADII, (paper, MatchedCutoffScorer(paper)), strict=True))
    datasets = {}
    for name, rows in (('apple', replay_apple(apple, scorers)), ('tum', replay_tum(tum_windows, scorers))):
        historical = load(ROOT / 'temporal_reference_control_comparison.json')['datasets'][name]
        summary = summarize(rows, bootstrap=name == 'apple')
        for radius in RADII:
            for pose in POSES:
                for margin in GAP_MARGINS:
                    expected = historical[str(margin)]['contrasts'][radius][pose]['common_to_both_poses']
                    actual = summary[radius][pose][str(margin)]['moved_vs_two']
                    if actual['units'] != expected['units']:
                        raise ValueError(f'Historical paired counts changed: {name}/{radius}/{pose}/{margin}')
        cases = []
        for unit, window, data in rows:
            for pose in POSES:
                data[pose]['pair_gaps'] = {radius: [dict(models=pair, **gaps) for pair, gaps in sorted(r.items())]
                                           for radius, r in data[pose]['pair_gaps'].items()}
            cases.append(dict(unit=unit, window=window, poses=data))
        datasets[name] = dict(summary=summary, cases=cases)
    atomic_json(OUTPUT, dict(scope='Post hoc historical common-support windows; no new confirmation scores used.',
                             control='One cubic RGB sample on the anchor at the composed coordinate. The backward '
                                     'coordinate fields are linearly sampled by the forward map. Two samples use the '
                                     'discrete virtual late RGB frame and then the forward map.',
                             script_sha256=sha(Path(__file__)),
                             input_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in
                                           apple_paths + tum_paths + [ROOT / 'temporal_reference_control_comparison.json']},
                             datasets=datasets))
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    main()
