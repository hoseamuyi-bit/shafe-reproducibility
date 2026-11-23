"""Measure residual translation errors and replay reprojection on the original RGB frame."""

import argparse
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np

import expand_arkitscenes_temporal as cohort
from diagnose_arkitscenes_temporal_depth import project_depth
from prepare_arkitscenes_temporal import make_grid, rgb_remap
from screen_arkitscenes_fixed_lr import near_pixels
from screen_arkitscenes_x3 import intrinsics, prepare_poses
from verify_arkitscenes_fixed_lr import pixels, rgb_tracks, sample
from verify_arkitscenes_temporal_acquisition import frame
from score_arkitscenes_temporal import MODELS, Scorer, artifact, pixel_sha, read as read_rgb


MARGIN = 0.0005
METHODS = ('fixed', 'direct', 'global_lr', 'lk_lr')
MAX_RGB_BIAS_HR = 1.25


def quantiles(values):
    return dict(p50=float(np.median(values)), p90=float(np.quantile(values, 0.9)),
                max=float(np.max(values)))


def roundtrip(anchor, forward, inverse):
    """Render the anchor on the later pixel lattice, then sample the reported forward map."""
    # Keep the full interpolation stencil, including when OpenCV rounds the phase up.
    corner = np.floor(forward.min(axis=(0, 1))).astype(int) - 3
    end = np.ceil(forward.max(axis=(0, 1))).astype(int) + 4
    target = pixels(np.arange(corner[0], end[0]), np.arange(corner[1], end[1]))
    backward, masks = inverse(target)
    local = forward - corner
    integer = np.floor(local).astype(int)
    usable = np.ones(forward.shape[:2], dtype=bool)
    for dx, dy in product(range(-1, 4), repeat=2):
        x, y = (integer + (dx, dy)).transpose(2, 0, 1)
        usable &= masks['visible'][y, x]
    reverse_at_forward = sample(backward.astype(np.float32), local)
    if not usable.all():
        return None, usable, reverse_at_forward
    source = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
    virtual = sample(source, backward, cv2.INTER_CUBIC)
    return sample(virtual, local, cv2.INTER_CUBIC), usable, reverse_at_forward


def ranks(reference, scores):
    pairs = []
    for a, b in combinations(MODELS, 2):
        g = reference[a]['SHAFE'] - reference[b]['SHAFE']
        h = scores[a]['SHAFE'] - scores[b]['SHAFE']
        if not np.isfinite([g, h]).all():
            raise ValueError('Non-finite SHAFE gap')
        pairs.append(dict(models=[a, b], sync_gap=g, control_gap=h,
                          reversal=abs(g) > MARGIN and abs(h) > MARGIN and g * h < 0))
    return dict(pairs=pairs, reversals=sum(row['reversal'] for row in pairs))


def check_rgb_bias(start, stop, grid, forward, scale):
    """Estimate a residual translation and test it on spatially held-out RGB tracks."""
    on_grid = ((start - grid[0, 0]) * scale).reshape(-1, 1, 2)
    predicted = sample(forward.astype(np.float32), on_grid).reshape(-1, 2)
    residual = (stop - predicted) * scale
    center = (np.ceil(grid[0, 0]) + np.floor(grid[-1, -1])) * 0.5
    quadrant = (start >= center).astype(int) @ [1, 2]
    counts = np.bincount(quadrant, minlength=4)
    before = np.linalg.norm(residual, axis=1)
    bias = np.median(residual, axis=0)
    report = dict(tracks_by_quadrant=counts.tolist(), before_hr_px=quantiles(before),
                  fitted_bias_xy_hr_px=bias.tolist(), held_out_hr_px=None, eligible=False)
    if len(start) < 8 or counts.min() == 0:
        return report, bias
    holdout = np.empty(len(start))
    for group in range(4):
        test = quadrant == group
        fit = np.median(residual[~test], axis=0)
        holdout[test] = np.linalg.norm(residual[test] - fit, axis=1)
    report['held_out_hr_px'] = quantiles(holdout)
    report['eligible'] = bool(report['held_out_hr_px']['p90'] < report['before_hr_px']['p90']
                              and np.linalg.norm(bias) <= MAX_RGB_BIAS_HR)
    return report, bias


def run_case(directory, source, score, scorer):
    selected = source['selected']
    video = selected['video']
    candidate = selected['candidate']
    t, u = (candidate[key] for key in ('anchor_time', 'shifted_time'))
    evidence = next(row for row in source['candidates'] if (row['anchor_time'], row['shifted_time']) == (t, u))
    site = directory / f'arkit_{video}'
    for name, record in source['assets']['assets'].items():
        if cohort.digest(site / name) != record['sha256']:
            raise ValueError(f'Geometry asset changed: {video}/{name}')
    low, wide = (intrinsics(site, label) for label in ('lowres_wide', 'wide'))
    p = prepare_poses(np.loadtxt(site / 'lowres_wide.traj'), (t, u))
    cameras = wide[t], wide[u]
    images = []
    for asset, records in (('wide', evidence['rgb_source']['wide']), ('highres_depth', evidence['depth_source'])):
        images.append(tuple(frame(site, asset, item, time, camera) for item, time, camera
                            in zip(records['frames'], (t, u), cameras, strict=True)))
    (anchor, moved), depths = images
    paths = {label: directory / rec['path'] for label, rec in selected['files'].items()}
    for label, path in paths.items():
        if cohort.digest(path) != selected['files'][label]['sha256']:
            raise ValueError(f'Scoring image changed: {video}/{label}')
    lx, ly, ex, ey = selected['score_xyxy']
    if score['score_xyxy'] != [lx, ly, ex, ey]:
        raise ValueError('Support differs between scoring and diagnostics')
    core = np.s_[ly:ey, lx:ex]
    grid, scale = near_pixels(make_grid(selected['context_origin_lr'], cohort.LR_CONTEXT)[core], low[t], wide[t])
    sync = rgb_remap(anchor, grid)
    if not np.array_equal(sync, read_rgb(paths['sync'])[core]):
        raise ValueError('Synchronous core reconstruction differs')
    lr_sha = selected['files']['lr']['sha256']
    if score['lr_sha256'] != lr_sha:
        raise ValueError('Synchronous LR has changed')
    sr = {}
    for model in MODELS:
        path = artifact(directory, model, lr_sha)
        if cohort.digest(path) != score['checks']['sr_sha256'][model]:
            raise ValueError(f'SR image changed: {video}/{model}')
        sr[model] = np.ascontiguousarray(read_rgb(path)[core])
        if pixel_sha(sr[model]) != score['sr_pixel_sha256'][model]:
            raise ValueError(f'SR core changed: {video}/{model}')
    baseline = {model: score['scores']['sync'][model]['fixed'] for model in MODELS}
    tracks = rgb_tracks(anchor, moved, grid, (grid,), scale)
    start, stop = (np.array(tracks[label]) for label in ('positions_xy_source', 'positions_xy_target'))

    results = {}
    for name, a, b in zip(cohort.POSE_NAMES, p[t][0], p[u][0], strict=True):
        forward, masks = project_depth(grid, *depths, *cameras, a, b)
        if not np.array_equal(rgb_remap(moved, forward), read_rgb(paths['depth_' + name])):
            raise ValueError(f'Frozen alternate reference reconstruction differs: {video}/{name}')
        valid = masks['visible']
        shift = ((forward - grid) * scale)[valid]
        local_median = np.median(shift, axis=0)
        # A correction translates image content by +action and samples HR at output minus action.
        residual = {method: {model: quantiles(np.linalg.norm(
            shift + score['actions']['moved'][model][method], axis=1)) for model in MODELS}
            for method in METHODS}
        # The median flow is a descriptive local translation, calculated using the depth reference.
        residual['median_flow'] = quantiles(np.linalg.norm(shift - local_median, axis=1))
        bias_report, bias = check_rgb_bias(start, stop, grid, forward, scale)

        def inverse(target):
            return project_depth(target, *depths[::-1], *cameras[::-1], b, a)

        control, usable, reverse = roundtrip(anchor, forward, inverse)
        common = usable & valid
        data = dict(translation_residual_px=residual, median_shift_px=local_median.tolist(),
                    forward_visible_fraction=float(valid.mean()),
                    roundtrip_visible_fraction=float(common.mean()),
                    roundtrip_geometry_px=quantiles(np.linalg.norm(((reverse - grid) * scale)[common], axis=1))
                    if common.any() else None, same_frame=None,
                    moved_over_sync=score['capture_delta']['depth_' + name],
                    moved_rank=ranks(baseline, {m: score['scores']['moved'][m]['depth_' + name] for m in MODELS}))
        references = dict(local_translation=rgb_remap(moved, grid + local_median / scale))
        if control is not None and valid.all():
            references['same_frame'] = control
        if bias_report['eligible']:
            references['rgb_bias_corrected'] = rgb_remap(moved, forward + bias / scale)
        data['rgb_bias_check'] = bias_report
        for label, reference in references.items():
            compared = scorer.evaluate([(sync, reference), *((sr[m], reference) for m in MODELS)])
            data[label] = dict(over_sync=compared[0],
                               ranking=ranks(baseline, dict(zip(MODELS, compared[1:], strict=True))))
        results[name] = data
    return dict(video=video, visit=score['visit'], poses=results)


def check_roundtrip():
    rng = np.random.default_rng(3)
    source = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    core = pixels(np.arange(40, 60), np.arange(40, 60))
    for shift in ((0, 0), (-3, 4), (1.25, -2.625)):
        control, usable, back = roundtrip(source, core + shift,
                                         lambda xy: (xy - shift, {'visible': np.ones(xy.shape[:2], bool)}))
        expected = source[40:60, 40:60, ::-1]
        if not usable.all() or not np.array_equal(back, core):
            raise ValueError('Round-trip translation failed at known coordinates')
        if np.array_equal(control, expected) != all(float(v).is_integer() for v in shift):
            raise ValueError('Round-trip interpolation failed its integer vs subpixel check')
    start = np.array([[x, y] for x in (41, 44, 55, 57) for y in (42, 46, 51, 58)])
    flow, bias = np.array([3., -4.]), np.array([0.5, -0.75])
    report, fitted = check_rgb_bias(start, start + flow + bias, core, core + flow, np.ones(2))
    if not report['eligible'] or report['held_out_hr_px']['p90'] != 0 or not np.array_equal(fitted, bias):
        raise ValueError('Held-out RGB residual failed its known bias check')


def run(directories):
    cv2.setNumThreads(1)
    check_roundtrip()
    scorer = Scorer()
    for directory in directories:
        inputs_path, scores_path = (directory / name for name in ('inputs.json', 'scores.json'))
        inputs, scores = (cohort.read(path) for path in (inputs_path, scores_path))
        input_hash = cohort.digest(inputs_path)
        plan_hash = cohort.digest(directory / 'protocol.json')
        if scores['inputs_sha256'] != input_hash or scores['protocol_sha256'] != plan_hash:
            raise ValueError(f'Frozen score provenance has changed: {directory}')
        visits = {row['visit']: row for row in inputs['visits'] if row['selected']}
        if visits.keys() != {row['visit'] for row in scores['visits']}:
            raise ValueError(f'Scoring and selection manifests differ: {directory}')
        results = [run_case(directory, visits[score['visit']], score, scorer) for score in scores['visits']]
        cohort.write(directory / 'reference_control_replay_v2.json', dict(
            inputs_sha256=input_hash, scores_sha256=cohort.digest(scores_path), protocol_sha256=plan_hash,
            script_sha256=cohort.digest(Path(__file__)), margin=MARGIN, visits=results))
        for name in cohort.POSE_NAMES:
            rows = [result['poses'][name] for result in results]
            usable = [row for row in rows if row['same_frame']]
            print(directory.name, name, f'{len(usable)}/{len(rows)} fully supported round trips',
                  'same-frame / moved-frame reversals on those cases',
                  sum(row['same_frame']['ranking']['reversals'] for row in usable),
                  sum(row['moved_rank']['reversals'] for row in usable), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    run(parser.parse_args().directories)
