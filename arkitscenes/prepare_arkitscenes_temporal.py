"""Audit real timestamp substitutions and prepare immutable scoring inputs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from fetch_arkitscenes_frames import FrameArchive, download_frames
from probe_arkitscenes_outside_pose import read, write
from screen_arkitscenes_fixed_lr import near_pixels, OUTPUT_SCALE
from screen_arkitscenes_x3 import intrinsics, prepare_poses, project, to_world
from verify_arkitscenes_fixed_lr import get_rgb, pixels, rgb_tracks, sample


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'arkitscenes_temporal'
DESIGN = ROOT / 'arkitscenes_small_window_design.json'
SUMMARIES = [ROOT / f'arkitscenes_small_window{label}_summary.json' for label in ('', '_holdout')]
WINDOW = 32
LR_CONTEXT = 128
INSET = 20
PAD = 24
SETTINGS = dict(
    scope='Exploratory timestamp substitution at unchanged LR. Existing 15 preselected validation visits.',
    selection='At most one eligible window per visit, ordered by lag in ms then anchor time then LR origin.',
    inherited_sync_gate=True, lag_seconds=[0.05, 0.25], median_shift_hr=[2.0, 8.0],
    minimum_sparse_tracks=8, sparse_track_quadrants=4, pose_interp_gap_s=0.151,
    minimum_pose_motion='Either 0.001 m or 0.001 radians in both interpolants',
    lr_window=WINDOW, lr_inference_context=LR_CONTEXT, lr_context_edge_margin=2,
    hr_score_inset=INSET, hr_score_size=WINDOW * OUTPUT_SCALE - 2 * INSET,
    coordinate_mapping='Map both RGB frames with the synchronous wide intrinsics into one fixed LR grid, '
                       'sampling both in the same way with bicubic interpolation and centered HR pixels.',
    appearance_screen=dict(flow='DIS medium at full patch resolution, bidirectional consistency',
                           minimum_consistent_fraction=0.95, fb_error_hr_px=1.0,
                           maximum_abs_median_rgb_offset=8, maximum_p90_absolute_rgb_error=20,
                           gradient_rms_ratio=[0.67, 1.5], score_crop_used=True),
    depth_diagnostic=dict(source='FARO + camera trajectory, two interpolation variants',
                          visibility_fraction=0.99, depth_difference_m=0.03,
                          min_heldout_tracks=8, rgb_projection_p90_error_hr=1.25,
                          local_pose_disagreement_max_hr=0.75,
                          optional='Depth checks are a prespecified sensitivity subset. '
                                   'Depth may be unavailable or projection insufficiently precise.'),
    scoring=dict(methods=['fixed', 'direct', 'global_lr', 'lk_lr'],
                 alignment_context='Frozen methods estimate a translation using the 128 LR pixel context. '
                                   'Only the common 88 HR px core is scored.',
                 primary='SHAFE pairwise strict reversals from synchronous fixed-grid reference at 0.0005 margin '
                         'in both conditions, all six model pairs retained in denominator.',
                 secondary='Strict zero margin, method-specific synchronous baselines, no-motion and '
                           'capture differences, all other metrics, two independent depth/pose reference variants.',
                 bootstrap='Percentile, visit-level, 10000 draws, seed 20261122. Exploratory conditional intervals.'))


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def source_for(video):
    return ROOT / f'arkit_{video}' / 'lr_first_rgb_triplets.json'


def lock_protocol(visits):
    sources = {}
    for row in visits:
        source = source_for(row['video_id'])
        if digest(source) != row['result_sha256']:
            raise ValueError(f'Triplet inventory has changed: {source}')
        sources[row['video_id']] = digest(source)
    plan = dict(settings=SETTINGS, sources=sources,
                hashes={path.name: digest(path) for path in
                        [DESIGN, *SUMMARIES, Path(__file__), ROOT / 'score_arkitscenes_temporal.py']})
    path = OUT / 'protocol.json'
    if path.exists():
        saved = read(path)
        if {key: saved[key] for key in plan} != plan:
            raise ValueError(f'Frozen exploratory protocol has changed: {path}')
    else:
        write(path, dict(plan, locked_utc=datetime.now(timezone.utc).isoformat()))


def make_grid(origin, length):
    xy = (np.arange(length * OUTPUT_SCALE) + 0.5) / OUTPUT_SCALE - 0.5
    return pixels(origin[0] + xy, origin[1] + xy)


def rgb_remap(image, xy):
    if (xy.min() < 2 or
            (xy[..., 0] >= image.shape[1] - 2).any() or (xy[..., 1] >= image.shape[0] - 2).any()):
        raise ValueError('Unsupported cubic sampling')
    return cv2.cvtColor(sample(image, xy, cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def photo_check(a, b):
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in (a, b)]
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    engine.setFinestScale(0)
    fwd, back = (engine.calc(u, v, None) for u, v in (gray, gray[::-1]))
    grid = pixels(np.arange(a.shape[1]), np.arange(a.shape[0]))
    coords = grid + fwd
    region = np.s_[PAD + INSET:-(PAD + INSET), PAD + INSET:-(PAD + INSET)]
    points = coords[region]
    support = ((points >= 2) & (points < np.array(a.shape[:2][::-1]) - 2)).all(axis=-1)
    fb = np.linalg.norm(fwd[region] + sample(back, points), axis=-1)
    valid = support & (fb <= 1.0)
    check = dict(flow_consistent_fraction=float(valid.mean()))
    if not valid.any():
        return dict(check, passed=False)
    aligned = sample(b, coords, cv2.INTER_CUBIC)
    delta = aligned[region].astype(float) - a[region].astype(float)
    offset = float(np.abs(np.median(delta[valid], axis=0)).max())
    residual = float(np.quantile(np.abs(delta[valid]), 0.9))
    energies = []
    for rgb in (a, aligned):
        g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0.8)
        gx, gy = (cv2.Sobel(g, -1, dx, dy)[region][valid] for dx, dy in ((1, 0), (0, 1)))
        energies.append(float(np.sqrt(np.mean(gx**2 + gy**2))))
    ratio = energies[1] / max(energies[0], 1e-6)
    passed = valid.mean() >= 0.95 and offset <= 8 and residual <= 20 and 0.67 <= ratio <= 1.5
    return dict(check, max_abs_channel_median_shift=offset, absolute_rgb_error_p90=residual,
                gradient_rms_ratio=ratio, passed=bool(passed))


def pose_motion(poses, t, u):
    measurements = []
    for (ra, ca), (rb, cb) in zip(poses[t][0], poses[u][0], strict=True):
        step = float(np.linalg.norm(ca - cb))
        angle = float(Rotation.from_matrix(rb @ ra.T).magnitude())
        measurements.append(dict(translation_m=step, rotation_rad=angle))
    gap = float(max(poses[t][1], poses[u][1]))
    moved = all(r['translation_m'] >= 0.001 or r['rotation_rad'] >= 0.001 for r in measurements)
    return dict(measurements=measurements, max_trajectory_gap_seconds=gap, passed=gap <= 0.151 and moved)


def independent_depth(site, t, u, cameras, poses, grid, scale, anchor, moved):
    archive = FrameArchive(site, 'Validation', 'highres_depth')
    if not {t, u} <= archive.frames.keys():
        return dict(passed=False, reason='Missing matching depth timestamp'), {}
    manifest = download_frames(archive, (t, u))
    depths = []
    for entry in manifest['frames']:
        payload = archive.root / Path(entry['member']).name
        image = cv2.imdecode(np.frombuffer(payload.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != anchor.shape[:2] or image.dtype != np.uint16:
            raise ValueError(f'Bad depth PNG: {payload}')
        depths.append(image)
    base_depth = sample(depths[0], grid, cv2.INTER_NEAREST) * 0.001
    measurements, projected, references = [], [], {}
    for name, a, b in zip(('linear', 'spline'), poses[t][0], poses[u][0], strict=True):
        world = to_world(grid, base_depth, cameras[t], a, 0.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            xy, z = project(world, cameras[u], b, 0.5)
        valid = (base_depth > 0) & (z > 0.01) & np.isfinite(xy).all(axis=-1)
        xy[~valid] = -1000
        inside = valid & ((xy >= 2) & (xy < cameras[u][:2] - 2)).all(axis=-1)
        depth_b = sample(depths[1], xy, cv2.INTER_NEAREST) * 0.001
        good_depth = inside & (depth_b > 0) & (np.abs(depth_b - z) <= 0.03)
        visible = float(good_depth.mean())
        row = dict(visible_fraction=visible, fraction_inside=float(inside.mean()))
        if inside.all():
            motion = (xy - grid) * scale
            median = np.median(motion, axis=(0, 1))
            deviations = np.linalg.norm(motion - median, axis=-1)
            row.update(median_shift_xy_hr_px=median.tolist(), bend_max_hr_px=float(deviations.max()),
                       bend_p90_hr_px=float(np.quantile(deviations, 0.9)))
            projected.append(xy)
            references[name] = rgb_remap(moved, xy)
        measurements.append(row)
    report = dict(passed=False, source=manifest, projection=measurements)
    if len(projected) != 2:
        return report, {}
    disagreement = float(np.linalg.norm((projected[0] - projected[1]) * scale, axis=-1).max())
    tracks = rgb_tracks(anchor, moved, grid, projected, scale)
    passed = (all(row['visible_fraction'] >= 0.99 for row in measurements) and disagreement <= 0.75
              and tracks['tracks'] >= 8 and min(tracks['tracks_by_quadrant']) > 0
              and max(tracks['rgb_p90_error_hr_px']) <= 1.25)
    return dict(report, passed=passed, interpolation_disagreement_max_hr_px=disagreement,
                optical_check=tracks), references if passed else {}


def save_rgb(directory, label, image):
    path = directory / f'{label}.png'
    tmp = directory / f'{label}.tmp.png'
    Image.fromarray(image).save(tmp)
    tmp.replace(path)
    return dict(path=str(path.relative_to(OUT)), sha256=digest(path))


def prepare_visit(row):
    video, visit = row['video_id'], row['visit_id']
    src = source_for(video)
    raw = read(src)['records']
    site = src.parent
    result = dict(video=video, visit=visit, candidates=[], selected=None)
    if not raw:
        return result
    low, high = (intrinsics(site, label) for label in ('lowres_wide', 'wide'))
    trajectory = np.loadtxt(site / 'lowres_wide.traj')
    poses = prepare_poses(trajectory, {r[key] for r in raw for key in ('anchor_time', 'shifted_time')})
    order = lambda r: (round((r['shifted_time'] - r['anchor_time']) * 1000),
                       r['anchor_time'], r['origin_lr'])
    cached = {}

    def frame(asset, time, camera):
        key = (asset, time)
        if key not in cached:
            cached[key] = get_rgb(site, asset, time, camera)
        return cached[key]

    for r in sorted(raw, key=order):
        audit = dict(**r, lag_ms=order(r)[0])
        result['candidates'].append(audit)
        motion = r.get('optical_temporal_motion', {})
        if not r.get('synchronized_lr_hr', {}).get('pass'):
            audit['reason'] = 'Synchronous reference check failed'
            continue
        if (not 50 <= audit['lag_ms'] <= 250 or motion.get('tracks', 0) < 8
                or min(motion['tracks_by_quadrant']) == 0 or not 2.0 <= motion['shift_length_hr_px'] <= 8.0):
            audit['reason'] = 'Lag, shift length or sparse tracking coverage outside prespecified bounds'
            continue
        t, u = r['anchor_time'], r['shifted_time']
        audit['pose'] = pose_motion(poses, t, u)
        if not audit['pose']['passed']:
            audit['reason'] = 'Camera pose check failed'
            continue
        x, y = r['origin_lr']
        lo, sha_lr = frame('lowres_wide', t, low[t])
        a, sha_a = frame('wide', t, high[t])
        b, sha_b = frame('wide', u, high[u])
        if dict(anchor_hr=sha_a, shifted_hr=sha_b, anchor_lr=sha_lr) != r['rgb_sha256']:
            raise ValueError(f'RGB hashes disagree: {video} {t} {u}')
        rgb_grid, scale = near_pixels(make_grid((x - PAD / 4, y - PAD / 4), WINDOW + PAD // 2),
                                      low[t], high[t])
        a_photo, b_photo = (rgb_remap(image, rgb_grid) for image in (a, b))
        audit['appearance'] = photo_check(a_photo, b_photo)
        if not audit['appearance']['passed']:
            audit['reason'] = 'Photometric, focus or temporal-flow check failed'
            continue
        if result['selected'] is not None:
            audit['reason'] = 'Later eligible window within the same visit'
            continue

        width, height = low[t][:2].astype(int)
        cx = int(np.clip(x - (LR_CONTEXT - WINDOW) // 2, 2, width - LR_CONTEXT - 2))
        cy = int(np.clip(y - (LR_CONTEXT - WINDOW) // 2, 2, height - LR_CONTEXT - 2))
        grid, _ = near_pixels(make_grid((cx, cy), LR_CONTEXT), low[t], high[t])
        lr = cv2.cvtColor(lo[cy:cy + LR_CONTEXT, cx:cx + LR_CONTEXT], cv2.COLOR_BGR2RGB)
        hr_a, hr_b = (rgb_remap(image, grid) for image in (a, b))
        context = OUT / 'inputs' / video
        context.mkdir(parents=True, exist_ok=True)
        files = {label: save_rgb(context, label, image)
                 for label, image in zip(('lr', 'sync', 'moved'), (lr, hr_a, hr_b), strict=True)}
        lx, ly = OUTPUT_SCALE * (x - cx) + INSET, OUTPUT_SCALE * (y - cy) + INSET
        side = SETTINGS['hr_score_size']
        reference_grid = grid[ly:ly+side, lx:lx+side]
        try:
            depth, refs = independent_depth(site, t, u, high, poses, reference_grid, scale, a, b)
        except (OSError, KeyError, ValueError, cv2.error) as error:
            depth, refs = dict(passed=False, reason=f'{type(error).__name__}: {error}'), {}
        files.update((f'depth_{label}', save_rgb(context, f'depth_{label}', image))
                     for label, image in refs.items())
        result['selected'] = dict(video=video, visit=visit, candidate=audit,
                                  context_origin_lr=[cx, cy], score_xyxy=[lx, ly, lx + side, ly + side],
                                  files=files, depth=depth)
    return result


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    site_index = {s['video']: s for p in SUMMARIES for s in read(p)['sites']}
    visits = [dict(row, result_sha256=site_index[row['video_id']]['result_sha256'])
              for row in read(DESIGN)['eligible']]
    lock_protocol(visits)
    path = OUT / 'inputs.json'
    if path.exists():
        raise ValueError(f'The input manifest is already frozen: {path}')
    finished = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(prepare_visit, visits):
            finished.append(result)
            selected = result['selected']
            depth_ok = None if selected is None else selected['depth']['passed']
            print(result['video'], 'windows', len(result['candidates']), 'selected', selected is not None,
                  'depth projection verified', depth_ok, flush=True)
    write(path, dict(protocol_sha256=digest(OUT / 'protocol.json'), visits=finished))


if __name__ == '__main__':
    main()
