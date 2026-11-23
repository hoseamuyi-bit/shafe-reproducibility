"""Download and screen new ARKitScenes visits for depth-verified temporal pairs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from diagnose_arkitscenes_temporal_depth import load_depth, project_depth
from fetch_arkitscenes_cohort import download_video
from fetch_arkitscenes_frames import FrameArchive, atomic_write, download_frames
from prepare_arkitscenes_temporal import (INSET, LR_CONTEXT, OUT as PILOT, OUTPUT_SCALE, PAD, ROOT,
                                          SETTINGS, WINDOW, digest, make_grid, photo_check,
                                          pose_motion, rgb_remap)
from probe_arkitscenes_outside_pose import read, write
from screen_arkitscenes_fixed_lr import near_pixels
from screen_arkitscenes_x3 import intrinsics, prepare_poses, project, to_world
from verify_arkitscenes_fixed_lr import exact_time_check, pixels, rgb_tracks


OUT = ROOT / 'arkitscenes_temporal_faro'
MAX_VISITS = 24
MAX_DEPTH_PAIRS = 24
MAX_RGB_PAIRS = 10
DEPTH_HYPOTHESES_M = (0.6, 0.9, 1.25, 1.8, 2.5, 3.5, 5.0, 7.5)
ORIGINS = tuple(product(range(16, 225, 16), range(16, 145, 16)))
ASSETS = ('wide_intrinsics.zip', 'lowres_wide_intrinsics.zip', 'lowres_wide.traj')
POSE_NAMES = ('linear', 'spline')
CORE = SETTINGS['hr_score_size']


def freeze():
    path = OUT / 'protocol.json'
    ranking = ROOT / 'arkitscenes_x4_download_priority.json'
    index = ROOT / 'arkitscenes_upsampling_split.csv'
    pilot = PILOT / 'protocol.json'
    dependencies = (Path(__file__), ROOT / 'fetch_arkitscenes_frames.py', ROOT / 'fetch_arkitscenes_cohort.py',
                    ROOT / 'diagnose_arkitscenes_temporal_depth.py', ROOT / 'verify_arkitscenes_fixed_lr.py',
                    ROOT / 'prepare_arkitscenes_temporal.py', ROOT / 'score_arkitscenes_temporal.py',
                    ROOT / 'screen_arkitscenes_x3.py', ROOT / 'screen_arkitscenes_fixed_lr.py')
    settings = dict(SETTINGS, scope='Depth-first temporal substitution on separate validation visits.',
                    selection='At most one geometry- and RGB-qualified window per visit in the frozen screen order.')
    rules = dict(settings=settings, pilot_protocol_sha256=digest(pilot),
                 index_sha256=digest(index), source_rank_sha256=digest(ranking),
                 hashes={file.name: digest(file) for file in dependencies},
                 expansion=dict(max_visits=MAX_VISITS, max_depth_pairs_per_visit=MAX_DEPTH_PAIRS,
                                max_rgb_pairs_per_visit=MAX_RGB_PAIRS, origins_lr_px=[list(p) for p in ORIGINS],
                                prior_depth_hypotheses_m=list(DEPTH_HYPOTHESES_M), prior_center_shift_hr=[2.25, 7.75],
                                prior_center_interpolation_disagreement_hr=0.60,
                                visit_order='Public validation visits in the archived geometry-prior order. '
                                            'Exclude pilot visit IDs and preexisting wide RGB screens for the video.',
                                pair_order='Descending count of plausible window centers, then lag and timestamps. '
                                           'Candidate pairs are fixed from intrinsics and poses before loading depth.',
                                rgb_order='Retain this pair order and grid origin order after dense depth screening. '
                                          'Stop after one eligible window or ten depth-qualified pairs.',
                                depth_geometry='Existing 100% projectability, 99% depth agreement, two pose '
                                               'interpolants within 0.75 HR px; also require a 2–8 HR px median '
                                               'geometric shift for both interpolants.',
                                rgb_geometry='Existing selected-window sync, photometric and tracking gates. '
                                             'Depth residual is validated on the 88-pixel core.'))
    if path.exists():
        saved = read(path)
        if any(saved[key] != value for key, value in rules.items()):
            raise ValueError('Frozen expansion protocol has changed')
        return saved
    sources = read(ranking)
    if sources['index_sha256'] != digest(index):
        raise ValueError('Public validation index has changed')
    excluded_visits = {row['visit'] for row in read(PILOT / 'inputs.json')['visits']}
    all_rows, exclusions, visits = [], [], set()
    for row in sources['results']:
        video, visit = row['video_id'], row['visit_id']
        site = ROOT / f'arkit_{video}'
        if visit in excluded_visits or (site / 'wide_frames.json').exists():
            exclusions.append(dict(video_id=video, visit_id=visit, reason='Prior pilot or RGB screen'))
            continue
        if visit not in visits and row['fold'] == 'Validation':
            visits.add(visit)
            all_rows.append(row)
    plan = dict(rules, excluded=exclusions, candidates=all_rows[:MAX_VISITS],
                locked_utc=datetime.now(timezone.utc).isoformat())
    write(path, plan)
    return plan


def plan_pairs(low, wide, poses, times):
    pairs = []
    origins = np.array(ORIGINS)
    hypotheses = np.array(DEPTH_HYPOTHESES_M)
    for i, t in enumerate(times):
        if t not in low:
            continue
        neighbors = times[i+1:np.searchsorted(times, t + 0.251)]
        centers, scale = near_pixels(origins + (WINDOW - 1) / 2, low[t], wide[t])
        starts = np.broadcast_to(centers[:, None, :], (len(centers), len(hypotheses), 2))
        depths = np.broadcast_to(hypotheses, starts.shape[:-1])
        worlds = [to_world(starts, depths, wide[t], pose, 0.5) for pose in poses[t][0]]
        for u in neighbors:
            lag = round((u - t) * 1000)
            if not 50 <= lag <= 250 or not pose_motion(poses, t, u)['passed']:
                continue
            guesses = []
            for world, pose in zip(worlds, poses[u][0], strict=True):
                with np.errstate(divide='ignore', invalid='ignore'):
                    xy, _ = project(world, wide[u], pose, 0.5)
                guesses.append((xy - starts) * scale)
            distances = np.linalg.norm(guesses, axis=-1)
            plausible = ((distances >= 2.25) & (distances <= 7.75)).all(axis=0)
            plausible &= np.linalg.norm(guesses[0] - guesses[1], axis=-1) <= 0.60
            count = int(plausible.any(axis=1).sum())
            if count:
                pairs.append(dict(anchor_time=float(t), shifted_time=float(u), lag_ms=lag,
                                  plausible_centers=count))
    return sorted(pairs, key=lambda row: (-row['plausible_centers'], row['lag_ms'],
                                          row['anchor_time'], row['shifted_time']))


def geometry(depth_a, depth_b, low, camera_a, camera_b, poses_a, poses_b):
    offset = (np.arange(INSET, INSET + CORE) + 0.5) / OUTPUT_SCALE - 0.5
    start_grid = pixels(offset, offset)
    grids, scale = near_pixels(np.array(ORIGINS)[:, None, None, :] + start_grid, low, camera_a)
    # remap only takes a 2-D array of output locations. Stack the windows vertically.
    tall = grids.reshape(-1, CORE, 2)
    measurements, maps = [], []
    stages = dict(windows=len(grids))
    qualified = np.ones(len(grids), dtype=bool)
    for name, a, b in zip(POSE_NAMES, poses_a, poses_b, strict=True):
        xy, masks = project_depth(tall, depth_a, depth_b, camera_a, camera_b, a, b)
        visible = masks['visible'].reshape(-1, CORE, CORE).mean(axis=(1, 2))
        supported = ~(masks['missing_source'] | masks['invalid_projection'] | masks['outside'])
        supported = supported.reshape(-1, CORE, CORE).all(axis=(1, 2))
        maps.append(xy.reshape(grids.shape))
        motion = (maps[-1] - grids) * scale
        median = np.median(motion, axis=(1, 2))
        lengths = np.linalg.norm(median, axis=-1)
        accepted = supported & (visible >= 0.99) & (lengths >= 2) & (lengths <= 8)
        measurements.append((visible, median, lengths))
        stages[name] = dict(fully_projectable=int(supported.sum()), visibility_gate=int((supported & (visible >= 0.99)).sum()),
                            visibility_and_shift_gates=int(accepted.sum()), max_visibility=float(visible.max()))
        qualified &= accepted
    disagreement = np.linalg.norm((maps[0] - maps[1]) * scale, axis=-1).max(axis=(1, 2))
    stages['before_pose_agreement'] = int(qualified.sum())
    qualified &= disagreement <= 0.75
    stages['passed'] = int(qualified.sum())
    candidates = []
    for i in np.flatnonzero(qualified):
        record = dict(origin_lr=ORIGINS[i], interpolation_disagreement_max_hr_px=float(disagreement[i]),
                      projection={name: dict(visible_fraction=float(visible[i]),
                                             median_shift_xy_hr_px=median[i].tolist(),
                                             median_shift_length_hr_px=float(lengths[i]))
                                  for name, (visible, median, lengths) in zip(POSE_NAMES, measurements, strict=True)})
        candidates.append((record, grids[i], [xy[i] for xy in maps], scale))
    return stages, candidates


def save_inputs(video, visit, t, u, low, wide, colors, candidate, depth, projections):
    x, y = candidate['origin_lr']
    width, height = low[:2].astype(int)
    cx = int(np.clip(x - (LR_CONTEXT - WINDOW) // 2, 2, width - LR_CONTEXT - 2))
    cy = int(np.clip(y - (LR_CONTEXT - WINDOW) // 2, 2, height - LR_CONTEXT - 2))
    lr, a, b = colors
    lr = cv2.cvtColor(lr[cy:cy + LR_CONTEXT, cx:cx + LR_CONTEXT], cv2.COLOR_BGR2RGB)
    grid, _ = near_pixels(make_grid((cx, cy), LR_CONTEXT), low, wide)
    images = dict(lr=lr, sync=rgb_remap(a, grid), moved=rgb_remap(b, grid))
    images.update((f'depth_{name}', rgb_remap(b, coords))
                  for name, coords in zip(POSE_NAMES, projections, strict=True))
    folder = OUT / 'inputs' / video
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, image in images.items():
        path = folder / f'{name}.png'
        partial = folder / f'{name}.part.png'
        Image.fromarray(image).save(partial)
        partial.replace(path)
        files[name] = dict(path=path.relative_to(OUT).as_posix(), sha256=digest(path))
    lx, ly = OUTPUT_SCALE * (x - cx) + INSET, OUTPUT_SCALE * (y - cy) + INSET
    return dict(video=video, visit=visit, candidate=dict(candidate, anchor_time=t, shifted_time=u),
                context_origin_lr=[cx, cy], score_xyxy=[lx, ly, lx + CORE, ly + CORE],
                files=files, depth=dict(depth, passed=True))


def read_rgb(archive, entry, camera):
    path = archive.root / Path(entry['member']).name
    if digest(path) != entry['sha256']:
        raise ValueError(f'RGB SHA-256 mismatch: {path}')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != tuple(camera[:2][::-1].astype(int)):
        raise ValueError(f'Invalid RGB PNG: {path}')
    return image


def screen_visit(row, result):
    video, visit = row['video_id'], row['visit_id']
    site = ROOT / f'arkit_{video}'
    # The cohort downloader writes provenance in the supplied root. Keep the
    # current expansion separate from prior cohorts and their frozen manifests.
    site, archive_records = acquire_geometry(row, site)
    result['assets'] = archive_records
    low, wide = (intrinsics(site, label) for label in ('lowres_wide', 'wide'))
    trajectory = np.loadtxt(site / 'lowres_wide.traj')
    depth_archive = FrameArchive(site, 'Validation', 'highres_depth')
    depth_times = depth_archive.frames.keys() & wide.keys()
    lower, upper = trajectory[[0, -1], 0]
    observations = sorted(t for t in depth_times if lower <= t <= upper)
    anchors = set(observations) & low.keys()
    result['paired_depth_and_intrinsics_timestamps'] = len(observations)
    result['possible_anchor_timestamps'] = len(anchors)
    if not anchors:
        return
    poses = prepare_poses(trajectory, observations)
    # A target needs wide intrinsics and depth but no synchronized LR. Limit
    # only anchor timestamps when constructing the temporal pairs.
    pairs = plan_pairs(low, wide, poses, np.array(observations))
    result['prior_eligible_pairs'] = len(pairs)
    result['pair_plan'] = pairs[:MAX_DEPTH_PAIRS]
    stage = OUT / 'visit_plans' / (video + '.json')
    if stage.exists() and read(stage) != result['pair_plan']:
        raise ValueError(f'Frozen temporal pair plan changed: {video}')
    write(stage, result['pair_plan'])

    @lru_cache(maxsize=8)
    def depth(time):
        entry = depth_archive.extract(time)
        return entry, load_depth(depth_archive, entry, wide[time])

    wide_archive, low_archive = None, None
    result['rgb_pairs_downloaded'] = 0
    for proposal in result['pair_plan']:
        t, u = proposal['anchor_time'], proposal['shifted_time']
        audit = dict(proposal, rgb_windows=[])
        result['candidates'].append(audit)
        (entry_a, depth_a), (entry_b, depth_b) = depth(t), depth(u)
        audit['depth_source'] = dict(url=depth_archive.url, etag=depth_archive.etag,
                                     zip_tail_sha256=depth_archive.tail_sha256, frames=[entry_a, entry_b])
        audit['depth_counts'], candidates = geometry(depth_a, depth_b, low[t], wide[t], wide[u],
                                                    poses[t][0], poses[u][0])
        if not candidates:
            continue
        if result['rgb_pairs_downloaded'] >= MAX_RGB_PAIRS:
            audit['reason'] = 'RGB pair budget reached'
            continue
        if wide_archive is None:
            wide_archive = FrameArchive(site, 'Validation', 'wide')
            low_archive = FrameArchive(site, 'Validation', 'lowres_wide')
        if not {t, u} <= wide_archive.frames.keys() or t not in low_archive.frames:
            audit['reason'] = 'Matching public RGB timestamps unavailable'
            continue
        downloads = dict(wide=download_frames(wide_archive, (t, u)), low=download_frames(low_archive, (t,)))
        audit['rgb_source'] = downloads
        result['rgb_pairs_downloaded'] += 1
        lr = read_rgb(low_archive, downloads['low']['frames'][0], low[t])
        a, b = [read_rgb(wide_archive, frame, wide[frame['timestamp']]) for frame in downloads['wide']['frames']]
        for record, grid, projections, scale in candidates:
            x, y = record['origin_lr']
            window = dict(record)
            audit['rgb_windows'].append(window)
            window['sync'] = exact_time_check((x, y), WINDOW, lr, a, low[t], wide[t])
            if not window['sync']['pass']:
                continue
            wide_grid, _ = near_pixels(make_grid((x, y), WINDOW), low[t], wide[t])
            motion = rgb_tracks(a, b, wide_grid, (wide_grid,), scale)
            window['rgb_motion'] = {key: motion.get(key) for key in
                                    ('tracks', 'tracks_by_quadrant', 'median_observed_rgb_shift_xy_hr_px')}
            if (motion['tracks'] < 8 or min(motion['tracks_by_quadrant']) == 0 or
                    not 2 <= np.linalg.norm(motion['median_observed_rgb_shift_xy_hr_px']) <= 8):
                continue
            optical = rgb_tracks(a, b, grid, projections, scale)
            window['optical_check'] = {key: optical.get(key) for key in
                                       ('tracks', 'tracks_by_quadrant', 'rgb_p90_error_hr_px',
                                        'rgb_max_error_hr_px')}
            if (optical['tracks'] < 8 or min(optical['tracks_by_quadrant']) == 0 or
                    max(optical['rgb_p90_error_hr_px']) > 1.25):
                continue
            photo_grid, _ = near_pixels(make_grid((x - PAD / 4, y - PAD / 4), WINDOW + PAD // 2),
                                        low[t], wide[t])
            window['appearance'] = photo_check(rgb_remap(a, photo_grid), rgb_remap(b, photo_grid))
            if not window['appearance']['passed']:
                continue
            window['passed'] = True
            result['selected'] = save_inputs(video, visit, t, u, low[t], wide[t], (lr, a, b), window,
                                             dict(window, source=audit['depth_source']), projections)
            return


def acquire_geometry(row, existing):
    site = OUT / existing.name
    site.mkdir(exist_ok=True)
    for name in ASSETS:
        source, dest = existing / name, site / name
        if source.exists() and not dest.exists():
            atomic_write(dest, source.read_bytes())
    identity = {key: row[key] for key in ('video_id', 'visit_id', 'fold')}
    records = download_video(OUT, identity, ASSETS)
    for name, evidence in row['assets'].items():
        if records['assets'][name]['sha256'] != evidence['sha256']:
            raise ValueError(f'Archived geometry source changed: {existing.name}/{name}')
    return site, records


def run_visit(row, protocol_sha):
    video = row['video_id']
    path = OUT / 'screened' / (video + '.json')
    if path.exists():
        previous = read(path)
        if previous['protocol_sha256'] != protocol_sha:
            raise ValueError(f'Visit belongs to a different protocol: {video}')
        if 'error' not in previous:
            return previous
    result = dict(protocol_sha256=protocol_sha, video=video, visit=row['visit_id'],
                  candidates=[], selected=None)
    try:
        screen_visit(row, result)
    except (OSError, ValueError, KeyError, cv2.error, subprocess.CalledProcessError) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    write(path, result)
    return result


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    (OUT / 'screened').mkdir(exist_ok=True)
    (OUT / 'visit_plans').mkdir(exist_ok=True)
    plan = freeze()
    protocol_sha = digest(OUT / 'protocol.json')
    rows = plan['candidates']
    complete = {}
    print(f'Frozen {len(rows)} public validation visits', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(run_visit, row, protocol_sha) for row in rows]
        for future in as_completed(tasks):
            result = future.result()
            complete[result['video']] = result
            print(result['video'], 'depth pairs:', len(result['candidates']),
                  'RGB pairs:', result.get('rgb_pairs_downloaded'),
                  'selected:', result['selected'] is not None, result.get('error', ''), flush=True)
    write(OUT / 'inputs.json', dict(protocol_sha256=protocol_sha,
                                   visits=[complete[row['video_id']] for row in rows]))


if __name__ == '__main__':
    main()
