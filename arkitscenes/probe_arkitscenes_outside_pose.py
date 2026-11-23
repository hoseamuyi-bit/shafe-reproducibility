"""Evaluate an outside-crop pose fit on a fixed set of public ARKitScenes visits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from diagnose_arkitscenes_pose import fit_parts, outside_tracks
from fetch_arkitscenes_frames import FrameArchive, atomic_write, download_frames
from screen_arkitscenes_fixed_lr import (LAG_SECONDS, MAX_POSITION_GAP, OUTPUT_SCALE, near_pixels,
                                         project_batch, window_grid)
from screen_arkitscenes_x3 import decode, intrinsics, prepare_poses, to_world
from triage_arkitscenes_native import timestamp
from verify_arkitscenes_fixed_lr import (MAX_RGB_P90_HR, dense_geometry, exact_time_check, get_rgb,
                                         pixels, rgb_tracks)


ROOT = Path(__file__).resolve().parent
VIDEOS = ('45663164', '47333934', '42898811')
SIZES = (64, 48, 32)
PAIR_LIMIT = 3
MIN_SHIFT = 1.0
MAX_SHIFT = 10.0
MAX_SCREEN_BEND = 1.5


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    atomic_write(path, (json.dumps(obj, indent=2) + '\n').encode())


def depth_members(archive):
    return {timestamp(name): name for name in archive.namelist() if name.endswith('.png')}


def shortlist(site, archive, high, low):
    trajectory = np.loadtxt(site / 'lowres_wide.traj')
    names = depth_members(archive)
    times = np.array(sorted(t for t in names.keys() & high.keys()
                            if trajectory[0, 0] <= t <= trajectory[-1, 0]))
    poses = prepare_poses(trajectory, times)
    times = np.array([t for t in times if poses[t][1] <= MAX_POSITION_GAP])
    cameras = np.array([high[t] for t in times])
    rotations, centers = (np.array([poses[t][0][0][k] for t in times]) for k in (0, 1))
    grids = {size: window_grid(size) for size in SIZES}
    best_by_pair = {size: {} for size in SIZES}
    for i, time in enumerate(times):
        if time not in low:
            continue
        nearby = np.flatnonzero((times != time) & (np.abs(times - time) <= LAG_SECONDS))
        if not nearby.size:
            continue
        depth = decode(archive, names[time], cv2.IMREAD_UNCHANGED)
        for size, (origins, grid) in grids.items():
            native, scale = near_pixels(grid, low[time], high[time])
            xy = np.rint(native).astype(int)
            valid = ((xy >= 0) & (xy < high[time][:2])).all(axis=(1, 2))
            xy = xy[valid]
            z = depth[xy[..., 1], xy[..., 0]] * 0.001
            covered = (z > 0).all(axis=1)
            if not covered.any():
                continue
            pos = native[valid][covered]
            world = to_world(pos, z[covered], high[time], poses[time][0][0], 0.5)
            projected, distance = project_batch(world, rotations[nearby], centers[nearby], cameras[nearby])
            motion = (projected - pos) * scale
            shift = np.median(motion, axis=2)
            length = np.linalg.norm(shift, axis=-1)
            bend = np.linalg.norm(motion - shift[:, :, None], axis=-1).max(axis=2)
            inside = ((projected >= 2) & (projected < cameras[nearby, None, None, :2] - 2)).all(axis=(2, 3))
            candidates = ((distance > 0.01).all(axis=2) & inside & (length >= MIN_SHIFT)
                          & (length <= MAX_SHIFT) & (bend <= MAX_SCREEN_BEND))
            for k, j in np.argwhere(candidates):
                target = float(times[nearby[k]])
                pair = tuple(sorted((float(time), target)))
                score = float(bend[k, j])
                previous = best_by_pair[size].get(pair)
                if previous is None or score < previous['predicted_bend_hr_px']:
                    best_by_pair[size][pair] = dict(
                        anchor_time=float(time), shifted_time=target, size_lr=size,
                        origin_lr=origins[valid][covered][j].tolist(),
                        predicted_bend_hr_px=score, predicted_shift_hr_px=float(length[k, j]))
    return [row for size in SIZES
            for row in sorted(best_by_pair[size].values(), key=lambda row: row['predicted_bend_hr_px'])[:PAIR_LIMIT]]


def fetch(site, chosen):
    anchor = {row['anchor_time'] for row in chosen}
    shifted = {row['shifted_time'] for row in chosen}
    missing = {}
    for asset, times in (('wide', anchor | shifted), ('lowres_wide', anchor)):
        archive = FrameArchive(site, 'Validation', asset)
        manifest = download_frames(archive, times & archive.frames.keys())
        missing[asset] = sorted(times - archive.frames.keys())
        manifest['unavailable_timestamps'] = missing[asset]
        write(site / f'{asset}_frames.json', manifest)
        print(f'{site.name}: {asset}, {len(manifest["frames"])} verified frames, '
              f'{sum(frame["bytes"] for frame in manifest["frames"]):,} bytes', flush=True)
    return missing


def evaluate(site, archive, high, low, chosen, missing):
    names = depth_members(archive)
    output = []
    for row in chosen:
        a, b = row['anchor_time'], row['shifted_time']
        record = dict(row, passed=False)
        output.append(record)
        if any(t in missing['wide'] for t in (a, b)) or a in missing['lowres_wide']:
            record['error'] = 'Required original RGB frame absent from archive'
            continue
        try:
            x, y = row['origin_lr']
            size = row['size_lr']
            offset = (np.arange(size * OUTPUT_SCALE) + 0.5) / OUTPUT_SCALE - 0.5
            grid, scale = near_pixels(pixels(x + offset, y + offset), low[a], high[a])
            depth_a, depth_b = (decode(archive, names[t], cv2.IMREAD_UNCHANGED) for t in (a, b))
            rgb_a, hash_a = get_rgb(site, 'wide', a, high[a])
            rgb_b, hash_b = get_rgb(site, 'wide', b, high[b])
            rgb_lr, hash_lr = get_rgb(site, 'lowres_wide', a, low[a])
            record['rgb_sha256'] = dict(anchor_hr=hash_a, shifted_hr=hash_b, anchor_lr=hash_lr)
            record['synchronized_lr_hr'] = exact_time_check((x, y), size, rgb_lr, rgb_a, low[a], high[a])
            cv2.setRNGSeed(261122)
            fitted, folds = fit_parts(*outside_tracks(rgb_a, rgb_b, depth_a, high[a], grid, scale), high[b])
            record['outside_pose_folds'] = folds
            dense, projections = dense_geometry(grid, depth_a, depth_b, high[a], high[b],
                                                ([(np.eye(3), np.zeros(3))] * 2, fitted), scale)
            record['dense'] = dense
            record['outside_rgb_pass'] = bool(all(fold['outside_heldout_p90_native_px'] * max(scale)
                                                  <= MAX_RGB_P90_HR for fold in folds))
            if not projections:
                continue
            record['reserved_rgb'] = rgb_tracks(rgb_a, rgb_b, grid, projections, scale)
            record['passed'] = all(record[stage]['pass']
                                   for stage in ('synchronized_lr_hr', 'dense', 'reserved_rgb')) and record['outside_rgb_pass']
        except (ValueError, cv2.error, IndexError) as error:
            record['error'] = str(error)
    return output


def run(site):
    high = intrinsics(site, 'wide')
    low = intrinsics(site, 'lowres_wide')
    plan_path = site / 'outside_pose_probe_selection.json'
    report_path = site / 'outside_pose_probe_results.json'
    script_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    source_geometry = read(site / 'fixed_lr_exploratory_screen.json')['assets']
    with ZipFile(site / 'raw_highres_depth.zip') as archive:
        if plan_path.exists():
            plan = read(plan_path)
            if (plan['source_geometry'] != source_geometry or plan['script_sha256'] != script_hash):
                raise ValueError(f'Inputs changed since writing {plan_path}')
        else:
            plan = dict(source_geometry=source_geometry, script_sha256=script_hash,
                        rules=dict(sizes_lr=SIZES, pair_limit_per_size=PAIR_LIMIT, max_lag_s=LAG_SECONDS,
                                   approximate_shift_band_hr_px=(MIN_SHIFT, MAX_SHIFT),
                                   approximate_bend_max_hr_px=MAX_SCREEN_BEND),
                        rows=shortlist(site, archive, high, low))
            write(plan_path, plan)
        picked = plan['rows']
        print(f'{site.name}: {len(picked)} windows shortlisted before fetching original RGB', flush=True)
        rows = evaluate(site, archive, high, low, picked, fetch(site, picked)) if picked else []
    qualified = sum(row['passed'] for row in rows)
    write(report_path, dict(selection_sha256=sha256(plan_path.read_bytes()).hexdigest(),
                            qualified=qualified, records=rows))
    print(f'{site.name}: {qualified}/{len(rows)} passed', flush=True)
    return dict(video=site.name.removeprefix('arkit_'), qualified=qualified, examined=len(rows))


def main():
    cv2.setNumThreads(1)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(run, ROOT / f'arkit_{video}'): video for video in VIDEOS}
        for job in as_completed(jobs):
            try:
                results.append(job.result())
            except (OSError, ValueError) as error:
                video = jobs[job]
                print(f'{video}: {error}', flush=True)
                results.append(dict(video=video, error=str(error)))
    write(ROOT / 'arkitscenes_outside_pose_probe_results.json', dict(results=sorted(results, key=lambda r: r['video'])))


if __name__ == '__main__':
    main()
