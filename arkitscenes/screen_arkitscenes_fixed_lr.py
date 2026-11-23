"""Exploratory screen anchored to one synchronized LR/HR capture per triplet."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from fetch_arkitscenes_frames import atomic_write
from screen_arkitscenes_x3 import decode, intrinsics, prepare_poses, to_world
from triage_arkitscenes_native import timestamp


SIZES = (64, 48, 32)
LAG_SECONDS = 2.1
SHIFT_BAND = (2.25, 6.0)
MAX_BEND = 0.5
MAX_POSE_SENSITIVITY = 0.35
MAX_POSITION_GAP = 0.151
PRELIMINARY_SAMPLES = 7
OUTPUT_SCALE = 4


def window_grid(size):
    step = size // 2
    origins = np.array(list(product(range(step, 256 - size, step), range(step, 192 - size, step))))
    xy = np.linspace(0, OUTPUT_SCALE * size - 1, PRELIMINARY_SAMPLES)
    offsets = (np.array(list(product(xy, repeat=2))) + 0.5) / OUTPUT_SCALE - 0.5
    return origins, origins[:, None] + offsets


def near_pixels(grid, lo, hi):
    ratio = hi[2:4] / lo[2:4]
    return (grid + 0.5 - lo[4:6]) * ratio + hi[4:6] - 0.5, OUTPUT_SCALE / ratio


def project_batch(world, rotations, centers, cameras):
    local = (world[None, :, :, :] - centers[:, None, None, :]) @ rotations[:, None, :, :].transpose(0, 1, 3, 2)
    distance = local[..., 2]
    pixels = local[..., :2] / np.maximum(distance[..., None], 0.001)
    pixels = pixels * cameras[:, None, None, 2:4] + cameras[:, None, None, 4:6] - 0.5
    return pixels, distance


def measure(pixels, origins, depth, lo, hi, base_pose, neighbors, stats):
    near, scale = near_pixels(pixels, lo, hi)
    indexes = np.rint(near).astype(int)
    covered = ((indexes >= 0) & (indexes < hi[:2])).all(axis=(1, 2))
    indexes = indexes[covered]
    z = 0.001 * depth[indexes[..., 1], indexes[..., 0]]
    mask = (z > 0).all(axis=1)
    good = np.flatnonzero(covered)[mask]
    if not good.size:
        return []

    near = near[good]
    world = to_world(near, z[mask], hi, base_pose[0], 0.5)
    cameras, rotation, center, spline_rotation, spline_center = neighbors
    projected, distance = project_batch(world, rotation, center, cameras)
    displacement = (projected - near[None]) * scale
    shifts = np.median(displacement, axis=2)
    residuals = np.linalg.norm(displacement - shifts[:, :, None], axis=-1).max(axis=2)
    lengths = np.linalg.norm(shifts, axis=-1)
    inside = (distance > 0.01).all(axis=2) & ((projected >= 2) &
             (projected < cameras[:, None, None, :2] - 2)).all(axis=(2, 3))
    band = inside & (lengths >= SHIFT_BAND[0]) & (lengths <= SHIFT_BAND[1])
    stats['in_range_with_depth'] += int(inside.sum())
    stats['shift_in_band'] += int(band.sum())
    if band.any():
        stats['min_bend_in_band'] = min(stats.get('min_bend_in_band', float('inf')), float(residuals[band].min()))
    preliminary = np.argwhere(band & (residuals <= MAX_BEND))
    stats['near_translation'] += len(preliminary)
    candidates = []
    if preliminary.size:
        alternate_world = to_world(near, z[mask], hi, base_pose[1], 0.5)
        check, _ = project_batch(alternate_world, spline_rotation, spline_center, cameras)
        check_delta = np.linalg.norm((check - projected) * scale, axis=-1).max(axis=2)
        for i, j in preliminary:
            if check_delta[i, j] <= MAX_POSE_SENSITIVITY:
                candidates.append(dict(target_index=int(i), origin_lr=origins[good[j]].tolist(),
                                       shift_xy_hr_px=shifts[i, j].tolist(),
                                       bend_max_hr_px=float(residuals[i, j]),
                                       sensitivity_max_hr_px=float(check_delta[i, j])))
    return candidates


def screen(root):
    depth_path = root / 'raw_highres_depth.zip'
    trajectory = np.loadtxt(root / 'lowres_wide.traj')
    wide = intrinsics(root, 'wide')
    low = intrinsics(root, 'lowres_wide')
    grids = {size: window_grid(size) for size in SIZES}
    matches = {size: [] for size in SIZES}
    counts = {size: Counter() for size in SIZES}

    with ZipFile(depth_path) as archive:
        members = {timestamp(name): name for name in archive.namelist() if name.endswith('.png')}
        available = np.array(sorted(t for t in members.keys() & wide.keys()
                                    if trajectory[0, 0] <= t <= trajectory[-1, 0]))
        poses = prepare_poses(trajectory, available)
        times = np.array([t for t in available if poses[t][1] <= MAX_POSITION_GAP])
        for time in times:
            if time not in low:
                continue
            target_times = times[(times != time) & (np.abs(times - time) <= LAG_SECONDS)]
            if not len(target_times):
                continue
            depth = decode(archive, members[time], cv2.IMREAD_UNCHANGED)
            if depth.dtype != np.uint16 or depth.shape != tuple(wide[time][:2][::-1].astype(int)):
                raise ValueError(f'Unexpected depth geometry at {time}')
            target_cams = np.array([wide[t] for t in target_times])
            rotation, center = (np.array([poses[t][0][0][k] for t in target_times]) for k in (0, 1))
            alt_rotation, alt_center = (np.array([poses[t][0][1][k] for t in target_times]) for k in (0, 1))
            neighbors = (target_cams, rotation, center, alt_rotation, alt_center)
            for size, (origins, grid) in grids.items():
                stats = counts[size]
                stats['source_frames'] += 1
                stats['target_pairs'] += len(target_times)
                results = measure(grid, origins, depth, low[time], wide[time], poses[time][0], neighbors, stats)
                for result in results:
                    i = result.pop('target_index')
                    matches[size].append(dict(anchor_time=float(time), shifted_time=float(target_times[i]), **result))
    for size, rows in matches.items():
        counts[size]['candidates'] = len(rows)
        rows.sort(key=lambda row: (row['bend_max_hr_px'], row['sensitivity_max_hr_px'], row['anchor_time']))
    assets = {}
    for name in ('raw_highres_depth.zip', 'wide_intrinsics.zip', 'lowres_wide_intrinsics.zip', 'lowres_wide.traj'):
        path = root / name
        with path.open('rb') as stream:
            assets[name] = dict(sha256=file_digest(stream, 'sha256').hexdigest(), bytes=path.stat().st_size)
    report = dict(video=root.name.removeprefix('arkit_'), analysis='exploratory_fixed_lr_temporal_geometry',
                  code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  depth_frames=len(members), lag_seconds=LAG_SECONDS, shift_band_hr_px=SHIFT_BAND,
                  max_bend_hr_px=MAX_BEND, max_pose_sensitivity_hr_px=MAX_POSE_SENSITIVITY,
                  samples_per_window=PRELIMINARY_SAMPLES ** 2, counts=counts, candidates=matches, assets=assets)
    atomic_write(root / 'fixed_lr_exploratory_screen.json', (json.dumps(report, indent=2) + '\n').encode())
    print(root.name, json.dumps(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sites', type=Path, nargs='+')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    for site in args.sites:
        screen(site)
