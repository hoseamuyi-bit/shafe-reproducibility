"""Find native x3 temporal pairs using only published calibration and FARO depth."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, RotationSpline

from triage_arkitscenes_native import poses_at, timestamp


SCALE = 3
LR_SIZE = 96
HR_SIZE = SCALE * LR_SIZE
ORIGINS_LR = np.array(list(product(range(64, 449, 64), range(64, 321, 64))))
SAMPLE = np.rint(np.linspace(0, HR_SIZE - 1, 7)).astype(int)
GRID_HR = SCALE * ORIGINS_LR[:, None, :] + np.array(list(product(SAMPLE, repeat=2)))
MAX_NEAR_TIME = 0.0205
MAX_SHIFT_TIME = 0.41
MAX_TRAJECTORY_GAP = 0.151
NEAR = 0.75
SHIFT = (2.25, 6.)
MAX_NON_TRANSLATION = 0.50
MAX_INTERPOLATION_DISAGREEMENT = 0.35
CONVENTIONS = (0., 0.5)


def intrinsics(root, asset):
    with ZipFile(root / f'{asset}_intrinsics.zip') as archive:
        return {timestamp(name): np.fromstring(archive.read(name).decode(), sep=' ')
                for name in archive.namelist() if name.endswith('.pincam')}


def decode(archive, name, mode):
    image = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), mode)
    if image is None:
        raise ValueError(f'Could not decode {name}')
    return image


def prepare_poses(trajectory, times):
    times = np.array(sorted(times))
    rotation, centers = poses_at(times, trajectory)
    base_times = trajectory[:, 0]
    base_rotation = Rotation.from_rotvec(trajectory[:, 1:4])
    base_centers = -base_rotation.inv().apply(trajectory[:, 4:])
    spline_rotation = RotationSpline(base_times, base_rotation)(times).as_matrix()
    spline_centers = PchipInterpolator(base_times, base_centers)(times)
    right = np.clip(np.searchsorted(base_times, times), 1, len(base_times) - 1)
    max_gap = base_times[right] - base_times[right - 1]
    return {
        time: (tuple((matrix, center) for matrix, center in
                     zip((linear, spline), (linear_center, spline_center), strict=True)), gap)
        for time, linear, linear_center, spline, spline_center, gap in
        zip(times, rotation, centers, spline_rotation, spline_centers, max_gap, strict=True)
    }


def to_world(pixel, z, camera, pose, convention):
    _, _, fx, fy, cx, cy = camera
    rotation, center = pose
    rays = (pixel + convention - [cx, cy]) / [fx, fy]
    return np.concatenate((rays * z[..., None], z[..., None]), axis=-1) @ rotation + center


def project(world, camera, pose, convention):
    _, _, fx, fy, cx, cy = camera
    rotation, center = pose
    point = (world - center) @ rotation.T
    z = point[..., 2]
    pixel = point[..., :2] / z[..., None] * [fx, fy] + [cx, cy] - convention
    return pixel, z


def delta_on_hr_grid(pixel, vga, scale=SCALE):
    return scale * (vga + 0.5) - 0.5 - pixel


def screen(root, mode, max_shift_time=None, shifted_bend=MAX_NON_TRANSLATION):
    depth_path = root / 'raw_highres_depth.zip'
    trajectory = np.loadtxt(root / 'lowres_wide.traj')
    hr_intr = intrinsics(root, 'wide')
    if mode == 'native_x3':
        asset = 'vga_wide'
        input_scale = output_scale = SCALE
        size = LR_SIZE
        origins, grid = ORIGINS_LR, GRID_HR
        conventions = CONVENTIONS
        near_time = MAX_NEAR_TIME
    else:
        asset = 'lowres_wide'
        input_scale, output_scale = 7.5, 4
        size = 64
        origins = np.array(list(product(range(32, 161, 32), range(24, 105, 24))))
        samples = np.rint(np.linspace(0, input_scale * size - 1, 7)).astype(int)
        grid = (input_scale * origins[:, None, :]).astype(int) + np.array(list(product(samples, repeat=2)))
        conventions = (0.5,)
        near_time = 0.0
    lr_intr = intrinsics(root, asset)
    shift_time = MAX_SHIFT_TIME if max_shift_time is None else max_shift_time
    counts = {c: Counter() for c in conventions}
    matches = {convention: defaultdict(lambda: dict(near=[], shift=[])) for convention in conventions}

    with ZipFile(depth_path) as archive:
        names = {timestamp(n): n for n in archive.namelist() if n.endswith('.png')}
        hr_times = sorted(t for t in names if t in hr_intr and trajectory[0, 0] <= t <= trajectory[-1, 0])
        possible_lr_times = sorted(lr_intr if near_time else lr_intr.keys() & names.keys() & hr_intr.keys())
        times = np.array([t for t in possible_lr_times if trajectory[0, 0] <= t <= trajectory[-1, 0]])
        poses = prepare_poses(trajectory, set(hr_times) | set(times))
        for hr_time in hr_times:
            hr_poses, source_gap = poses[hr_time]
            if source_gap > MAX_TRAJECTORY_GAP:
                continue
            depth = decode(archive, names[hr_time], cv2.IMREAD_UNCHANGED)
            shape = tuple(hr_intr[hr_time][:2][::-1].astype(int))
            if depth.dtype != np.uint16 or depth.shape != shape:
                raise ValueError(f'Unexpected depth geometry: {hr_time} {depth.shape} {depth.dtype}')
            z = depth[grid[..., 1], grid[..., 0]] * 0.001
            valid = np.flatnonzero((z > 0).all(axis=1))
            if not valid.size:
                continue
            pixels = grid[valid]
            temporal_candidates = times[np.abs(times - hr_time) <= shift_time]
            for convention in conventions:
                world = [to_world(pixels, z[valid], hr_intr[hr_time], pose, convention) for pose in hr_poses]
                stats = counts[convention]
                stats['frames_with_valid_sparse_depth'] += 1
                for vga_time in temporal_candidates:
                    target_poses, target_gap = poses[vga_time]
                    if target_gap > MAX_TRAJECTORY_GAP:
                        continue
                    stats['frame_pairs'] += 1
                    stats['windows_with_sparse_depth'] += valid.size
                    camera = lr_intr[vga_time]
                    vga, projected_depth = project(world[0], camera, target_poses[0], convention)
                    inside = ((projected_depth > 0.01).all(axis=1) &
                              ((vga >= 2) & (vga <= camera[:2] - 3)).all(axis=(1, 2)))
                    delta = delta_on_hr_grid(pixels, vga, input_scale) * (output_scale / input_scale)
                    shift = np.median(delta, axis=1)
                    max_bend = np.linalg.norm(delta - shift[:, None, :], axis=-1).max(axis=1)
                    length = np.linalg.norm(shift, axis=1)
                    near = ((length <= NEAR) & (abs(vga_time - hr_time) <= near_time) &
                            (max_bend <= MAX_NON_TRANSLATION))
                    shifted = (length >= SHIFT[0]) & (length <= SHIFT[1])
                    stats['shift_windows_before_non_translation'] += int((inside & shifted).sum())
                    shifted &= max_bend <= shifted_bend
                    stats['shift_windows_before_interpolation'] += int((inside & shifted).sum())
                    selected = np.flatnonzero(inside & (near | shifted))
                    stats['preliminary_windows'] += selected.size
                    if not selected.size:
                        continue
                    vga_spline, _ = project(world[1][selected], camera, target_poses[1], convention)
                    sensitivity = output_scale * np.linalg.norm(vga_spline - vga[selected], axis=-1).max(axis=1)
                    for index, disagreement in zip(selected, sensitivity, strict=True):
                        if disagreement > MAX_INTERPOLATION_DISAGREEMENT:
                            continue
                        arm = 'near' if near[index] else 'shift'
                        stats[f'{arm}_windows'] += 1
                        origin = tuple(map(int, origins[valid[index]]))
                        matches[convention][(float(vga_time), origin)][arm].append(
                            dict(hr_time=hr_time, shift_xy_hr_px=shift[index].tolist(),
                                 shift_norm_hr_px=float(length[index]),
                                 max_non_translation_hr_px=float(max_bend[index]),
                                 max_interpolation_disagreement_hr_px=float(disagreement)))

    paired = {}
    for convention, windows in matches.items():
        paired[convention] = [dict(lr_time=t, origin_lr=origin, **arms)
                              for (t, origin), arms in sorted(windows.items()) if arms['near'] and arms['shift']]
        counts[convention]['paired_windows'] = len(paired[convention])
        counts[convention]['paired_lr_frames'] = len({row['lr_time'] for row in paired[convention]})
        counts[convention]['paired_near_hr_frames'] = len(
            {arm['hr_time'] for row in paired[convention] for arm in row['near']})
    assets = {}
    for path in (depth_path, root / 'wide_intrinsics.zip', root / f'{asset}_intrinsics.zip',
                 root / 'lowres_wide.traj'):
        with path.open('rb') as stream:
            assets[path.name] = dict(bytes=path.stat().st_size, sha256=file_digest(stream, 'sha256').hexdigest())
    report = dict(video=root.name.removeprefix('arkit_'), mode=mode, lr_asset=asset, capture_scale=input_scale,
                  score_scale=output_scale, lr_size=size, origins_lr=origins.tolist(),
                  sparse_samples_per_window=grid.shape[1],
                  depth_frames=len(names), hr_frames_covered_by_trajectory=len(hr_times), eligible_lr_times=len(times),
                  thresholds=dict(max_near_time_s=near_time, max_shift_time_s=shift_time,
                                  max_trajectory_gap_s=MAX_TRAJECTORY_GAP, near_max_hr_px=NEAR,
                                  shift_band_hr_px=SHIFT, near_max_non_translation_hr_px=MAX_NON_TRANSLATION,
                                  shift_max_non_translation_hr_px=shifted_bend,
                                  max_interpolation_disagreement_hr_px=MAX_INTERPOLATION_DISAGREEMENT),
                  pixel_coordinate_conventions={'0.0': 'integer centers', '0.5': 'half-integer centers'},
                  counts=counts, pairs=paired, assets=assets)
    suffix = '' if max_shift_time is None else '_' + str(max_shift_time).replace('.', 'p') + 's'
    if shifted_bend != MAX_NON_TRANSLATION:
        suffix += '_bend_' + str(shifted_bend).replace('.', 'p')
    path = root / f'{mode}_sparse_screen{suffix}.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--mode', choices=('native_x3', 'synchronized_x4'), default='native_x3')
    parser.add_argument('--max-shift-time', type=float)
    parser.add_argument('--shift-max-bend', type=float, default=MAX_NON_TRANSLATION)
    args = parser.parse_args()
    screen(args.root, args.mode, args.max_shift_time, args.shift_max_bend)
