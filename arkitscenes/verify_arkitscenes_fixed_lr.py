"""Qualify a sparse ARKitScenes candidate with dense depth and original RGB."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from check_arkitscenes_sample_centers import grayscale, registration
from fetch_arkitscenes_frames import atomic_write
from screen_arkitscenes_fixed_lr import MAX_BEND, MAX_POSE_SENSITIVITY, OUTPUT_SCALE, SHIFT_BAND, near_pixels
from screen_arkitscenes_x3 import decode, intrinsics, prepare_poses, project, to_world
from triage_arkitscenes_native import timestamp


MAX_DEPTH_ERROR_M = 0.03
MIN_TARGET_DEPTH_AGREEMENT = 0.99
MAX_RGB_P90_HR = 0.75
MAX_RGB_ERROR_HR = 1.25
MIN_TRACKS = 8


def pixels(x, y):
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy), axis=-1)


def sample(image, xy, method=cv2.INTER_LINEAR):
    return cv2.remap(image, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), method)


def get_rgb(root, asset, time, camera):
    data = json.loads((root / f'{asset}_frames.json').read_bytes())
    member = next(row for row in data['frames'] if row['timestamp'] == time)
    raw = (root / asset / Path(member['member']).name).read_bytes()
    if sha256(raw).hexdigest() != member['sha256']:
        raise ValueError(f'{asset} frame changed: {time}')
    rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape[:2] != tuple(camera[:2][::-1].astype(int)):
        raise ValueError(f'{asset} frame has unexpected dimensions: {time}')
    return rgb, member['sha256']


def dense_geometry(source_grid, source_depth, target_depth, source_cam, target_cam, poses, scale):
    depth = sample(source_depth, source_grid, cv2.INTER_NEAREST) * 0.001
    if not (depth > 0).all():
        return {'pass': False, 'anchor_valid_fraction': float((depth > 0).mean())}, []
    projections = []
    records = []
    for src_pose, dst_pose in zip(*poses, strict=True):
        world = to_world(source_grid, depth, source_cam, src_pose, 0.5)
        projected, z = project(world, target_cam, dst_pose, 0.5)
        disparity = (projected - source_grid) * scale
        shift = np.median(disparity, axis=(0, 1))
        bend = np.linalg.norm(disparity - shift, axis=-1)
        inside = (z > 0.01) & ((projected >= 2) & (projected < target_cam[:2] - 2)).all(axis=-1)
        measured = sample(target_depth, projected, cv2.INTER_NEAREST) * 0.001
        depth_agree = inside & (measured > 0) & (np.abs(z - measured) <= MAX_DEPTH_ERROR_M)
        records.append(dict(shift_xy_hr_px=shift.tolist(), shift_length_hr_px=float(np.linalg.norm(shift)),
                            bend_max_hr_px=float(bend.max()), bend_p90_hr_px=float(np.quantile(bend, 0.9)),
                            target_inside_fraction=float(inside.mean()),
                            target_depth_agreement_fraction=float(depth_agree.mean())))
        projections.append(projected)
    disagreement = float(np.linalg.norm((projections[0] - projections[1]) * scale, axis=-1).max())
    passed = all(SHIFT_BAND[0] <= row['shift_length_hr_px'] <= SHIFT_BAND[1]
                 and row['bend_max_hr_px'] <= MAX_BEND and row['target_inside_fraction'] == 1
                 and row['target_depth_agreement_fraction'] >= MIN_TARGET_DEPTH_AGREEMENT for row in records)
    return dict(pose_variants=records, pose_disagreement_max_hr_px=disagreement,
                **{'pass': passed and disagreement <= MAX_POSE_SENSITIVITY}), projections


def rgb_tracks(anchor, moved, grid, projections, scale):
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (anchor, moved)]
    lower, upper = np.ceil(grid[0, 0]).astype(int), np.floor(grid[-1, -1]).astype(int)
    x0, y0 = lower
    x1, y1 = upper
    mask = np.zeros(gray[0].shape, np.uint8)
    mask[y0:y1+1, x0:x1+1] = 255
    origin = cv2.goodFeaturesToTrack(gray[0], 100, 0.005, 12, mask=mask, blockSize=7)
    if origin is None:
        return {'pass': False, 'tracks': 0, 'reason': 'no textured corners'}
    settings = dict(winSize=(21, 21), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))
    target, ok, _ = cv2.calcOpticalFlowPyrLK(*gray, origin, None, **settings)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(*gray[::-1], target, None, **settings)
    fb = np.linalg.norm((back - origin) * scale, axis=2)[:, 0]
    keep = (ok[:, 0] > 0) & (ok_back[:, 0] > 0) & (fb <= 0.3)
    start = origin[keep, 0]
    stop = target[keep, 0]
    if not len(start):
        return {'pass': False, 'tracks': 0, 'reason': 'no forward/backward-consistent tracks'}
    on_grid = ((start - grid[0, 0]) * scale).reshape(-1, 1, 2)
    expected = [sample(projected.astype(np.float32), on_grid).reshape(-1, 2) for projected in projections]
    errors = np.array([np.linalg.norm((stop - prediction) * scale, axis=-1) for prediction in expected])
    mid = (lower + upper) * 0.5
    quadrants = (start >= mid).astype(int) @ [1, 2]
    coverage = np.bincount(quadrants, minlength=4)
    med = np.median((stop - start) * scale, axis=0)
    p90 = np.quantile(errors, 0.9, axis=1)
    worst = errors.max(axis=1)
    return dict(tracks=len(start), tracks_by_quadrant=coverage.tolist(),
                median_observed_rgb_shift_xy_hr_px=med.tolist(), rgb_p90_error_hr_px=p90.tolist(),
                rgb_max_error_hr_px=worst.tolist(),
                positions_xy_source=start.tolist(), positions_xy_target=stop.tolist(),
                **{'pass': bool(len(start) >= MIN_TRACKS and (coverage > 0).all()
                                and (p90 <= MAX_RGB_P90_HR).all() and (worst <= MAX_RGB_ERROR_HR).all())})


def exact_time_check(origin, size, low, high, low_cam, high_cam):
    x, y = origin
    pad = 8
    grid_lr = pixels(np.arange(x-pad, x+size+pad), np.arange(y-pad, y+size+pad))
    high_xy, _ = near_pixels(grid_lr, low_cam, high_cam)
    template = grayscale(low)[y-pad:y+size+pad, x-pad:x+size+pad]
    estimates = []
    for sigma in (2.25, 3.0, 3.75):
        filtered = cv2.GaussianBlur(grayscale(high), (0, 0), sigma)
        aligned = sample(filtered, high_xy)
        shift = registration(template, aligned)
        estimates.append(None if shift is None else (shift * OUTPUT_SCALE).tolist())
    norm = [float(np.linalg.norm(value)) for value in estimates if value is not None]
    return dict(extent_lr=[x-pad, y-pad, x+size+pad, y+size+pad], offset_hr_px=estimates,
                gaussian_sigma_native_px=[2.25, 3.0, 3.75],
                **{'pass': len(norm) == len(estimates) and max(norm) <= 0.75})


def verify(root):
    screen_path = root / 'fixed_lr_exploratory_screen.json'
    screen = json.loads(screen_path.read_bytes())
    trajectory = np.loadtxt(root / 'lowres_wide.traj')
    wide = intrinsics(root, 'wide')
    low = intrinsics(root, 'lowres_wide')
    output = []
    with ZipFile(root / 'raw_highres_depth.zip') as archive:
        members = {timestamp(name): name for name in archive.namelist() if name.endswith('.png')}
        for size, candidates in screen['candidates'].items():
            for candidate in candidates:
                a, b = candidate['anchor_time'], candidate['shifted_time']
                if a not in members or b not in members:
                    raise ValueError('Candidate missing from the depth archive')
                x, y = candidate['origin_lr']
                lr_offset = (np.arange(int(size) * OUTPUT_SCALE) + 0.5) / OUTPUT_SCALE - 0.5
                full_grid, scale = near_pixels(pixels(x + lr_offset, y + lr_offset), low[a], wide[a])
                depth_a, depth_b = (decode(archive, members[t], cv2.IMREAD_UNCHANGED) for t in (a, b))
                poses = prepare_poses(trajectory, (a, b))
                geometry, projections = dense_geometry(full_grid, depth_a, depth_b, wide[a], wide[b],
                                                       (poses[a][0], poses[b][0]), scale)
                row = dict(**candidate, size_lr=int(size), dense=geometry)
                if projections:
                    rgb_a, sha_a = get_rgb(root, 'wide', a, wide[a])
                    rgb_b, sha_b = get_rgb(root, 'wide', b, wide[b])
                    rgb_lr, sha_lr = get_rgb(root, 'lowres_wide', a, low[a])
                    row['rgb_sha256'] = dict(anchor_hr=sha_a, shifted_hr=sha_b, anchor_lr=sha_lr)
                    row['rgb'] = rgb_tracks(rgb_a, rgb_b, full_grid, projections, scale)
                    row['synchronized_lr_hr'] = exact_time_check((x, y), int(size), rgb_lr, rgb_a, low[a], wide[a])
                row['pass'] = all(row.get(stage, {}).get('pass', False)
                                  for stage in ('dense', 'rgb', 'synchronized_lr_hr'))
                output.append(row)
    report = dict(video=screen['video'], input_screen_sha256=sha256(screen_path.read_bytes()).hexdigest(),
                  verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), records=output)
    atomic_write(root / 'fixed_lr_dense_rgb_check.json', (json.dumps(report, indent=2) + '\n').encode())
    print(f'{screen["video"]}: {sum(row["pass"] for row in output)}/{len(output)} candidates qualified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path)
    cv2.setNumThreads(1)
    verify(parser.parse_args().site)
