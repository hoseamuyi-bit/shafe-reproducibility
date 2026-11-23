"""Attribute missing temporal reprojections without modifying the frozen gate."""

import cv2
import numpy as np

from fetch_arkitscenes_frames import FrameArchive, download_frames
from prepare_arkitscenes_temporal import OUT, digest, make_grid, source_for
from probe_arkitscenes_outside_pose import read, write
from screen_arkitscenes_fixed_lr import near_pixels
from screen_arkitscenes_x3 import intrinsics, prepare_poses, project, to_world
from verify_arkitscenes_fixed_lr import get_rgb, rgb_tracks, sample


def load_depth(archive, entry, camera):
    path = archive.root / entry['member'].rsplit('/', 1)[-1]
    if digest(path) != entry['sha256']:
        raise ValueError(f'Depth SHA-256 mismatch: {path}')
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != tuple(camera[:2][::-1].astype(int)) or image.dtype != np.uint16:
        raise ValueError(f'Invalid depth PNG: {path}')
    return image


def project_depth(grid, depth_a, depth_b, camera_a, camera_b, pose_a, pose_b):
    source = sample(depth_a, grid, cv2.INTER_NEAREST) * 0.001
    with np.errstate(divide='ignore', invalid='ignore'):
        xy, z = project(to_world(grid, source, camera_a, pose_a, 0.5), camera_b, pose_b, 0.5)
    valid_source = source > 0
    valid_projected = valid_source & (z > 0.01) & np.isfinite(z) & np.isfinite(xy).all(axis=-1)
    safe_xy = np.where(valid_projected[..., None], xy, -1000)
    inside = valid_projected & ((safe_xy >= 2) & (safe_xy < camera_b[:2] - 2)).all(axis=-1)
    target = sample(depth_b, safe_xy, cv2.INTER_NEAREST) * 0.001
    has_target = inside & (target > 0)
    visible = has_target & (np.abs(target - z) <= 0.03)
    masks = dict(missing_source=(~valid_source), invalid_projection=(valid_source & ~valid_projected),
                 outside=(valid_projected & ~inside), missing_target=(inside & ~has_target),
                 depth_error_over_3cm=(has_target & ~visible), visible=visible)
    return safe_xy, masks


def diagnose(row):
    source = source_for(row['video'])
    site = source.parent
    candidate = row['candidate']
    t, u = candidate['anchor_time'], candidate['shifted_time']
    low, wide = (intrinsics(site, label) for label in ('lowres_wide', 'wide'))
    poses = prepare_poses(np.loadtxt(site / 'lowres_wide.traj'), (t, u))
    lx, ly, ex, ey = row['score_xyxy']
    context = make_grid(row['context_origin_lr'], 128)[ly:ey, lx:ex]
    grid, scale = near_pixels(context, low[t], wide[t])
    depth_archive = FrameArchive(site, 'Validation', 'highres_depth')
    manifest = download_frames(depth_archive, (t, u))
    if manifest != row['depth']['source']:
        raise ValueError('Depth source differs from the frozen window')
    depths = tuple(load_depth(depth_archive, entry, wide[entry['timestamp']])
                   for entry in manifest['frames'])
    pose_pairs = tuple(zip(poses[t][0], poses[u][0], strict=True))
    projections, visible_masks, counts = [], [], {}
    for name, (a, b), old in zip(('linear', 'spline'), pose_pairs, row['depth']['projection'], strict=True):
        xy, masks = project_depth(grid, *depths, wide[t], wide[u], a, b)
        record = {label: int(mask.sum()) for label, mask in masks.items()}
        if sum(record.values()) != grid.shape[0] * grid.shape[1]:
            raise ValueError('Projection failure categories do not partition the score core')
        inside = ~(masks['missing_source'] | masks['invalid_projection'] | masks['outside'])
        if masks['visible'].mean() != old['visible_fraction'] or inside.mean() != old['fraction_inside']:
            raise ValueError('Decomposed counts disagree with frozen projection logs')
        counts[name] = record
        projections.append(xy)
        visible_masks.append(masks['visible'])
    common = visible_masks[0] & visible_masks[1]
    disagreement = np.linalg.norm((projections[0] - projections[1]) * scale, axis=-1)
    anchor, moved = (get_rgb(site, 'wide', time, wide[time])[0] for time in (t, u))
    # The tracker reports its RGB-only point pairs regardless of the supplied predictions.
    points = rgb_tracks(anchor, moved, grid, (grid, grid), scale)
    tracks = dict(rgb_consistent_count=points['tracks'], dual_depth_valid_count=0)
    if points['tracks']:
        start, end = (np.array(points[label]) for label in ('positions_xy_source', 'positions_xy_target'))
        expected, valid = [], []
        for a, b in pose_pairs:
            xy, masks = project_depth(start[:, None, :], *depths, wide[t], wide[u], a, b)
            expected.append(xy[:, 0])
            valid.append(masks['visible'][:, 0])
        keep = valid[0] & valid[1]
        if keep.any():
            center = (np.ceil(grid[0, 0]) + np.floor(grid[-1, -1])) * 0.5
            quadrants = (start[keep] >= center).astype(int) @ [1, 2]
            residuals = np.linalg.norm((np.array(expected)[:, keep] - end[keep]) * scale, axis=-1)
            tracks.update(dual_depth_valid_count=int(keep.sum()),
                          dual_depth_valid_by_quadrant=np.bincount(quadrants, minlength=4).tolist(),
                          rgb_p90_error_hr_px=np.quantile(residuals, 0.9, axis=1).tolist(),
                          rgb_max_error_hr_px=residuals.max(axis=1).tolist())
    return dict(video=row['video'], visit=row['visit'], pixels=int(common.size), failures=counts,
                dual_visible_count=int(common.sum()),
                dual_visible_pose_disagreement_max_hr_px=float(disagreement[common].max())
                if common.any() else None, tracks=tracks, accepted_by_frozen_gate=row['depth']['passed'])


def main():
    cv2.setNumThreads(1)
    manifest = OUT / 'inputs.json'
    saved = read(manifest)
    if saved['protocol_sha256'] != digest(OUT / 'protocol.json'):
        raise ValueError('Input manifest differs from the frozen protocol')
    results = []
    for row in saved['visits']:
        if row['selected']:
            result = diagnose(row['selected'])
            results.append(result)
            print(result, flush=True)
    write(OUT / 'depth_failure_diagnostics.json', dict(inputs_sha256=digest(manifest), visits=results))


if __name__ == '__main__':
    main()
