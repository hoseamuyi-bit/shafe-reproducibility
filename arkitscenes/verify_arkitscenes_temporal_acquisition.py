"""Rebuild selected temporal scoring inputs from official cached frame members."""

import argparse
from hashlib import sha256
from pathlib import Path
from zlib import crc32

import cv2
import numpy as np
from PIL import Image

import expand_arkitscenes_temporal as cohort
from arkitscenes_roster import registered_candidates
from diagnose_arkitscenes_temporal_depth import project_depth
from prepare_arkitscenes_temporal import make_grid, pose_motion, rgb_remap
from screen_arkitscenes_fixed_lr import near_pixels
from screen_arkitscenes_x3 import intrinsics, prepare_poses
from verify_arkitscenes_fixed_lr import exact_time_check, rgb_tracks


def frame(site, asset, record, time, camera):
    path = site / asset / Path(record['member']).name
    raw = path.read_bytes()
    if (record['timestamp'] != time or len(raw) != record['bytes'] or
            f'{crc32(raw):08x}' != record['crc32'] or cohort.digest(path) != record['sha256']):
        raise ValueError(f'Frame provenance differs: {path}')
    flags = cv2.IMREAD_UNCHANGED if asset == 'highres_depth' else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    shape = tuple(camera[:2][::-1].astype(int))
    if flags == cv2.IMREAD_UNCHANGED:
        correct = image is not None and image.shape == shape and image.dtype == np.uint16
    else:
        correct = image is not None and image.shape == (*shape, 3) and image.dtype == np.uint8
    if not correct:
        raise ValueError(f'Frame dimensions or pixel format differ: {path}')
    return image


def replay(directory, row, expected_pixels=None):
    selected = row['selected']
    video = row['video']
    candidate = selected['candidate']
    t, u = (candidate[key] for key in ('anchor_time', 'shifted_time'))
    pair = next(p for p in row['candidates'] if (p['anchor_time'], p['shifted_time']) == (t, u))
    site = directory / f'arkit_{video}'
    for name, record in row['assets']['assets'].items():
        if cohort.digest(site / name) != record['sha256']:
            raise ValueError(f'Geometry asset differs: {video}/{name}')
    low, high = (intrinsics(site, label) for label in ('lowres_wide', 'wide'))
    poses = prepare_poses(np.loadtxt(site / 'lowres_wide.traj'), (t, u))
    lr = frame(site, 'lowres_wide', pair['rgb_source']['low']['frames'][0], t, low[t])
    a, b = (frame(site, 'wide', data, time, high[time])
            for data, time in zip(pair['rgb_source']['wide']['frames'], (t, u), strict=True))
    depths = tuple(frame(site, 'highres_depth', data, time, high[time])
                   for data, time in zip(pair['depth_source']['frames'], (t, u), strict=True))

    cx, cy = selected['context_origin_lr']
    lx, ly, ex, ey = selected['score_xyxy']
    context_grid, scale = near_pixels(make_grid((cx, cy), cohort.LR_CONTEXT), low[t], high[t])
    core = context_grid[ly:ey, lx:ex]
    images = dict(lr=cv2.cvtColor(lr[cy:cy + cohort.LR_CONTEXT, cx:cx + cohort.LR_CONTEXT], cv2.COLOR_BGR2RGB),
                  sync=rgb_remap(a, context_grid), moved=rgb_remap(b, context_grid))
    projections, visibility, motion = [], [], []
    for name, p, q in zip(cohort.POSE_NAMES, poses[t][0], poses[u][0], strict=True):
        xy, masks = project_depth(core, *depths, high[t], high[u], p, q)
        if any(masks[k].any() for k in ('missing_source', 'invalid_projection', 'outside')):
            raise ValueError(f'Scoring core has unsupported coordinates: {video}')
        projections.append(xy)
        visibility.append(float(masks['visible'].mean()))
        motion.append(float(np.linalg.norm(np.median((xy - core) * scale, axis=(0, 1)))))
        images['depth_' + name] = rgb_remap(b, xy)
    disagreement = float(np.linalg.norm((projections[0] - projections[1]) * scale, axis=-1).max())
    tracks = rgb_tracks(a, b, core, projections, scale)
    x, y = candidate['origin_lr']
    sync = exact_time_check((x, y), cohort.WINDOW, lr, a, low[t], high[t])
    if (not pose_motion(poses, t, u)['passed'] or not sync['pass'] or min(visibility) < 0.99 or
            min(motion) < 2 or max(motion) > 8 or disagreement > 0.75 or tracks['tracks'] < 8 or
            min(tracks['tracks_by_quadrant']) < 1 or max(tracks['rgb_p90_error_hr_px']) > 1.25):
        raise ValueError(f'Selected case failed replay geometry gates: {video}')
    pixel_hashes = {label: dict(shape=list(image.shape), dtype=str(image.dtype),
                               sha256=sha256(image.tobytes()).hexdigest()) for label, image in images.items()}
    if expected_pixels is None:
        for label, expected in images.items():
            record = selected['files'][label]
            path = directory / record['path']
            with Image.open(path) as stored:
                if cohort.digest(path) != record['sha256'] or not np.array_equal(expected, np.asarray(stored)):
                    raise ValueError(f'Reconstructed scoring image differs: {video}/{label}')
    elif pixel_hashes != expected_pixels:
        raise ValueError(f'Reconstructed scoring pixel hashes differ: {video}')
    return dict(video=video, visit=row['visit'], lag_ms=pair['lag_ms'], visible_fraction=visibility,
                geometric_displacement_hr_px=motion, pose_disagreement_hr_px=disagreement,
                optical_tracks=tracks['tracks'], optical_p90_error_hr_px=tracks['rgb_p90_error_hr_px'],
                verified_input_images=len(images), pixel_hashes=pixel_hashes)


def verify(directory):
    cv2.setNumThreads(1)
    protocol_path, input_path = (directory / file for file in ('protocol.json', 'inputs.json'))
    plan, inputs = (cohort.read(file) for file in (protocol_path, input_path))
    protocol_sha = cohort.digest(protocol_path)
    candidates = registered_candidates(directory, plan)
    expected = [(row['video_id'], row['visit_id']) for row in candidates]
    actual = [(row['video'], row['visit']) for row in inputs['visits']]
    if expected != actual or len(set(visit for _, visit in actual)) != len(actual):
        raise ValueError('Acquisition differs from the frozen visit list')
    if inputs['protocol_sha256'] != protocol_sha:
        raise ValueError('Acquisition manifest mismatches protocol SHA-256')
    cases = []
    for row in inputs['visits']:
        if cohort.read(directory / 'screened' / (row['video'] + '.json')) != row:
            raise ValueError(f'Audit differs from acquisition manifest: {row["video"]}')
        if row.get('error'):
            raise ValueError(f'Unresolved screening error: {row["video"]}')
        if row['selected']:
            result = replay(directory, row)
            del result['pixel_hashes']
            cases.append(result)
    result = dict(protocol_sha256=protocol_sha, inputs_sha256=cohort.digest(input_path),
                  screened_visits=len(actual), cases=cases)
    cohort.write(directory / 'acquisition_replay.json', result)
    print(f'{len(actual)} visits checked, {len(cases)} cases reconstructed from cached captures', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
