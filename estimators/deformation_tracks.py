"""Describe how the frozen LK acceptance filters respond to smooth deformation."""

import cv2
import numpy as np

import camera_deformation as camera
import local_deformation as warp
import registration_benchmarks as benchmarks
from holdout_run import atomic_dump, images_for, load, sha


DEST = warp.DEST.parent / 'deformation_tracks.json'


def initial_tracks(anchor):
    mask = np.zeros(anchor.shape, dtype=np.uint8)
    mask[benchmarks.ROI] = 255
    return cv2.goodFeaturesToTrack(anchor, 96, .01, 6, mask=mask, blockSize=5)


def audit_tracks(anchor, projected, points, center, diameter, added):
    counts = {'detected': 0 if points is None else len(points)}
    for field in ('forward_backward', 'accepted', 'in_disk_forward_backward', 'in_disk_accepted'):
        counts[field] = 0
    if counts['detected'] < benchmarks.MIN_TRACKS:
        return counts
    params = dict(winSize=(9, 9), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, matched, _ = cv2.calcOpticalFlowPyrLK(anchor, projected, points, None, **params)
    reverse, returned, _ = cv2.calcOpticalFlowPyrLK(projected, anchor, forward, None, **params)
    displacement = (forward - points)[:, 0, :]
    fb_error = np.linalg.norm((reverse - points)[:, 0, :], axis=1)
    fb_good = (matched[:, 0] > 0) & (returned[:, 0] > 0) & (fb_error <= .5)
    accepted = fb_good & (np.max(np.abs(displacement), axis=1) <= benchmarks.RADIUS_LR)
    estimated_hr = 4 * forward[:, 0, :] + 1.5 - added
    in_disk = np.linalg.norm(estimated_hr - center, axis=1) < diameter / 2
    counts['forward_backward'] = int(fb_good.sum())
    counts['accepted'] = int(accepted.sum())
    counts['in_disk_forward_backward'] = int((fb_good & in_disk).sum())
    counts['in_disk_accepted'] = int((accepted & in_disk).sum())
    return counts


def audit_cohort(folder, manifest, direction_key):
    plan = load(folder / 'lock.json')
    lock_hash = sha(folder / 'lock.json')
    rows = []
    for record in load(manifest)['records']:
        scene = record['scene']
        saved = load(folder / 'scenes' / f'{scene}.json')
        assert saved['stamp']['lock_sha256'] == lock_hash
        original = {row['condition']: row for row in saved['rows']}
        lr, hr, _, _ = images_for(record)
        anchor = warp.gray8(warp.luminance(lr))
        points = initial_tracks(anchor)
        direction = np.asarray(record[direction_key])
        center = plan['parameters'][scene]['center']
        local_direction = np.asarray(plan['parameters'][scene]['local_direction'])
        for diameter in warp.SIDES:
            weights = warp.profile(center, diameter)
            for amplitude in warp.AMPLITUDES:
                deformed = warp.deform(hr, weights, amplitude * local_direction)
                for translation in warp.TRANSLATIONS:
                    added = translation * direction
                    moving = warp.shift(deformed, added)
                    name = warp.condition(diameter, amplitude, translation)
                    archived = original[name]
                    assert warp.pixel_sha(moving) == archived['hr_pixel_sha256']
                    projected = warp.gray8(warp.project(warp.luminance(moving)))
                    tracks = audit_tracks(anchor, projected, points, center, diameter, added)
                    assert tracks['accepted'] == archived['n_tracks'], (scene, name)
                    rows.append(dict(scene=scene, condition=name, **tracks))
    strata = {}
    for name in sorted({row['condition'] for row in rows}):
        cases = [row for row in rows if row['condition'] == name]
        strata[name] = {field: float(np.mean([r[field] for r in cases])) for field in (
            'detected', 'forward_backward', 'accepted', 'in_disk_forward_backward', 'in_disk_accepted')}
    return dict(lock_sha256=lock_hash, rows=rows, mean_tracks_per_scene=strata)


if __name__ == '__main__':
    camera.verify_inputs()
    cv2.setNumThreads(1)
    report = dict(scope='Post hoc diagnostic. In-disk counts label accepted or forward-backward '
                        'track endpoints by position relative to the known support after subtracting '
                        'the injected global translation. The count describes spatial support, not '
                        'correctness of individual tracks.', cohorts={})
    for label, folder, manifest, key in (('RealSR', camera.DEST, camera.INPUTS, 'hr_direction'),
                                         ('COCO', warp.DEST, warp.INPUTS, 'direction')):
        cohort = audit_cohort(folder, manifest, key)
        report['cohorts'][label] = cohort
        atomic_dump(DEST, report)
        print(f'{label}: {len(cohort["rows"])} rows match the frozen LK counts', flush=True)
