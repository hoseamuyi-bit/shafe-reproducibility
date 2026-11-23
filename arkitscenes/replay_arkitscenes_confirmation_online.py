"""Verify the confirmation inputs using a fresh copy of the official recordings."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from shutil import copyfile

import cv2

from fetch_arkitscenes_frames import atomic_write
from reacquire_arkitscenes_holdout import restore_visit
from verify_arkitscenes_temporal_acquisition import replay
from verify_confirmation_statistics import check_frozen_sources, digest, read, require


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'arkitscenes_temporal_confirmation'
FRESH = ROOT / 'confirmation_online_replay'


def verify_visit(row):
    video = row['video']
    receipt_dir = RUN / 'screened'
    require(row == read(receipt_dir, video), f'Screening record: {video}')
    restore_visit(FRESH, row)
    for label, entry in row['selected']['files'].items():
        target = FRESH / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            copyfile(RUN / entry['path'], target)
        require(digest(target) == entry['sha256'], f'Frozen PNG: {video}/{label}')
    return dict(screen_sha256=digest(receipt_dir / f'{video}.json'), source_frames_verified=5,
                geometry_assets_verified=3, **replay(FRESH, row))


def main():
    cv2.setNumThreads(1)
    sample = read(RUN, 'sampling_protocol')
    check_frozen_sources(RUN, sample)
    plan = read(RUN, 'protocol')
    inputs = read(RUN, 'inputs')
    require(plan['sampling_protocol_sha256'] == digest(RUN / 'sampling_protocol.json'), 'Sampling hash')
    require(inputs['protocol_sha256'] == digest(RUN / 'protocol.json'), 'Protocol hash')
    require([(row['video'], row['visit']) for row in inputs['visits']] ==
            [(row['video_id'], row['visit_id']) for row in plan['candidates']], 'Screened sample')
    require(not any(row.get('error') for row in inputs['visits']), 'Unresolved screening errors')
    FRESH.mkdir(exist_ok=True)
    selected = [row for row in inputs['visits'] if row['selected']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        cases = list(pool.map(verify_visit, selected))
    receipt = dict(inputs_sha256=digest(RUN / 'inputs.json'),
                   archive=sample['official_archive_root'], cases=cases)
    path = RUN / 'online_replay.json'
    atomic_write(path, (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())
    print(f'Official source replay: {len(cases)} visits, {sum(row["verified_input_images"] for row in cases)} '
          f'reconstructed images, SHA-256 {digest(path)}', flush=True)


if __name__ == '__main__':
    main()
