"""Fetch the calibrated geometry assets for a fixed public validation cohort."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import file_digest
import json
from pathlib import Path
import random
import subprocess
from zipfile import ZipFile

from fetch_arkitscenes_frames import BASE, atomic_write


ASSETS = ('wide_intrinsics.zip', 'vga_wide_intrinsics.zip', 'lowres_wide_intrinsics.zip',
          'lowres_wide.traj', 'highres_depth.zip')


def download_video(root, row, asset_names=ASSETS):
    video = row['video_id']
    out = root / f'arkit_{video}'
    out.mkdir(exist_ok=True)
    assets = {}
    for name in asset_names:
        url = f'{BASE}/raw/Validation/{video}/{name}'
        path = out / ('raw_highres_depth.zip' if name == 'highres_depth.zip' else name)
        partial = path.with_suffix(path.suffix + '.part')
        if not path.exists():
            subprocess.run(['curl.exe', '-fLsS', '--retry', '2', '--connect-timeout', '15',
                            '--max-time', '260', '--continue-at', '-', '-o', str(partial), url], check=True)
        payload = path if path.exists() else partial
        record = dict(url=url, bytes=payload.stat().st_size)
        if name.endswith('.zip'):
            with ZipFile(payload) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f'ZIP CRC failed: {video}/{name}/{bad}')
                record['crc_verified_members'] = len(archive.infolist())
        else:
            with payload.open('rt', encoding='ascii') as stream:
                if not all(len(line.split()) == 7 for line in stream if line.strip()) or not record['bytes']:
                    raise ValueError(f'Invalid trajectory: {video}/{name}')
        with payload.open('rb') as stream:
            record['sha256'] = file_digest(stream, 'sha256').hexdigest()
        if payload == partial:
            partial.replace(path)
        assets[path.name] = record
    manifest = dict(**row, assets=assets)
    atomic_write(out / 'geometry_source.json', (json.dumps(manifest, indent=2) + '\n').encode())
    if 'raw_highres_depth.zip' in assets:
        counts = assets['raw_highres_depth.zip']['crc_verified_members']
        print(f'{video} visit {row["visit_id"]}: {counts} depth ZIP members verified', flush=True)
    return manifest


def main(root):
    index = root / 'arkitscenes_upsampling_split.csv'
    with index.open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    visits = {}
    for row in records:
        if row['fold'] == 'Validation' and row['visit_id'] != '381658':
            visits.setdefault(row['visit_id'], []).append(row)
    sample = [min(visits[visit], key=lambda row: row['video_id'])
              for visit in random.Random(261122).sample(sorted(visits), 6)]
    with index.open('rb') as stream:
        index_sha256 = file_digest(stream, 'sha256').hexdigest()
    selection = dict(seed=261122, index_sha256=index_sha256, eligible_visits=len(visits), videos=sample,
                     method='uniform sample of sorted eligible validation visit IDs, lowest video ID per visit')
    atomic_write(root / 'arkitscenes_validation_selection.json', (json.dumps(selection, indent=2) + '\n').encode())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_video, root, row) for row in sample]
        errors = []
        for row, future in zip(sample, futures, strict=True):
            try:
                future.result()
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                errors.append(f'{row["video_id"]}: {exc}')
    if errors:
        raise RuntimeError('\n'.join(errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
