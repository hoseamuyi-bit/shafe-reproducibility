"""Fetch the selected original ARKitScenes frames from the official ZIP ranges."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from audit_arkitscenes_holdout_endpoints import HOLDOUT, load, sha, verify_roster
from fetch_arkitscenes_frames import BASE, FrameArchive, atomic_write


def restore_geometry(site, video, records):
    for name in ('wide_intrinsics.zip', 'lowres_wide_intrinsics.zip', 'lowres_wide.traj'):
        record = records[name]
        url = f'{BASE}/raw/Training/{video}/{name}'
        path = site / name
        if record['url'] != url:
            raise ValueError(f'Unexpected source URL for {video}/{name}')
        if not path.exists():
            with urlopen(url, timeout=30) as response:
                data = response.read(record['bytes'] + 1)
            if len(data) != record['bytes']:
                raise ValueError(f'Geometry asset size changed: {video}/{name}')
            atomic_write(path, data)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Geometry asset differs from the download receipt: {video}/{name}')


def restore_visit(destination, row):
    video = row['video']
    if len(video) != 8 or not video.isascii() or not video.isdecimal():
        raise ValueError(f'Invalid ARKitScenes video ID: {video}')
    site = destination / f'arkit_{video}'
    site.mkdir(exist_ok=True)
    restore_geometry(site, video, row['assets']['assets'])
    window = row['selected']['candidate']
    times = (window['anchor_time'], window['shifted_time'])
    pair, = (p for p in row['candidates'] if (p['anchor_time'], p['shifted_time']) == times)
    sources = {'highres_depth': pair['depth_source'], 'lowres_wide': pair['rgb_source']['low'],
               'wide': pair['rgb_source']['wide']}
    for asset, source in sources.items():
        archive = FrameArchive(site, 'Training', asset)
        if ((archive.url, archive.etag, archive.tail_sha256)
                != (source['url'], source['etag'], source['zip_tail_sha256'])):
            raise ValueError(f'Official ZIP differs from the download receipt: {video}/{asset}')
        for expected in source['frames']:
            if archive.extract(expected['timestamp']) != expected:
                raise ValueError(f'Official ZIP member differs from the receipt: {expected["member"]}')
    print(f'{video}: three geometry assets and five source frames verified', flush=True)


def run(destination, video_ids):
    verify_roster(HOLDOUT)
    inputs_path = HOLDOUT / 'inputs.json'
    summary = load(HOLDOUT / 'holdout_endpoint_summary.json')
    if sha(inputs_path) != summary['artifact_sha256']['inputs']:
        raise ValueError('Input manifest differs from the verified measurements')
    selected = {row['video']: row for row in load(inputs_path)['visits'] if row['selected']}
    unknown = set(video_ids) - selected.keys()
    if unknown:
        raise ValueError(f'Videos outside the selected cohort: {sorted(unknown)}')
    destination.mkdir(parents=True, exist_ok=True)
    for video in dict.fromkeys(video_ids or selected):
        restore_visit(destination, selected[video])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('video_ids', nargs='*', help='Selected video IDs. By default, download all selected videos.')
    args = parser.parse_args()
    run(args.destination, args.video_ids)
