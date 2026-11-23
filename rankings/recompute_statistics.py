"""Recompute the frozen analysis of the archived score records without images or a GPU."""
import argparse
import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

import analyze_ranking
from camera_ranking_statistics import analyze_camera
from ranking_common import LOCK, ROOT, file_sha, load
from run_frozen_analysis import run


def validate_archive():
    manifest = load(ROOT / 'checksums.json')
    for name, expected in manifest.items():
        if file_sha(ROOT / name) != expected:
            raise ValueError(f'Archived file changed: {name}')

    lock, digest = load(LOCK), file_sha(LOCK)
    if digest != '376c300f452ad3802d59480979ecac281be5fa384dad055f31ff08196c344013':
        raise ValueError('Formal lock digest mismatch')
    records = load(ROOT / 'inputs/manifest.json')['records']
    if len(records) != 144 or len({record['scene'] for record in records}) != 144:
        raise ValueError('The archive must contain 144 unique frozen scenes')
    for source, expected in lock['sha256'].items():
        name = PureWindowsPath(source).name
        local = ROOT / ('inputs/manifest.json' if name == 'manifest.json' else name)
        if local.is_file() and file_sha(local) != expected:
            raise ValueError(f'Frozen snapshot changed: {local}')

    camera = ROOT / 'camera_ranking'
    camera_lock = load(camera / 'lock.json')
    if file_sha(camera / 'lock.json') != '73b120dc87fec29cf5d7ff9fd16e25135fe6bdca50bccbc32c500d5b4c8ce411':
        raise ValueError('Camera lock digest mismatch')
    if camera_lock['coco_lock_sha256'] != digest:
        raise ValueError('Camera parent lock digest mismatch')
    for source, expected in camera_lock['sha256'].items():
        name = PureWindowsPath(source).name
        local = (camera if name == 'batch_inputs.json' else ROOT) / name
        if file_sha(local) != expected:
            raise ValueError(f'Frozen camera snapshot changed: {local}')
    return lock, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional destination for the recalculated analysis JSON')
    args = parser.parse_args()

    lock, digest = validate_archive()
    archived = load(ROOT / 'formal/analysis.json')
    # Use the original frozen statistical code and a separate output location.
    # Images and weight files at absolute paths are verified in the pixel-level audit.
    analyze_ranking.verify_lock = lambda: (lock, digest)
    with TemporaryDirectory() as folder:
        fresh_path = Path(folder) / 'analysis.json'
        write_json = analyze_ranking.atomic_json
        analyze_ranking.atomic_json = lambda _path, report: write_json(fresh_path, report)
        run()
        fresh = load(fresh_path)

    def equal(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) <= 1e-10
        return a == b

    if not equal(fresh, archived):
        raise ValueError('Recomputed analysis differs from the archived results')
    camera = analyze_camera()
    if not equal(camera, load(ROOT / 'camera_ranking/analysis.json')):
        raise ValueError('Recomputed camera analysis differs from the archived results')
    if args.output:
        write_json(args.output, {'coco': fresh, 'realsr': camera})
    shafe = camera['pooled']['native_discordance']['strict']['direct vs lk_lr']['SHAFE']
    print(json.dumps({'analysis_matches': True, 'scene_count': fresh['scenes'],
                      'primary_effect_pp': fresh['primary'], 'passes': fresh['success'],
                      'camera_analysis_matches': True, 'camera_scene_count': camera['pooled']['scenes'],
                      'camera_native_shafe_opposite': shafe['opposite'], 'camera_pairs': shafe['total']}))


if __name__ == '__main__':
    main()
