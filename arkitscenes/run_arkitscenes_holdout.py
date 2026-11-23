"""Acquire a disjoint, score-blind ARKitScenes replication from Apple archives."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
from random import Random
from shutil import disk_usage

import cv2

import screen_arkitscenes_training_temporal as train


screen = train.screen
ROOT = screen.ROOT
OUT = ROOT / 'arkitscenes_temporal_holdout'
PARENT = train.OUT / 'protocol.json'
PARENT_SAMPLE = train.OUT / 'sampling_protocol.json'
DRAW = 240
SEED = 20261124


def freeze_sample():
    parent, original = map(screen.read, (PARENT, PARENT_SAMPLE))
    if (parent['sampling_protocol_sha256'] != screen.digest(PARENT_SAMPLE)
            or parent['index_sha256'] != screen.digest(train.INDEX)
            or any(screen.digest(ROOT / name) != sha for name, sha in parent['hashes'].items())):
        raise ValueError('Archived acquisition differs from its frozen sources')
    rows = sorted(csv.DictReader(train.INDEX.read_text(encoding='utf-8').splitlines()),
                  key=lambda row: row['video_id'])
    excluded = {row['visit_id'] for row in original['candidates']}
    excluded.update(row['visit_id'] for row in rows if row['fold'] == 'Validation')
    previous = {}
    for manifest in sorted(ROOT.glob('arkitscenes_temporal*/inputs.json')):
        if manifest.parent == OUT:
            continue
        previous[manifest.parent.name] = screen.digest(manifest)
        excluded.update(row['visit'] for row in screen.read(manifest)['visits'])
    available = {}
    for row in rows:
        if row['fold'] == train.FOLD and row['visit_id'] not in excluded:
            available.setdefault(row['visit_id'], row)
    sample = [available[visit] for visit in Random(SEED).sample(sorted(available), DRAW)]
    settings = dict(parent['settings'], scope='Disjoint replication of measured temporal reference mismatch.')
    hashes = dict(parent['hashes'], **{Path(__file__).name: screen.digest(Path(__file__))})
    return train.locked('sampling_protocol.json', dict(
        index_sha256=screen.digest(train.INDEX), parent_protocol_sha256=screen.digest(PARENT),
        earlier_input_sha256=previous, hashes=hashes, settings=settings, available_visits=len(available),
        sample_seed=SEED, draw=DRAW, sample=sample,
        selection='Random disjoint visit sample, lowest video ID per visit, then all pose-qualified '
                  'visits screened with the archived depth and RGB rules; one window per visit.',
        endpoint=dict(baseline='four-model synchronous fixed-grid ordering',
                      comparison='direct minus lk_lr strict reversal rates (six pairs per visit)',
                      primary=dict(metric='SHAFE', cutoff_radius=50, practical_tie=0.0005),
                      sensitivity=dict(metric='SHAFE', cutoff_radius=50 * 88 / 448, practical_tie=0.0005,
                                       additional_metrics=['PSNR', 'SSIM', 'ST-LPIPS']),
                      geometry='Mean visit-paired direct minus LK p90 geometric residual for both '
                               'pose interpolants, with equal weights for direct scores from the four models.',
                      uncertainty=dict(unit='visit', bootstrap_draws=10000, seed=20261122)))), parent


def collect(rows, task, workers, label):
    results = {}
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, row): row['video_id'] for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            video = futures[future]
            try:
                result = future.result()
                results[video] = result
                if result.get('error'):
                    errors.append((video, result['error']))
            except (OSError, ValueError, KeyError, cv2.error, train.subprocess.CalledProcessError) as error:
                errors.append((video, str(error)))
            if count % 10 == 0 or count == len(rows):
                print(f'{label}: {count}/{len(rows)} checked; {len(errors)} errors', flush=True)
    if errors:
        screen.write(OUT / f'{label}_errors.json', errors)
        raise RuntimeError(f'{label}: retry {len(errors)} unresolved visits; see acquisition log')
    return [results[row['video_id']] for row in rows]


def run():
    OUT.mkdir(exist_ok=True)
    for folder in ('pose_priority', 'screened', 'visit_plans'):
        (OUT / folder).mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    train.OUT = OUT
    sampling, parent = freeze_sample()
    print('Locked disjoint draw', DRAW, screen.digest(sampling), flush=True)
    draw = screen.read(sampling)
    ranking = collect(draw['sample'], train.priority, 8, 'pose')
    ranking.sort(key=lambda row: (-row['priority_centers'], -row['priority_pairs'], row['video_id']))
    candidates = [row for row in ranking if row['priority_pairs']]
    path = train.locked('protocol.json', dict(
        sampling_protocol_sha256=screen.digest(sampling), ranking=ranking, candidates=candidates,
        hashes=draw['hashes'], settings=draw['settings'], index_sha256=draw['index_sha256'],
        expansion=dict(parent['expansion'], max_visits=len(candidates)), endpoint=draw['endpoint']))
    protocol = screen.digest(path)
    print('Frozen', len(candidates), 'pose-qualified visits for depth and RGB', protocol, flush=True)
    screen.OUT, screen.FrameArchive, screen.acquire_geometry = OUT, train.train_archive, train.acquire

    def acquire(row):
        if disk_usage(OUT).free < 1024**3:
            raise OSError('Less than 1 GiB of free disk space before frame extraction')
        result = train.run_screen(row, protocol)
        if result['selected']:
            print('Selected', result['video'], 'visit', result['visit'], flush=True)
        return result

    final = collect(candidates, acquire, 4, 'frames')
    screen.write(OUT / 'inputs.json', dict(protocol_sha256=protocol, visits=final))
    print(f'{sum(bool(row["selected"]) for row in final)} eligible of {DRAW} disjoint visits', flush=True)


if __name__ == '__main__':
    run()
