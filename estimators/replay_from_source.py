"""Rebuild selected frozen inputs, rerun VOSR, and verify actions and SHAFE."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import cv2
import torch

from replay_frozen import ROOT, compare_record, read_cohort
from dev_pipeline import OLD, degrade, dump, pixel_sha, read, save, sha
from experiment import VOSR, VOSR_ARGS, load
from freeze import GATES, LOCK, verify
from score import BATCH_SIZE, DEVICE


def crop(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'SHA-256 differs: {path}')
    return path


def stage(split, row, folder):
    full = read(checked(row['source'], row['source_sha256']))
    if split == 'formal':
        render = crop(full, row['render_crop_xyxy'])
        lr = degrade(render, row['scene'], row['degradation'] == 'strong')
        hr = render[24:536, 24:536]
    else:
        camera_lr = read(checked(row['camera_lr_source'], row['camera_lr_source_sha256']))
        lr = crop(camera_lr, row['lr_crop_xyxy'])
        hr = crop(full, row['hr_crop_xyxy'])

    lr_path = folder / 'input' / Path(row['lr']).name
    hr_path = folder / 'hr' / lr_path.name
    save(lr_path, lr)
    save(hr_path, hr)
    checked(lr_path, row['lr_sha256'])
    checked(hr_path, row['hr_sha256'])
    return dict(row, lr=str(lr_path), hr=str(hr_path), sr=str(folder / 'vosr/images' / lr_path.name))


def replay(folder):
    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric
    from measure import measure_record

    folder.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    frozen = verify()
    cohorts = {split: read_cohort(split, frozen) for split in ('formal', 'external')}
    provenance = load(ROOT / 'environment_verified.json')
    runner_hash = provenance['code_and_records'][str(VOSR)]
    checked(VOSR, runner_hash)
    weights = {item['path']: item['sha256'] for item in provenance['vosr_weights']}
    for path, digest in weights.items():
        checked(path, digest)
    checked(DEFAULT_WEIGHT, frozen['weights']['shafe']['sha256'])

    selected = []
    for split, (records, _, _, _) in cohorts.items():
        scene_ids = defaultdict(set)
        for row in records:
            scene_ids[row['dataset']].add(row['scene'])
        middle_ids = {sorted(ids)[len(ids) // 2] for ids in scene_ids.values()}
        selected.extend((split, row, stage(split, row, folder)) for row in records if row['scene'] in middle_ids)

    command = [sys.executable, '-B', str(VOSR), '--input', str(folder / 'input'),
               '--output', str(folder / 'vosr'), *VOSR_ARGS]
    subprocess.run(command, check=True)
    run = load(folder / 'vosr/run.json')
    if run['input_count'] != len(selected) or (run['upscale'], run['precision'], run['seed_per_image']) != (4, 'fp32', 42):
        raise ValueError('VOSR runtime configuration differs')

    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)
    gates = load(GATES)

    def evaluate(pairs):
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    checks = []
    for split, original, fresh in selected:
        fresh_path = Path(fresh['sr'])
        old_sr = pixel_sha(read(original['sr']))
        fresh_sr = pixel_sha(read(fresh_path))
        if fresh_sr != old_sr:
            raise ValueError(f'{fresh_path}: VOSR output pixels differ')
        dump(fresh_path.with_suffix('.json'), dict(
            lr_sha256=fresh['lr_sha256'], runner_sha256=runner_hash, args=VOSR_ARGS,
            weights=weights, sr_sha256=sha(fresh_path),
        ))
        observed = measure_record(fresh, reference=False, include_direct=True)
        check = compare_record(fresh, observed, cohorts[split][1], gates, evaluate)
        checks.append(dict(check, split=split, dataset=original['dataset'], sr_pixel_sha256=fresh_sr,
                           lr_sha256=sha(fresh['lr']), hr_sha256=sha(fresh['hr'])))
        print(f'{check["scene"]} {check["degradation"]}: source-to-score replay matches', flush=True)
    dump(folder / 'source_replay.json', dict(
        completed_utc=datetime.now(timezone.utc).isoformat(),
        selection='Middle scene by sorted identifier in each dataset or camera, all of its degradation settings',
        runner_sha256=sha(__file__), formal_lock_sha256=sha(LOCK),
        vosr_runner_sha256=runner_hash, vosr_weights=weights, run=run,
        torch_version=torch.__version__, checks=checks,
        inputs={split: data[2] for split, data in cohorts.items()},
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    replay(parser.parse_args().output.resolve())
