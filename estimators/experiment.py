"""Scene-level design, deterministic inputs, and restartable VOSR inference."""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dev_pipeline import CONFIG, HERE, OLD, degrade, dump, local_sr, pixel_sha, read, save, sha, shift

SPLITS = ('dev', 'calibration', 'formal', 'external')
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
VOSR = OLD / 'work/vosr/run_baseline.py'
VOSR_ARGS = ['--scale', '4', '--precision', 'fp32', '--seed', '42']
CONDITIONS = ('zero', 'hr_global', 'sr_global', 'both_global', 'sr_local',
              'hr2_sr_local', 'hr_replacement', 'hr2_replacement')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(scenes, split):
    for dataset in sorted({s['dataset'] for s in scenes}):
        rows = sorted((s for s in scenes if s['dataset'] == dataset), key=lambda s: s['id'])
        rng = random.Random(f'20261122|{split}|{dataset}|injection-v2')
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            magnitude = (.5, 2, 6)[i % 3]
            dh = rng.choice(DIRECTIONS)
            ds = rng.choice(DIRECTIONS)
            x, y = (rng.randint(64, 320) for _ in range(2))
            tx, ty = (256 * rng.randrange(2) for _ in range(2))
            sx = 0 if tx else row['width'] - 256
            sy = 0 if ty else row['height'] - 256
            ox, oy = (u + 24 for u in row['render_crop_xyxy'][:2])
            overlap = max(0, min(ox+tx+256, sx+256)-max(ox+tx, sx))
            overlap *= max(0, min(oy+ty+256, sy+256)-max(oy+ty, sy))
            yield dict(scene=row['id'], dataset=dataset, group=row['group'], source=row['source'],
                       source_sha256=row['sha256'], render_crop_xyxy=row['render_crop_xyxy'],
                       magnitude=magnitude, hr_direction=list(dh), sr_direction=list(ds),
                       sr_local_xy=[x,y], hr_replace_xy=[tx,ty], replacement_source_xy=[sx,sy],
                       replacement_overlap_px=overlap)


def prepare(split):
    base = HERE / split
    path = HERE / 'data/manifest_frozen.json'
    if not path.exists() and split == 'dev':
        path = HERE / 'data/manifest_provisional.json'
    manifest = load(path)
    scene_plans = list(plan([s for s in manifest['scenes'] if s['split'] == split], split))
    assert scene_plans
    records = []
    for row in scene_plans:
        assert sha(row['source']) == row['source_sha256']
        full = read(row['source'])
        x, y, x1, y1 = row['render_crop_xyxy']
        render = full[y:y1,x:x1]
        assert render.shape == (560,560,3)
        folder = base / 'inputs' / row['scene']
        hr_path = folder / 'hr.png'
        save(hr_path, render[24:536,24:536])
        _, mask = local_sr(np.zeros((512,512,3), dtype='float32'), *row['sr_local_xy'], (0,0))
        save(folder / 'sr_local_mask.png', mask*255)
        tx, ty = row['hr_replace_xy']
        mask = np.zeros((512,512), dtype=np.uint8)
        mask[ty:ty+256,tx:tx+256] = 255
        save(folder / 'hr_replacement_mask.png', mask)
        for kind in ('normal', 'strong'):
            lr = degrade(render, row['scene'], kind == 'strong')
            filename = f"{row['scene']}__{kind}.png"
            lr_path = base / 'lr' / filename
            save(lr_path, lr)
            rec = dict(row, degradation=kind, hr=str(hr_path), hr_sha256=sha(hr_path),
                       lr=str(lr_path), lr_sha256=sha(lr_path), lr_pixel_sha256=pixel_sha(lr),
                       sr=str(base / 'vosr/images' / filename),
                       sr_mask_sha256=sha(folder/'sr_local_mask.png'),
                       hr_mask_sha256=sha(folder/'hr_replacement_mask.png'))
            records.append(rec)
    dump(base / 'batch_inputs.json', dict(config=CONFIG, split=split, design_sha256=sha(path), records=records))
    print(f'{split}: {len(scene_plans)} scenes, {len(records)} LR inputs', flush=True)


def infer(split):
    base = HERE / split
    inputs = load(base / 'batch_inputs.json')['records']
    provenance = load(HERE / 'environment_verified.json')
    weights = {row['path']:row['sha256'] for row in provenance['vosr_weights']}
    runner_hash = sha(VOSR)
    if runner_hash != provenance['code_and_records'][str(VOSR)]:
        raise ValueError('VOSR entry point changed since verification')
    pending = []
    for row in inputs:
        if sha(row['lr']) != row['lr_sha256']:
            raise ValueError(f"LR changed: {row['lr']}")
        path = Path(row['sr'])
        check = path.with_suffix('.json')
        config = dict(lr_sha256=row['lr_sha256'], runner_sha256=runner_hash,
                      args=VOSR_ARGS, weights=weights)
        if path.exists() and check.exists() and load(check) == dict(config, sr_sha256=sha(path)):
            continue
        pending.append((row, path, check, config))
    if not pending:
        return
    # Use bounded chunks so a failed invocation leaves prior chunks verified.
    for path, expected in weights.items():
        with open(path, 'rb') as stream:
            import hashlib
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'VOSR weight changed: {path}')
    for start in range(0, len(pending), 100):
        chunk = pending[start:start+100]
        retry = 0
        while (base / f'vosr_attempt_{start:04d}_{retry:03d}').exists():
            retry += 1
        batch = base / f'vosr_attempt_{start:04d}_{retry:03d}'
        lr_dir = batch / 'input'
        lr_dir.mkdir(parents=True)
        for row, path, _, _ in chunk:
            shutil.copyfile(row['lr'], lr_dir / path.name)
        subprocess.run([sys.executable, '-B', str(VOSR), '--input', str(lr_dir),
                        '--output', str(batch), *VOSR_ARGS], check=True)
        run = load(batch / 'run.json')
        assert run['precision'] == 'fp32' and run['upscale'] == 4 and run['seed_per_image'] == 42
        for _, path, check, config in chunk:
            generated = batch / 'images' / path.name
            assert read(generated).shape == (512,512,3)
            path.parent.mkdir(parents=True, exist_ok=True)
            generated.replace(path)
            dump(check, dict(config, sr_sha256=sha(path)))


def variants(record, hr, sr, full=None):
    dh, ds = np.array(record['hr_direction']), np.array(record['sr_direction'])
    vh, vs = dh*record['magnitude'], ds*record['magnitude']
    zero = np.zeros(2)
    h0, h1 = shift(hr, zero), shift(hr, vh)
    s0, s1 = shift(sr, zero), shift(sr, vs)
    yield 'zero', h0, s0, zero, zero
    yield 'hr_global', h1, s0, vh, zero
    yield 'sr_global', h0, s1, zero, vs
    yield 'both_global', h1, s1, vh, vs
    if full is None:
        return
    changed, _ = local_sr(s0, *record['sr_local_xy'], ds*4)
    local = shift(changed, zero)
    tx, ty = record['hr_replace_xy']
    sx, sy = record['replacement_source_xy']
    replaced = h0.copy()
    replaced[ty:ty+256,tx:tx+256] = full[sy:sy+256,sx:sx+256]
    yield 'sr_local', h0, local, zero, zero
    yield 'hr2_sr_local', shift(hr, dh*2), local, dh*2, zero
    yield 'hr_replacement', shift(replaced, zero), s0, zero, zero
    yield 'hr2_replacement', shift(replaced, dh*2), s0, dh*2, zero


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'infer', 'prepare-infer'))
    parser.add_argument('split', choices=SPLITS)
    args = parser.parse_args()
    if args.split in ('formal','external'):
        from freeze import verify, verify_inputs
        frozen = verify()
    if args.action in ('prepare', 'prepare-infer'):
        if args.split=='external':
            parser.error('Use camera.py to prepare measured camera pairs')
        prepare(args.split)
    if args.action in ('infer', 'prepare-infer'):
        if args.split=='formal':
            verify_inputs(load(HERE/'formal/batch_inputs.json'),frozen)
        infer(args.split)
