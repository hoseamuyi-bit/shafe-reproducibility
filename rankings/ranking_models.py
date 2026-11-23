"""Generate RGB ×4 SR outputs with scene-level cache integrity."""
import argparse
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from ranking_common import ASSETS, MODELS, NEXT, ROOT, atomic_json, file_sha, load, verify_lock
from dev_pipeline import OLD, read, save
from experiment import VOSR, VOSR_ARGS
from model_transfer import WEIGHT as REAL_WEIGHT

SWIN_WEIGHT = ASSETS / 'SwinIR-S_x4.pth'


def dev_records():
    records = [r for r in load(NEXT / 'dev/batch_inputs.json')['records']
               if r['degradation'] == 'strong']
    records = [next(r for r in records if r['magnitude'] == a) for a in (.5, 2, 6)]
    return [dict(r, split='old_dev', hr_added=[0, r['magnitude']], sr_added=[r['magnitude'], 0])
            for r in records]


def artifact(base, model, scene):
    return base / 'sr' / model / (scene + '.png')


def stamp(path, row, config):
    if read(path).shape != (512, 512, 3):
        raise ValueError(f'SR image has an unexpected size: {path}')
    atomic_json(path.with_suffix('.json'),
                dict(config=config, lr_sha256=row['lr_sha256'], image_sha256=file_sha(path)))


def pending_outputs(records, base, model, config):
    for row in records:
        path = artifact(base, model, row['scene'])
        check = path.with_suffix('.json')
        if not (path.is_file() and check.is_file() and
                load(check) == dict(config=config, lr_sha256=row['lr_sha256'], image_sha256=file_sha(path))):
            yield row, path


def setup_cuda(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def run_vosr(pending, base, config):
    for start in range(0, len(pending), 72):
        chunk = pending[start:start + 72]
        index = sum(1 for _ in base.glob('vosr_attempt_*'))
        attempt = base / f'vosr_attempt_{index:03}'
        batch = attempt / 'input'
        batch.mkdir(parents=True)
        for row, out in chunk:
            shutil.copyfile(row['lr'], batch / out.name)
        subprocess.run([sys.executable, '-B', str(VOSR), '--input', str(batch),
                        '--output', str(attempt), *VOSR_ARGS], check=True)
        metadata = load(attempt / 'run.json')
        if (metadata['precision'], metadata['upscale'], metadata['seed_per_image']) != ('fp32', 4, 42):
            raise ValueError('VOSR ran with unexpected settings')
        for row, out in chunk:
            out.parent.mkdir(parents=True, exist_ok=True)
            (attempt / 'images' / out.name).replace(out)
            stamp(out, row, config)


def torch_outputs(model, pending, config):
    import torch
    setup_cuda(torch)
    if model == 'Real-ESRGAN':
        from realesrgan_x4 import RRDBNetX4
        network = RRDBNetX4()
        weight = torch.load(REAL_WEIGHT, map_location='cpu', weights_only=True)['params_ema']
        layout = torch.channels_last
    else:
        sys.path[:0] = [str(ASSETS / 'SwinIR'), str(OLD / 'work/shafe_formal/deps')]
        from models.network_swinir import SwinIR
        network = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
                         depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                         mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        weight = torch.load(SWIN_WEIGHT, map_location='cpu', weights_only=True)['params']
        layout = torch.contiguous_format
    network.load_state_dict(weight, strict=True)
    network = network.eval().to(device='cuda', memory_format=layout)
    with torch.inference_mode():
        for i, (row, out) in enumerate(pending, 1):
            tensor = torch.from_numpy(read(row['lr'])).permute(2, 0, 1)[None].to('cuda').div(255)
            if model == 'SwinIR-S':
                # Official reflection-by-concatenation and crop, including one full window for 128 px.
                tensor = torch.cat((tensor, tensor.flip((2,))), dim=2)[:, :, :136, :]
                tensor = torch.cat((tensor, tensor.flip((3,))), dim=3)[:, :, :, :136]
            prediction = network(tensor.contiguous(memory_format=layout))[0, :, :512, :512]
            save(out, prediction.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy())
            stamp(out, row, config)
            if i % 24 == 0 or i == len(pending):
                print(f'{model}: {i}/{len(pending)}', flush=True)
    print(f'{model}: peak GPU allocation {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB', flush=True)


def run(cohort, chosen):
    base = ROOT / cohort
    if cohort == 'formal':
        _, config = verify_lock()
        records = load(ROOT / 'inputs/manifest.json')['records']
    else:
        records = dev_records()
        config = f'dev:{file_sha(__file__)}'
    for row in records:
        if file_sha(row['lr']) != row['lr_sha256']:
            raise ValueError(f'LR input changed: {row["scene"]}')
    for model in chosen:
        pending = list(pending_outputs(records, base, model, config))
        print(f'{model}: {len(pending)} pending', flush=True)
        if not pending:
            continue
        if model == 'VOSR':
            run_vosr(pending, base, config)
        elif model == 'Bicubic':
            for row, out in pending:
                with Image.open(row['lr']) as image:
                    save(out, np.asarray(image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)))
                stamp(out, row, config)
        else:
            torch_outputs(model, pending, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('dev', 'dev_repeat', 'formal'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    args = parser.parse_args()
    run(args.cohort, args.models)
