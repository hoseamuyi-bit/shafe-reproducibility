"""Evaluate frozen alignment actions on a common valid field of view."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

from ranking_common import ASSETS, CONDITIONS, METRICS, MODELS, NEXT, ROOT, atomic_json, file_sha, load, verify_lock
from dev_pipeline import OLD, luminance, pixel_sha, project, read, shift
from pixel_metric_audit import psnr, ssim
from registration import candidate, direct_candidate
from robust_controls import gray8, lucas_kanade
from ranking_models import artifact, dev_records, pending_outputs, setup_cuda

ROI = np.s_[32:480, 32:480]
BATCH = 2
KERNEL = cv2.getGaussianKernel(11, 1.5, cv2.CV_64F)
ST_RUNTIME = ASSETS / 'stlpips_runtime'
ST_WEIGHT = ST_RUNTIME / 'stlpips_pytorch/weights/vST0.0/alex_shift_tolerant.pth'
SHAFE_WEIGHT = OLD / 'work/shafe_pilot/resnetaa50d.d_in12k.features.safetensors'


def crop(image):
    return np.ascontiguousarray(np.clip(np.rint(image[ROI]), 0, 255).astype('uint8'))


def gray(image):
    return np.asarray(Image.fromarray(image).convert('L'))


def blur(image):
    return cv2.sepFilter2D(image, -1, KERNEL, KERNEL)


def pixel_scores(sr, hr):
    a, b = gray(sr).astype(np.float64), gray(hr).astype(np.float64)
    sse = np.square(a - b).sum()
    return psnr(sse, a.size) if sse else 'inf', ssim(a, b, blur)


class Scorer:
    def __init__(self):
        import torch
        sys.path[:0] = [str(OLD / 'work/shafe_formal'), str(OLD / 'work/shafe_formal/deps'),
                        str(ST_RUNTIME)]
        from run_formal import paper_scores
        from run_shafe import load_metric
        from stlpips_pytorch.stlpips import LPIPS

        setup_cuda(torch)
        self.torch = torch
        self.paper_scores = paper_scores
        self.shafe = load_metric(SHAFE_WEIGHT, 2).to('cuda')
        self.shafe.device = 'cuda'
        self.shafe.eth_feature.device = 'cuda'
        self.shafe.eth_feature.model.to('cuda')
        self.st = LPIPS(net='alex', variant='shift_tolerant', model_path=str(ST_WEIGHT),
                        verbose=False).eval().to('cuda')

    def evaluate(self, pairs):
        torch = self.torch

        def rgb(images):
            return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to('cuda').float() / 255

        results = []
        with torch.inference_mode():
            for start in range(0, len(pairs), BATCH):
                batch = pairs[start:start + BATCH]
                sr, hr = zip(*batch)
                shafe = self.paper_scores(self.shafe, [(gray(a), gray(b)) for a, b in batch],
                                          'cuda', BATCH)
                st = self.st(rgb(sr), rgb(hr), normalize=True).flatten().tolist()
                if not np.isfinite(st).all():
                    raise ValueError('ST-LPIPS produced a non-finite score')
                for (a, b), sh, lp in zip(batch, shafe, st, strict=True):
                    results.append(dict(zip(METRICS, (*pixel_scores(a, b), sh, lp), strict=True)))
        return results


def apply(image, vector):
    return shift(image, vector) if any(vector) else image


def score_scene(row, sr_paths, scorer):
    hr, lr = read(row['hr']), read(row['lr'])
    if hr.shape != (512, 512, 3) or lr.shape != (128, 128, 3):
        raise ValueError(f"Wrong reference or input dimensions for {row['scene']}")
    hr_added, sr_added = row['hr_added'], row['sr_added']
    cases, pairs = [], {}
    zero = [0, 0]
    truths = dict(zip(CONDITIONS, ((zero, zero), (hr_added, zero), (zero, sr_added),
                                  (hr_added, sr_added)), strict=True))
    anchor = gray8(luminance(lr))
    shifted_hr = apply(hr, hr_added)
    lr_fits = {}
    for hvec, reference in ((zero, hr), (hr_added, shifted_hr)):
        whole = candidate(reference, lr)
        lk, n_tracks = lucas_kanade(anchor, gray8(project(luminance(reference))))
        lr_fits[tuple(hvec)] = (whole, lk, n_tracks)
    for model, path in sr_paths.items():
        normal_sr = read(path)
        shifted_sr = apply(normal_sr, sr_added)
        for condition, (hvec, svec) in truths.items():
            h = shifted_hr if any(hvec) else hr
            s = shifted_sr if any(svec) else normal_sr
            whole, lk, n_tracks = lr_fits[tuple(hvec)]
            actions = dict(fixed=zero, direct=direct_candidate(h, s),
                           global_lr=whole['vector'] if whole['executable'] else zero,
                           lk_lr=lk, oracle=[-v for v in hvec], clean_hr=None)
            cropped_sr = crop(s)
            sr_hash = pixel_sha(cropped_sr)
            pair_ids = {}
            for method, vector in actions.items():
                intervention = crop(hr if method == 'clean_hr' else apply(h, vector))
                key = sr_hash + ':' + pixel_sha(intervention)
                pairs.setdefault(key, (cropped_sr, intervention))
                pair_ids[method] = key
            cases.append(dict(model=model, condition=condition, hr_added=hvec, sr_added=svec,
                              actions=actions, global_valid=whole['valid'], lk_tracks=n_tracks,
                              hr_rgb_pixel_sha256=pixel_sha(h), sr_rgb_pixel_sha256=pixel_sha(s),
                              score_pair_ids=pair_ids))
    scores = dict(zip(pairs, scorer.evaluate(list(pairs.values())), strict=True))
    for case in cases:
        case.update(scores={m: scores[k] for m, k in case['score_pair_ids'].items()})
    return dict(scene=row['scene'], split=row['split'], magnitude=row['magnitude'],
                lr_pixel_sha256=pixel_sha(lr), n_unique_pairs=len(pairs), cases=cases)


def run(cohort):
    base = ROOT / cohort
    if cohort == 'formal':
        _, config = verify_lock()
        records = load(ROOT / 'inputs/manifest.json')['records']
    else:
        records = dev_records()
        config = f'dev:{file_sha(Path(__file__).with_name("ranking_models.py"))}'
    for model in MODELS:
        if list(pending_outputs(records, base, model, config)):
            raise ValueError(f'{model} inference is incomplete')
    cv2.setNumThreads(1)
    scorer = None
    start = time.monotonic()
    for i, row in enumerate(records, 1):
        for field in ('hr', 'lr'):
            if file_sha(row[field]) != row[field + '_sha256']:
                raise ValueError(f'{field} changed: {row["scene"]}')
        paths = {model: artifact(base, model, row['scene']) for model in MODELS}
        checks = dict(config=config, scorer_sha256=file_sha(__file__),
                      hr_sha256=row['hr_sha256'], lr_sha256=row['lr_sha256'],
                      sr_sha256={model: file_sha(path) for model, path in paths.items()})
        out = base / 'scores' / (row['scene'] + '.json')
        if not out.exists() or load(out)['checks'] != checks:
            if scorer is None:
                scorer = Scorer()
            atomic_json(out, dict(checks=checks, **score_scene(row, paths, scorer)))
        if i % 12 == 0 or i == len(records):
            print(f'Scored or checked {i}/{len(records)} ({time.monotonic()-start:.1f} s)', flush=True)
    if scorer is not None:
        print(f'GPU peak {scorer.torch.cuda.max_memory_allocated()/2**20:.0f} MiB', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('dev', 'dev_repeat', 'formal'))
    run(parser.parse_args().cohort)
