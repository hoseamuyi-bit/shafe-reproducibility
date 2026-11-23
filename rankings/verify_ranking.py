"""Check development repeatability and independently audit the formal result."""
import argparse
from collections import Counter
import itertools
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

from ranking_common import CONDITIONS, LOCK, METRICS, METHODS, MODELS, ROOT, atomic_json, file_sha, load, verify_lock
from dev_pipeline import pixel_sha, read
from ranking_models import artifact, dev_records
from ranking_scores import Scorer


def independent_crop(source, dx=0, dy=0):
    if dx == round(dx) and dy == round(dy):
        x, y = int(dx), int(dy)
        return np.clip(np.rint(source[32-y:480-y, 32-x:480-x]), 0, 255).astype(np.uint8)
    y, x = np.mgrid[32:480, 32:480]
    sampled = cv2.remap(source.astype(np.float32), (x-dx).astype(np.float32),
                        (y-dy).astype(np.float32), cv2.INTER_CUBIC)
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def move(source, dx, dy):
    y, x = np.mgrid[:512, :512]
    return cv2.remap(source.astype(np.float32), (x-dx).astype(np.float32),
                     (y-dy).astype(np.float32), cv2.INTER_CUBIC)


def recompute_pixel_metrics(first, second):
    x, y = (np.asarray(Image.fromarray(img).convert('L'), dtype=float) for img in (first, second))
    mse = np.mean((x - y)**2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse else float('inf')

    def smooth(img):
        return gaussian_filter(img, 1.5, radius=5)[5:-5, 5:-5]

    a, b = smooth(x), smooth(y)
    c, d = 6.5025, 58.5225
    ssim = ((2*a*b+c) * (2 * (smooth(x*y) - a*b) + d)
            / ((a*a+b*b+c) * (smooth(x*x)+smooth(y*y)-a*a-b*b+d))).mean()
    return np.array([psnr, ssim])


def verify_dev():
    values = []
    for row in dev_records():
        scene = row['scene']
        first, repeat = (load(ROOT / version / 'scores' / f'{scene}.json')
                         for version in ('dev', 'dev_repeat'))
        assert first == repeat, f'Development score cache mismatch: {scene}'
        for model in MODELS:
            a = artifact(ROOT / 'dev', model, scene)
            b = artifact(ROOT / 'dev_repeat', model, scene)
            assert np.array_equal(read(a), read(b)), f'Nonidentical repeated {model} RGB output'
        values.append(first)
    probe = read(dev_records()[0]['hr'])[32:480, 32:480].copy()
    reversed_probe = probe[:, ::-1].copy()
    self, different = Scorer().evaluate([(probe, probe), (probe, reversed_probe)])
    assert self['PSNR'] == 'inf' and self['SSIM'] == 1
    assert abs(self['SHAFE']) < 1e-6 and self['ST-LPIPS'] == 0
    assert different['ST-LPIPS'] > self['ST-LPIPS'] and different['SHAFE'] > self['SHAFE']
    for obj in values:
        for case in obj['cases']:
            if obj['magnitude'] != .5 or not any(case['hr_added']):
                assert case['score_pair_ids']['oracle'] == case['score_pair_ids']['clean_hr']
    summary = dict(scenes=len(values), repeated_rgb_outputs=len(values) * len(MODELS),
                   repeated_score_rows=sum(len(o['cases']) for o in values),
                   same_image=self, reflected_image=different)
    atomic_json(ROOT / 'development_verification.json', summary)
    print(summary, flush=True)


def verify_formal():
    lock, digest = verify_lock()
    records = load(ROOT / 'inputs/manifest.json')['records']
    summary = load(ROOT / 'formal/analysis.json')
    checked, differences, flip_counts = [], [], Counter()
    max_error = np.zeros(2)
    max_learned_error = np.zeros(2)
    cross_checked = 0
    learned_checked = 0
    scorer = None
    tolerance = lock['ranking']['epsilon']['SHAFE']
    for row in records:
        scene = row['scene']
        obj = load(ROOT / 'formal/scores' / f'{scene}.json')
        assert obj['checks']['config'] == digest and obj['lr_pixel_sha256'] == row['lr_pixel_sha256']
        assert file_sha(row['source']) == row['sha256']
        for key in ('hr', 'lr'):
            assert file_sha(row[key]) == obj['checks'][f'{key}_sha256'] == row[f'{key}_sha256']
        for model in MODELS:
            assert file_sha(artifact(ROOT / 'formal', model, scene)) == obj['checks']['sr_sha256'][model]
        by_condition = {name: [case for case in obj['cases'] if case['condition'] == name]
                        for name in CONDITIONS}
        for name, cases in by_condition.items():
            assert {r['model'] for r in cases} == set(MODELS) and len(cases) == len(MODELS)
            assert len({tuple(r['actions']['lk_lr']) for r in cases}) == 1
            assert len({tuple(r['actions']['global_lr']) for r in cases}) == 1
            for r in cases:
                assert len({s.split(':')[0] for s in r['score_pair_ids'].values()}) == 1
                assert r['hr_added'] == (row['hr_added'] if name in ('hr_global', 'both_global') else [0, 0])
                assert r['sr_added'] == (row['sr_added'] if name in ('sr_global', 'both_global') else [0, 0])
                if row['magnitude'] != .5 or not any(r['hr_added']):
                    assert r['score_pair_ids']['oracle'] == r['score_pair_ids']['clean_hr']
        local = Counter()
        for a, b in itertools.combinations(by_condition['sr_global'], 2):
            ref = a['scores']['oracle']['SHAFE'] - b['scores']['oracle']['SHAFE']
            for method in ('direct', 'lk_lr'):
                observed = a['scores'][method]['SHAFE'] - b['scores'][method]['SHAFE']
                local[method] += abs(ref) > tolerance and abs(observed) > tolerance and ref * observed < 0
        flip_counts.update(local)
        differences.append((row['split'], row['magnitude'], (local['direct'] - local['lk_lr']) / 6))
        if scene not in lock['audit_scenes']:
            continue
        original_hr = read(row['hr'])
        learned_pairs = {}
        for case in obj['cases']:
            native_sr = read(artifact(ROOT / 'formal', case['model'], scene))
            sr = independent_crop(native_sr, *case['sr_added'])
            h = move(original_hr, *case['hr_added'])
            for method in METHODS:
                hr = (independent_crop(original_hr) if method == 'clean_hr'
                      else independent_crop(h, *case['actions'][method]))
                assert case['score_pair_ids'][method] == f'{pixel_sha(sr)}:{pixel_sha(hr)}'
                ref = np.array([case['scores'][method][key] for key in METRICS[:2]])
                error = np.abs(recompute_pixel_metrics(sr, hr) - ref)
                max_error = np.maximum(error, max_error)
                assert np.all(error < 1e-9)
                cross_checked += 1
                if case['condition'] == 'sr_global' and method in ('direct', 'lk_lr', 'oracle'):
                    learned_pairs.setdefault(case['score_pair_ids'][method], (sr, hr, case['scores'][method]))
        if scorer is None:
            scorer = Scorer()
        rescored = scorer.evaluate([(sr, hr) for sr, hr, _ in learned_pairs.values()])
        for fresh, (_, _, stored) in zip(rescored, learned_pairs.values(), strict=True):
            error = np.abs([fresh[key] - stored[key] for key in METRICS[2:]])
            max_learned_error = np.maximum(error, max_learned_error)
            assert np.all(error < 2e-6)
            learned_checked += 1
        checked.append(scene)
    for method, count in flip_counts.items():
        assert count == summary['results']['sr_global']['SHAFE']['methods'][method]['flips']
    observed = np.mean([d for _, _, d in differences]) * 100
    assert abs(observed - summary['primary']['estimate']) < 1e-12
    rng = np.random.default_rng(lock['ranking']['seed'] + 1)
    replicas = []
    for split, amplitude in sorted({(s, a) for s, a, _ in differences}):
        group = [d for s, a, d in differences if (s, a) == (split, amplitude)]
        replicas.append(rng.choice(group, size=(10000, len(group)), replace=True))
    interval = np.quantile(np.concatenate(replicas, axis=1).mean(axis=1) * 100, [.025, .975])
    assert all(abs(interval - summary['primary']['ci95']) < .5)
    result = dict(scene_files=len(records), hashes_checked=cross_checked, audited_scenes=checked,
                  max_abs_pixel_metric_error=dict(zip(METRICS[:2], max_error.tolist())),
                  rescored_learned_pairs=learned_checked,
                  max_abs_learned_metric_error=dict(zip(METRICS[2:], max_learned_error.tolist())),
                  independent_primary_pp=observed, independent_ci95_pp=interval.tolist(),
                  independent_flip_counts=dict(flip_counts))
    atomic_json(ROOT / 'formal/verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('dev', 'formal'))
    (verify_dev if parser.parse_args().cohort == 'dev' else verify_formal)()
