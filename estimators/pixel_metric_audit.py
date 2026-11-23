"""Compare PSNR, SSIM and SHAFE on identically hashed scoring pixels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

from verify_confirmation import check_summary, index


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
PLAN = AUDIT / 'pixel_metric_plan.json'
OUTPUT = AUDIT / 'pixel_metric_results.json'
SPLITS = {'COCO': (AUDIT / 'coco_confirm', 144, 'pair_ids'),
          'RealSR': (ROOT / 'external', 32, 'score_pair_ids')}
CONDITIONS = ('zero', 'sr_global')
METHODS = ('direct', 'oracle')
METRICS = ('PSNR', 'SSIM', 'SHAFE')
EXPECTED = {'COCO': {0.5: 48, 2: 48, 6: 48}, 'RealSR': {0.5: 12, 2: 10, 6: 10}}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def tracked():
    return [Path(__file__), *[path / name for path, _, _ in SPLITS.values()
                             for name in ('batch_inputs.json', 'scores.json')]]


def freeze():
    if PLAN.exists() or OUTPUT.exists():
        raise ValueError('An analysis plan or result already exists')
    previous = PLAN.with_stem(PLAN.stem + '_attempt1')
    save(PLAN, dict(
        defined_utc=datetime.now(timezone.utc).isoformat(),
        status='Post hoc on established cohorts and archived alignment actions',
        prior_attempt=dict(sha256=digest(previous), correction='The first run stopped at the RealSR manifest '
                           'coverage assertion before writing a result. Corrected the expected RealSR magnitude '
                           'counts from the archived manifest; no endpoint or estimator changed.'),
        sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in tracked()},
        populations={'COCO': '144 strong x4 scenes; 48 at each of 0.5, 2 and 6 HR px',
                     'RealSR': '32 prealigned x4 scenes; each camera has six 0.5, five 2 and five 6 HR px cases'},
        scoring='Same hashed 448x448 Pillow L8 pairs used for SHAFE, direct and known-inverse HR actions',
        psnr='10 log10(255^2 / MSE); exact matches have infinite PSNR and would stop this finite summary',
        ssim='Single-scale, 11x11 normalized Gaussian window, sigma 1.5, K1=.01, K2=.03, L=255, '
             'weighted population moments, mean over valid window centers',
        primary='SR-only: direct minus oracle for PSNR and SSIM, oracle minus direct for SHAFE; '
                'positive indicates a more favorable score after direct alignment',
        secondary='For each method, favorable zero score minus favorable SR-only score; '
                  'also report oracle loss minus direct loss and fraction with positive primary difference',
        inference='Descriptive percentile 95% intervals from 5000 paired scene resamples (seed 20261125), '
                  'stratified by camera in RealSR; separate overall and magnitude strata in COCO',
    ))
    print(f'Analysis plan: {digest(PLAN)}', flush=True)


def archives():
    plan = load(PLAN)
    assert plan['sha256'] == {path.relative_to(ROOT).as_posix(): digest(path) for path in tracked()}
    for cohort, (folder, size, ids) in SPLITS.items():
        records = load(folder / 'batch_inputs.json')['records']
        archived = index(load(folder / 'scores.json')['rows'])
        assert len(records) == len({row['scene'] for row in records}) == size
        assert len({(row['dataset'], row['group']) for row in records}) == size
        assert Counter(row['magnitude'] for row in records) == EXPECTED[cohort]
        if cohort == 'RealSR':
            assert Counter((r['dataset'], r['magnitude']) for r in records) == {
                (camera, magnitude): count // 2 for camera in ('Canon', 'Nikon')
                for magnitude, count in EXPECTED[cohort].items()}
        for record in records:
            scene = record['scene']
            for condition in CONDITIONS:
                row = archived[scene, condition]
                assert row['group'] == record['group'] and row['dataset'] == record['dataset']
                assert row['hr_added'] == [0, 0] and row['actions']['oracle'] == [0, 0]
                assert max(abs(x) for x in row['sr_added']) == (record['magnitude'] if condition == 'sr_global' else 0)
        yield cohort, records, archived, ids


def psnr(squared_error, count):
    if not squared_error:
        raise ValueError('Encountered an exact match with infinite PSNR')
    return float(10 * np.log10(255**2 * count / squared_error))


def ssim(x, y, blur):
    ux, uy = blur(x), blur(y)
    vx, vy = blur(x * x) - ux**2, blur(y * y) - uy**2
    cov = blur(x * y) - ux * uy
    c1, c2 = (.01 * 255)**2, (.03 * 255)**2
    luminance = (2 * ux * uy + c1) / (ux**2 + uy**2 + c1)
    structure = (2 * cov + c2) / (vx + vy + c2)
    return float((luminance * structure)[5:-5, 5:-5].mean())


def pixels(alternate):
    sys.path.insert(0, str(ROOT))
    import cv2
    from scipy.ndimage import gaussian_filter

    from dev_pipeline import pixel_sha, shift
    from holdout_run import images_for, variants_for
    from score import luminance_crop

    cv2.setNumThreads(1)
    if alternate:
        kernel = cv2.getGaussianKernel(11, 1.5, cv2.CV_64F)

        def blur(image):
            return cv2.sepFilter2D(image, -1, kernel, kernel)
    else:
        def blur(image):
            return gaussian_filter(image, sigma=1.5, radius=5)

    rows = []
    for cohort, records, archived, ids in archives():
        for i, record in enumerate(records, 1):
            _, hr, sr, full = images_for(record)
            scene = record['scene']
            cache = {}
            for condition, h, s, vh, vs, _ in variants_for(record, hr, sr, full):
                if condition not in CONDITIONS:
                    continue
                original = archived[scene, condition]
                assert vh.tolist() == original['hr_added'] and vs.tolist() == original['sr_added']
                fixed = luminance_crop(s)
                fixed_sha = pixel_sha(fixed)
                scores = {}
                for method in METHODS:
                    vector = original['actions'][method]
                    reference = luminance_crop(shift(h, vector) if any(vector) else h)
                    key = f'{fixed_sha}:{pixel_sha(reference)}'
                    assert key == original[ids][method], (cohort, scene, condition, method)
                    assert fixed.shape == reference.shape == (448, 448)
                    if key not in cache:
                        x, y = fixed.astype(np.float64), reference.astype(np.float64)
                        squared_error = int(np.square(x - y).sum())
                        cache[key] = dict(squared_error=squared_error, PSNR=psnr(squared_error, x.size),
                                          SSIM=ssim(x, y, blur))
                    scores[method] = dict(cache[key], pair_id=key, SHAFE=original['scores'][method])
                rows.append(dict(cohort=cohort, condition=condition, magnitude=record['magnitude'], scores=scores,
                                 **{k: record[k] for k in ('scene', 'dataset', 'group')}))
            if i % 24 == 0 or i == len(records):
                print(f'{cohort}: {i}/{len(records)} scenes', flush=True)
    return rows


def units(rows):
    for cohort, records, archived, ids in archives():
        cases = index([row for row in rows if row['cohort'] == cohort])
        assert set(cases) == set(product([record['scene'] for record in records], CONDITIONS))
        for record in records:
            scene = record['scene']
            for condition in CONDITIONS:
                case = cases[scene, condition]
                original = archived[scene, condition]
                assert all(case[k] == record[k] for k in ('dataset', 'group', 'magnitude'))
                assert set(case['scores']) == set(METHODS)
                for method, score in case['scores'].items():
                    assert score['pair_id'] == original[ids][method]
                    assert score['SHAFE'] == original['scores'][method]
                    assert score['PSNR'] == psnr(score['squared_error'], 448**2)
                    assert -1 <= score['SSIM'] <= 1
        selected = [cases[record['scene'], 'sr_global'] for record in records]
        assert all(cases[row['scene'], 'zero']['scores']['oracle']['pair_id'].split(':')[1]
                   == row['scores']['oracle']['pair_id'].split(':')[1] for row in selected)
        strata = {'all': selected}
        if cohort == 'COCO':
            strata.update({f'{value:g}px': [r for r in selected if r['magnitude'] == value]
                           for value in EXPECTED[cohort]})
        for label, subset in strata.items():
            for metric in METRICS:
                direction = -1 if metric == 'SHAFE' else 1

                def gain(row, method):
                    return direction * row['scores'][method][metric]

                def loss(row, method):
                    return gain(cases[row['scene'], 'zero'], method) - gain(row, method)

                counts = dict(positive=sum(gain(row, 'direct') > gain(row, 'oracle') for row in subset),
                              zero=sum(gain(row, 'direct') == gain(row, 'oracle') for row in subset))
                yield cohort, label, metric, subset, gain, loss, counts


def run():
    from holdout_analysis import paired_summary

    if OUTPUT.exists():
        raise ValueError('Archived results already exist. Use replay to repeat the pixel calculation.')
    rows = pixels(alternate=False)
    results = {}
    for cohort, label, metric, subset, gain, loss, counts in units(rows):
        results.setdefault(cohort, {}).setdefault(label, {})[metric] = dict(
            favorable_score=paired_summary(subset, METHODS, gain, [('direct', 'oracle')]),
            loss_from_zero=paired_summary(subset, METHODS, loss, [('oracle', 'direct')]), **counts)
    save(OUTPUT, dict(plan_sha256=digest(PLAN), rows=rows, results=results))


def verify():
    output = load(OUTPUT)
    assert output['plan_sha256'] == digest(PLAN)
    assert {r['cohort'] for r in output['rows']} == set(SPLITS)
    for cohort, label, metric, subset, gain, loss, counts in units(output['rows']):
        result = output['results'][cohort][label][metric]
        assert all(result[k] == v for k, v in counts.items())
        check_summary(subset, result['favorable_score'], gain)
        check_summary(subset, result['loss_from_zero'], loss)
    print(f"{len(output['rows'])} cases and all paired estimates verified", flush=True)
    return output


def replay():
    original = verify()
    observed = pixels(alternate=True)
    errors = []
    for a, b in zip(observed, original['rows'], strict=True):
        for method in METHODS:
            left, right = a['scores'][method].copy(), b['scores'][method].copy()
            errors.append(abs(left.pop('SSIM') - right.pop('SSIM')))
            assert left == right
        assert all(a[k] == b[k] for k in a if k != 'scores')
    maximum = max(errors)
    assert maximum < 1e-12
    save(AUDIT / 'pixel_metric_replay.json', dict(result_sha256=digest(OUTPUT), cases=len(observed),
                                                method_pairs=len(errors), max_ssim_difference=maximum,
                                                implementation='OpenCV separable Gaussian vs scipy.ndimage'))
    print(f'Independent pixel replay: {len(errors)} method pairs, max SSIM difference {maximum:.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('freeze', 'run', 'verify', 'replay'))
    globals()[parser.parse_args().step]()
