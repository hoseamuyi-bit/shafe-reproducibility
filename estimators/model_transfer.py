"""Post-primary Real-ESRGAN transfer check. Reuse the original splits and gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

from analysis import attach, paired_analysis, relative_vectors, scoring_stats, strata
from dev_pipeline import HERE, OLD, dump, read, save, sha
from experiment import load
from freeze import LOCK, verify, verify_inputs
from measure import measure_record
from score import score_record

BASE = HERE / 'paper/transfer'
WEIGHT = BASE / 'RealESRGAN_x4plus.pth'
WEIGHT_SHA256 = '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
UPSTREAM = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'


def make_plan():
    frozen = verify()
    formal = HERE / 'formal/batch_inputs.json'
    camera = HERE / 'external/batch_inputs.json'
    verify_inputs(load(formal), frozen)
    settings = dict(
        scope='Exploratory model transfer designed after the VOSR geometry outcomes but before Real-ESRGAN inference.',
        model='RealESRGAN_x4plus', architecture='BasicSR RRDBNet x4',
        source=UPSTREAM, weights_sha256=WEIGHT_SHA256, weights_key='params_ema',
        inference='RGB FP32, no padding or tiling, TF32 disabled, clip and round to 8-bit',
        formal='All 150 held-out strong-degradation scenes and their eight existing interventions',
        camera='The same 32 RealSR scenes and four interventions, with response relative to zero injection',
        outcomes='Matched 80% false-correction difference, fixed actions, SR-only response and SHAFE error',
        gates='Reuse VOSR weights and thresholds without any fitting or calibration on Real-ESRGAN',
        primary_lock_sha256=sha(LOCK), formal_manifest_sha256=sha(formal),
        camera_manifest_sha256=sha(camera), runner_sha256=sha(__file__),
        architecture_sha256=sha(HERE/'realesrgan_x4.py'))
    path = BASE/'transfer_plan.json'
    if path.exists():
        assert load(path)['settings']==settings
    else:
        dump(path, dict(locked_utc=datetime.now(timezone.utc).isoformat(), settings=settings))
    records = [r for r in load(formal)['records'] if r['degradation']=='strong'] + load(camera)['records']
    assert len(records)==182 and len({r['scene'] for r in records})==182
    return [dict(r, sr=str(BASE/'sr'/Path(r['sr']).name)) for r in records]


def infer(records):
    import torch
    from realesrgan_x4 import RRDBNetX4

    assert sha(WEIGHT)==WEIGHT_SHA256
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan_sha = sha(BASE/'transfer_plan.json')
    model = None
    manifest = []
    for i, row in enumerate(records, 1):
        path = Path(row['sr'])
        stamp = path.with_suffix('.json')
        assert sha(row['lr'])==row['lr_sha256']
        current = dict(lr_sha256=row['lr_sha256'], plan_sha256=plan_sha)
        if not (path.exists() and stamp.exists() and load(stamp)==dict(current,sr_sha256=sha(path))):
            if model is None:
                model = RRDBNetX4()
                weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)['params_ema']
                model.load_state_dict(weights, strict=True)
                model = model.eval().to(device='cuda', memory_format=torch.channels_last)
            image = torch.from_numpy(read(row['lr'])).permute(2,0,1)[None].to(device='cuda')
            with torch.inference_mode():
                sr = model((image / 255).contiguous(memory_format=torch.channels_last))
                saved = sr[0].permute(1,2,0).clamp(0,1).mul(255).round().to(torch.uint8).cpu().numpy()
            assert saved.shape==(512,512,3)
            save(path,saved)
            dump(stamp,dict(current,sr_sha256=sha(path)))
        manifest.append(dict(row, sr_sha256=load(stamp)['sr_sha256']))
        if i%20==0 or i==len(records):
            print(f'Real-ESRGAN inference: {i}/{len(records)}',flush=True)
    dump(BASE/'batch_inputs.json',dict(torch_version=str(torch.__version__),records=manifest))


def geometry_record(row):
    cv2.setNumThreads(2)
    path = BASE/'geometry'/Path(row['sr']).with_suffix('.json').name
    settings = dict(plan_sha256=sha(BASE/'transfer_plan.json'), sr_sha256=sha(row['sr']),
                    lr_sha256=row['lr_sha256'], hr_sha256=row['hr_sha256'])
    cache = load(path) if path.exists() else {}
    if cache.get('settings') != settings:
        cache = dict(settings=settings,rows=measure_record(row,reference=False,include_direct=True))
        dump(path,cache)
    return cache['rows']


def geometry(records, workers):
    gates = load(HERE/'calibration/gates.json')
    measured = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(geometry_record,records),1):
            measured.extend(result)
            if i%20==0 or i==len(records):
                print(f'Real-ESRGAN geometry: {i}/{len(records)}',flush=True)
    rows = attach(measured,gates)
    assert len(rows)==1328
    key = lambda row:(row['scene'],row['degradation'],row['condition'])
    original = {key(row):row for split in ('formal','external')
                for row in load(HERE/split/'decisions.json')['rows']}
    for row in rows:
        old = original[key(row)]
        assert all(row[name]==old[name] for name in (
            'hr_fit','hr_added','sr_added','hr_pixel_sha256','lr_pixel_sha256','candidate_correct'))
        for method in ('raw','simple_lr','single','oracle'):
            assert row['actions'][method]==old['actions'][method]
        assert row['confidence']['single']==old['confidence']['single']
    synthetic = [row for row in rows if row['degradation']=='strong']
    dump(BASE/'decisions.json',dict(rows=rows))
    dump(BASE/'evaluation.json',dict(primary=paired_analysis(synthetic),
        groups=strata(synthetic),diagnostics=relative_vectors(rows)))


def score(records):
    sys.path.insert(0,str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    assert sha(DEFAULT_WEIGHT)==verify()['weights']['shafe']['sha256']
    metric = load_metric(DEFAULT_WEIGHT,4).to('cuda')
    metric.device = 'cuda'
    metric.eth_feature.device = 'cuda'
    metric.eth_feature.model.to('cuda')
    gates = load(HERE/'calibration/gates.json')
    index = {}
    for row in load(BASE/'decisions.json')['rows']:
        index.setdefault((row['scene'],row['degradation']),{})[row['condition']] = row
    scores = []
    for i, row in enumerate(records,1):
        key = row['scene'],row['degradation']
        path = BASE/'score_cache'/Path(row['sr']).with_suffix('.json').name
        code = sha(BASE/'transfer_plan.json')
        rows = index[key]
        settings = dict(plan_sha256=code, sr_sha256=sha(row['sr']), geometry=rows)
        cache = load(path) if path.exists() else {}
        if cache.get('settings') != settings:
            result,unique = score_record(row,rows,gates,lambda pairs:paper_scores(metric,pairs,'cuda',2))
            cache = dict(settings=settings,rows=result,unique_pairs=unique)
            dump(path,cache)
        scores.extend(cache['rows'])
        if i%20==0 or i==len(records):
            print(f'Real-ESRGAN SHAFE: {i}/{len(records)}',flush=True)
    dump(BASE/'scores.json',dict(rows=scores))
    scoring_stats(scores,'paper/transfer')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('steps',nargs='+',choices=('infer','geometry','score'))
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    records = make_plan()
    for step in args.steps:
        if step=='infer':
            infer(records)
        elif step=='geometry':
            geometry(records,args.workers)
        else:
            score(records)
