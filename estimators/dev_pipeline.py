"""Isolated development pilot. Never reads calibration/formal image outcomes.

Pixel centres are integer OpenCV centres. Positive vectors move content right/down.
All displacement operations, including zero, use OpenCV INTER_CUBIC (-0.75).
Downsampling uses Pillow float bicubic with its antialias filter. Border pixels are
excluded from matching (48 HR px) and scoring (32 HR px).
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OLD = Path(os.environ.get('SHAFE_PRIOR_ROOT', HERE.parent / 'shafe-prior'))
CONFIG = dict(version='development-v1', seed=20261122, scale=4,
              blur_normal=1.2, blur_strong=2.4, noise_sigma=2/255,
              jpeg_quality=40, jpeg_subsampling=2, noise_seed='sha256(scene|strong|20261122) first 8 bytes little endian',
              displacement_sampling='OpenCV INTER_CUBIC; BORDER_CONSTANT=0; integer pixel centres',
              downsample='Pillow float-mode BICUBIC, antialiased; render560->140 then centre128',
              gaussian='OpenCV GaussianBlur, radius ceil(4*sigma), BORDER_REFLECT_101',
              match_roi_lr=[12,12,116,116], score_roi_hr=[32,32,480,480],
              normalization='independent global ROI mean/std, MSE',
              integer_radius=8, refine_step=0.25, refine_radius=1,
              independent_peak_distance=2, tile_agreement_tolerance=0.5,
              residual_tie_rounding_decimals=10,
              status='DEVELOPMENT_ONLY_NOT_FROZEN')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read(path):
    return np.array(Image.open(path).convert('RGB'))


def save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array),0,255).astype('uint8')).save(path)


def luminance(image):
    if image.ndim == 2:
        return image.astype('float32')
    return image.astype('float32') @ np.array([.299,.587,.114], dtype='float32')


def shift(image, vector):
    matrix = np.array([[1,0,vector[0]], [0,1,vector[1]]], dtype='float32')
    return cv2.warpAffine(image.astype('float32'), matrix, (image.shape[1],image.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def project(image, sigma=1.2):
    radius = int(np.ceil(4*sigma))
    blurred = cv2.GaussianBlur(image.astype('float32'), (2*radius+1,2*radius+1), sigma,
                               borderType=cv2.BORDER_REFLECT_101)
    size = (image.shape[1]//4,image.shape[0]//4)
    if image.ndim == 2:
        return np.array(Image.fromarray(blurred).resize(size, Image.Resampling.BICUBIC))
    return np.stack([np.array(Image.fromarray(blurred[:,:,k]).resize(size, Image.Resampling.BICUBIC))
                     for k in range(3)], axis=-1)


def degrade(render, scene, strong):
    output = project(render, 2.4 if strong else 1.2)[6:134,6:134]
    if strong:
        seed = int.from_bytes(hashlib.sha256((scene+'|strong|20261122').encode()).digest()[:8], 'little')
        output = output + np.random.default_rng(seed).normal(0,2,output.shape)
    output = np.clip(np.rint(output),0,255).astype('uint8')
    if strong:
        stream = io.BytesIO()
        Image.fromarray(output).save(stream, format='JPEG', quality=40, subsampling=2, optimize=False)
        output = np.array(Image.open(io.BytesIO(stream.getvalue())).convert('RGB'))
    return output


def prepare():
    manifest = json.loads((HERE/'data/manifest_provisional.json').read_text(encoding='utf-8'))
    selected = [next(r for r in manifest['scenes'] if r['dataset']==ds and r['split']=='dev')
                for ds in ('DIV2K','Urban100')]
    records = []
    for row in selected:
        full = read(row['source'])
        x,y,x1,y1 = row['render_crop_xyxy']
        render = full[y:y1,x:x1]
        assert render.shape == (560,560,3)
        folder = HERE/'dev/inputs'/row['id']
        save(folder/'render.png', render)
        save(folder/'hr.png', render[24:536,24:536])
        for strong in (False,True):
            kind = 'strong' if strong else 'normal'
            lr = degrade(render,row['id'],strong)
            target = HERE/'dev/lr_smoke'/f"{row['id']}__{kind}.png"
            save(target,lr)
            records.append(dict(scene=row['id'],dataset=row['dataset'],degradation=kind,
                source=row['source'],source_sha256=row['sha256'],render=str(folder/'render.png'),
                hr=str(folder/'hr.png'),lr=str(target),lr_sha256=sha(target),lr_pixel_sha256=pixel_sha(lr),
                sr=str(HERE/'dev/vosr_smoke/images'/target.name)))
    dump(HERE/'dev/inputs_manifest.json', dict(config=CONFIG, provisional_manifest_sha256=sha(HERE/'data/manifest_provisional.json'),
                                            selection='first scene ID in provisional dev split per dataset; no outcome selection',records=records))
    print('Prepared 2 scenes x 2 degradations: 4 original128 LR inputs',flush=True)


def register(moving, lr):
    moving, anchor = luminance(moving), luminance(lr)[12:116,12:116]
    std = float(anchor.std())
    if std < 1e-6 or not np.isfinite(anchor).all():
        return dict(valid=False, reason='constant_or_nonfinite_anchor',vector=[0.,0.],executable=False)
    anchor = (anchor-anchor.mean())/std
    cache = {}
    tiles = []
    def evaluate(vector):
        vector = tuple(float(v) for v in vector)
        if vector in cache:
            return cache[vector]
        predicted = project(shift(moving,vector))[12:116,12:116]
        pred_std = float(predicted.std())
        if pred_std < 1e-6 or not np.isfinite(predicted).all():
            value = float('inf')
            tile = np.full(16,np.inf)
        else:
            residual = ((predicted-predicted.mean())/pred_std-anchor)**2
            value = float(residual.mean())
            tile = residual.reshape(4,26,4,26).mean(axis=(1,3)).ravel()
        cache[vector]=(value,tile)
        return cache[vector]
    def rank(vector):
        return (round(evaluate(vector)[0],10),vector[0]**2+vector[1]**2,vector[0],vector[1])
    integer = [(x,y) for y in range(-8,9) for x in range(-8,9)]
    ordered = sorted(integer,key=rank)
    peaks = []
    for candidate in ordered:
        if all(np.linalg.norm(np.array(candidate)-p)>=2 for p in peaks):
            peaks.append(candidate)
            if len(peaks)==3:
                break
    for x,y in peaks:
        for dx in np.arange(-1,1.001,.25):
            for dy in np.arange(-1,1.001,.25):
                vector = (x+dx,y+dy)
                if max(abs(vector[0]),abs(vector[1])) <= 8:
                    evaluate(vector)
    best = min(cache,key=rank)
    if not np.isfinite(evaluate(best)[0]):
        return dict(valid=False,reason='no_finite_projection',vector=[0.,0.],executable=False)
    independent = [v for v in cache if np.linalg.norm(np.array(v)-best)>=2]
    second = min(independent,key=rank)
    vectors = list(cache)
    tile_values = np.stack([cache[v][1] for v in vectors])
    tile_vectors = np.array(vectors)[np.argmin(tile_values,axis=0)]
    agreement = float(np.mean(np.max(np.abs(tile_vectors-np.array(best)),axis=1)<=.5))
    error = evaluate(best)[0]
    return dict(valid=True,reason=None,vector=list(best),executable=max(abs(v) for v in best)>.25,
                residual=error,zero_improvement=evaluate((0,0))[0]-error,
                independent_gap=evaluate(second)[0]-error,independent_vector=list(second),
                magnitude=float(np.linalg.norm(best)),boundary=max(abs(v) for v in best)==8,
                tile_agreement=agreement,tile_vectors=tile_vectors.tolist(),hypotheses=len(cache))


def geometry():
    cv2.setNumThreads(4)
    rng=np.random.default_rng(731)
    grid=rng.normal(127,35,(560,560)).astype('float32')
    grid=cv2.GaussianBlur(grid,(5,5),.7)
    lr=project(grid)[6:134,6:134]
    rows=[]
    for vector in [(0,0),(.5,0),(0,-2),(6,0),(-2,6)]:
        hr=shift(grid,vector)[24:536,24:536]
        fit=register(hr,lr)
        residual=np.array(fit['vector'])+vector
        assert np.max(np.abs(residual))<=.25,(vector,fit)
        rows.append(dict(injected=vector,candidate=fit['vector'],remaining=residual.tolist()))
    impulse=np.zeros((64,64),dtype='float32');impulse[32,32]=1
    assert np.unravel_index(shift(impulse,(2,-3)).argmax(),impulse.shape)==(29,34)
    assert np.array_equal(shift(grid,(0,0)),grid)
    # Scaling and half-pixel convention: downsampling an affine ramp maps LR
    # centre j to HR centre 4*j+1.5, away from finite-image boundaries.
    ramp=np.tile(np.arange(560,dtype='float32'),(560,1))
    reduced=project(ramp)[6:134,6:134]
    assert abs(float(reduced[64,64])-(24+4*64+1.5))<1e-4
    dump(HERE/'dev/geometry_checks.json',dict(config=CONFIG,passed=True,cases=rows,
           impulse='right2/up3 -> x34/y29',identity='bitwise equal',scale='LR centre j -> HR centre 4*j+1.5'))
    print(json.dumps(rows),flush=True)


def local_sr(sr, x,y,vector):
    mask=np.zeros((512,512),dtype='float32')
    weights=np.ones(128,dtype='float32')
    weights[:12]=(1-np.cos(np.pi*np.arange(12)/12))/2
    weights[-12:]=weights[:12][::-1]
    mask[y:y+128,x:x+128]=weights[:,None]*weights[None,:]
    shifted=shift(sr,vector)
    return sr*(1-mask[:,:,None])+shifted*mask[:,:,None],mask


def variants(record):
    hr=read(record['hr']).astype('float32');sr=read(record['sr']).astype('float32')
    full=read(record['source'])
    seed=int.from_bytes(hashlib.sha256((record['scene']+'|inject-v1').encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    directions=[(1,0),(-1,0),(0,1),(0,-1)]
    dh=np.array(directions[rng.integers(4)],dtype=float)
    ds=np.array(directions[rng.integers(4)],dtype=float)
    # Pilot covers half-pixel and large shifts; full balanced assignment remains
    # pending split freeze and must replace this explicitly pilot-only schedule.
    magnitude=.5 if record['dataset']=='DIV2K' else 6
    vh,vs=dh*magnitude,ds*magnitude
    x,y=[int(v) for v in rng.integers(64,321,2)]
    local,mask=local_sr(sr,x,y,ds*4)
    folder=HERE/'dev/masks'/record['scene'];save(folder/'sr_local.png',mask*255)
    # A far corner from the same source, selected by source coordinates rather
    # than by a registration outcome. Record overlap with the target support.
    target_x,target_y=[int(v)*256 for v in rng.integers(0,2,2)]
    h,w=full.shape[:2]
    source_x=0 if target_x else w-256
    source_y=0 if target_y else h-256
    replaced=hr.copy();replaced[target_y:target_y+256,target_x:target_x+256]=full[source_y:source_y+256,source_x:source_x+256]
    mask_hr=np.zeros((512,512));mask_hr[target_y:target_y+256,target_x:target_x+256]=255
    save(folder/'hr_replacement.png',mask_hr)
    specs=[('zero',hr,sr,(0,0),(0,0)),('hr_global',hr,sr,vh,(0,0)),
           ('sr_global',hr,sr,(0,0),vs),('both_global',hr,sr,vh,vs),
           ('sr_local',hr,local,(0,0),(0,0)),('hr2_sr_local',hr,local,dh*2,(0,0)),
           ('hr_replacement',replaced,sr,(0,0),(0,0)),('hr2_replacement',replaced,sr,dh*2,(0,0))]
    return [(name,shift(h,a),shift(s,b),np.array(a),dict(sr_added=list(b),
             pilot_magnitude=magnitude,sr_local_xy=[x,y],hr_replace_xy=[target_x,target_y],
             replacement_source_xy=[source_x,source_y])) for name,h,s,a,b in specs]


def run_pilot():
    cv2.setNumThreads(4)
    manifest=json.loads((HERE/'dev/inputs_manifest.json').read_text(encoding='utf-8'))
    rows=[];cache={};tick=time.perf_counter()
    for record in manifest['records']:
        assert sha(record['lr'])==record['lr_sha256']
        lr=read(record['lr']);anchor_hash=pixel_sha(lr)
        assert read(record['sr']).shape==(512,512,3)
        for name,hr,sr,added,detail in variants(record):
            fits=[]
            for moving in (hr,sr):
                key=(pixel_sha(moving),anchor_hash)
                if key not in cache:
                    cache[key]=register(moving,lr)
                fits.append(cache[key])
            hf,sf=fits;vector=np.array(hf['vector'])
            needs=bool(np.max(np.abs(added))>0)
            correct=bool(hf['executable'] and needs and np.max(np.abs(vector+added))<=.25)
            rows.append(dict(scene=record['scene'],dataset=record['dataset'],degradation=record['degradation'],
                condition=name,hr_added=added.tolist(),details=detail,hr_fit=hf,sr_fit=sf,
                needs_correction=needs,candidate_correct=correct,
                simple_false_correction=bool(not needs and hf['executable']),
                lr_pixel_sha256=anchor_hash,sr_pixel_sha256=pixel_sha(sr)))
            assert pixel_sha(lr)==anchor_hash
        print(f"{record['scene']} {record['degradation']}: 8 cases registered",flush=True)
    summary={}
    invariance=[]
    for record in manifest['records']:
        current={r['condition']:r for r in rows if r['scene']==record['scene'] and r['degradation']==record['degradation']}
        for first,second in [('zero','sr_global'),('hr_global','both_global')]:
            assert current[first]['hr_fit']==current[second]['hr_fit']
            invariance.append(dict(scene=record['scene'],degradation=record['degradation'],
                                   pair=[first,second],hr_candidate_and_features_identical=True))
    for degradation in ('normal','strong'):
        subset=[r for r in rows if r['degradation']==degradation]
        positives=[r for r in subset if r['needs_correction']]
        negatives=[r for r in subset if not r['needs_correction']]
        summary[degradation]=dict(cases=len(subset),hr_should_move=len(positives),hr_should_hold=len(negatives),
                candidate_correct=sum(r['candidate_correct'] for r in positives),
                simple_false_corrections=sum(r['simple_false_correction'] for r in negatives))
    dump(HERE/'dev/pilot_candidates.json',dict(status='DEVELOPMENT_ONLY; no gates fitted',
        config=CONFIG,elapsed_s=time.perf_counter()-tick,unique_registrations=len(cache),
        counterfactual_checks=invariance,summary=summary,rows=rows))
    print(json.dumps(summary),flush=True)


def score_pilot():
    import torch
    sys.path.insert(0,str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import load_metric, DEFAULT_WEIGHT
    manifest=json.loads((HERE/'dev/inputs_manifest.json').read_text(encoding='utf-8'))
    candidates=json.loads((HERE/'dev/pilot_candidates.json').read_text(encoding='utf-8'))
    indexed={(r['scene'],r['degradation'],r['condition']):r for r in candidates['rows']}
    pending=[];hashes={};rows=[];invariants=[]
    def grayscale(image):
        # Pin the archived Pillow 8-bit luminance preprocessing after resampling.
        return np.array(Image.fromarray(np.clip(np.rint(image),0,255).astype('uint8')).convert('L'))
    def score_id(hr,sr):
        pair=(grayscale(sr)[32:480,32:480],grayscale(hr)[32:480,32:480])
        key=hashlib.sha256(pair[0].tobytes()+pair[1].tobytes()).hexdigest()
        if key not in hashes:
            hashes[key]=len(pending);pending.append(pair)
        return hashes[key]
    for record in manifest['records']:
        assert sha(record['lr'])==record['lr_sha256']
        clean_hr=read(record['hr'])
        for name,hr,sr,added,detail in variants(record):
            key=(record['scene'],record['degradation'],name)
            candidate=indexed[key]
            assert pixel_sha(sr)==candidate['sr_pixel_sha256']
            sr_hash=pixel_sha(sr)
            vector=candidate['hr_fit']['vector'] if candidate['hr_fit']['executable'] else [0,0]
            ids=dict(raw=score_id(shift(hr,(0,0)),sr),
                simple_lr=score_id(shift(hr,vector),sr),oracle=score_id(shift(hr,-added),sr))
            assert pixel_sha(sr)==sr_hash
            # For pure global HR injections, compare undo-resampled oracle HR
            # with clean HR against exactly the same actual SR.
            if name in ('zero','hr_global','sr_global','both_global','sr_local','hr2_sr_local'):
                ids['clean_hr']=score_id(shift(clean_hr,(0,0)),sr)
            rows.append(dict(scene=record['scene'],degradation=record['degradation'],condition=name,
                score_ids=ids,sr_unchanged=True,lr_unchanged=True))
        invariants.append(dict(scene=record['scene'],degradation=record['degradation'],
                              lr_sha256=sha(record['lr']),sr_sha256=sha(record['sr']),sr_shape=list(read(record['sr']).shape)))
    metric=load_metric(DEFAULT_WEIGHT,4)
    cpu=paper_scores(metric,pending[:2],'cpu',1)
    # A fresh object prevents the official lowpass cache from retaining CPU tensors.
    del metric
    device='cuda'
    metric=load_metric(DEFAULT_WEIGHT,4).to(device)
    metric.device=device;metric.eth_feature.device=device;metric.eth_feature.model.to(device)
    gpu_check=paper_scores(metric,pending[:2],device,1)
    diffs=np.abs(np.array(cpu)-gpu_check)
    assert diffs.max()<1e-4,dict(cpu=cpu,gpu=gpu_check)
    values=paper_scores(metric,pending,device,2)
    for row in rows:
        row['scores']={name:values[i] for name,i in row.pop('score_ids').items()}
    summary={}
    for kind in ('normal','strong'):
        subset=[r for r in rows if r['degradation']==kind]
        summary[kind]={name+'_mae_vs_oracle':float(np.mean([abs(r['scores'][name]-r['scores']['oracle']) for r in subset]))
                       for name in ('raw','simple_lr')}
        subset=[r for r in subset if 'clean_hr' in r['scores']]
        summary[kind]['oracle_vs_clean_resampling_mae']=float(np.mean([abs(r['scores']['oracle']-r['scores']['clean_hr']) for r in subset]))
    dump(HERE/'dev/pilot_scores.json',dict(status='DEVELOPMENT_ONLY; raw/simple_lr/oracle only; no gates or direct baseline',
        scoring=dict(radius=50,temperature=.01,features=[1,2],crop448=True,
                     luminance='Pillow RGB->8-bit L after resampling',weight_sha256=sha(DEFAULT_WEIGHT)),
        cpu_gpu_check=dict(cpu=cpu,gpu=gpu_check,max_abs_difference=float(diffs.max()),tolerance=1e-4),
        unique_pairs=len(pending),invariants=invariants,summary=summary,rows=rows))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','geometry','pilot','score'])
    action=parser.parse_args().action
    {'prepare':prepare,'geometry':geometry,'pilot':run_pilot,'score':score_pilot}[action]()
