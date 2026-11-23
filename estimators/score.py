"""Restartable SHAFE scoring for frozen actions with fixed SR inputs."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from dev_pipeline import HERE, OLD, dump, pixel_sha, read, sha, shift
from experiment import CONDITIONS, load, variants
from gate import METHODS, actions

BATCH_SIZE = 2
DEVICE = 'cuda'
CROP = (32,32,480,480)


def luminance_crop(image):
    image = Image.fromarray(np.clip(np.rint(image),0,255).astype('uint8'))
    return np.asarray(image.convert('L').crop(CROP))


def score_record(record, rows, gates, evaluate):
    for key in ('lr','hr','source'):
        if sha(record[key]) != record[key+'_sha256']:
            raise ValueError(f'{key} input drift: {record[key]}')
    sr_path = Path(record['sr'])
    if sha(sr_path) != load(sr_path.with_suffix('.json'))['sr_sha256']:
        raise ValueError(f'SR input drift: {sr_path}')
    hr,sr = (read(record[k]) for k in ('hr','sr'))
    camera = record['degradation']=='camera'
    full = None if camera else read(record['source'])
    if tuple(rows) != (CONDITIONS[:4] if camera else CONDITIONS):
        raise ValueError('Unexpected geometry case set or order')

    pairs = {}
    results = []
    for name,disturbed_hr,disturbed_sr,vh,vs in variants(record,hr,sr,full):
        geometry = rows[name]
        if (pixel_sha(disturbed_hr),pixel_sha(disturbed_sr)) != (
                geometry['hr_pixel_sha256'],geometry['sr_pixel_sha256']):
            raise ValueError(f'Scoring variant drift: {record["scene"]} {name}')
        if (vh.tolist(),vs.tolist()) != (geometry['hr_added'],geometry['sr_added']):
            raise ValueError(f'Scoring injection drift: {record["scene"]} {name}')
        fixed_sr = luminance_crop(disturbed_sr)
        sr_hash = pixel_sha(fixed_sr)
        action_vectors,_,_,_,_ = actions(geometry,gates)
        pair_ids = {}
        for method in METHODS + ('clean_hr',):
            if method == 'clean_hr':
                if 'replacement' in name:
                    continue
                reference = hr
            else:
                vector = action_vectors[method]
                if vector is None:
                    raise ValueError('Missing direct registration control')
                reference = shift(disturbed_hr,vector) if any(vector) else disturbed_hr
            reference = luminance_crop(reference)
            key = sr_hash + ':' + pixel_sha(reference)
            pairs.setdefault(key,(fixed_sr,reference))
            pair_ids[method] = key
        row = {k:geometry[k] for k in ('scene','group','dataset','degradation','condition',
                                       'hr_added','sr_added','hr_pixel_sha256','sr_pixel_sha256')}
        results.append(dict(row,actions=action_vectors,score_pair_ids=pair_ids))
    values = dict(zip(pairs,evaluate(list(pairs.values())),strict=True))
    for row in results:
        row['scores'] = {m:values[key] for m,key in row['score_pair_ids'].items()}
    return results,len(values)


def run(split,batch_size=BATCH_SIZE,selected=None):
    if split in ('formal','external'):
        from freeze import verify, verify_inputs
        frozen = verify()
        if selected or batch_size!=frozen['scoring']['batch_size']:
            raise ValueError('Frozen scoring requires every case and the locked batch size')
    base = HERE/split
    sys.path.insert(0,str(OLD/'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import CONFIGURATIONS, DEFAULT_WEIGHT, load_metric, sha256

    geom_path = base/'geometry.json'
    input_path = base/'batch_inputs.json'
    gates_path = HERE/'calibration/gates.json'
    inputs = load(input_path)
    geom = load(geom_path)
    if geom['settings']['sources_sha256'] != sha(input_path):
        raise ValueError('Input manifest differs from the one measured')
    if split=='formal':
        verify_inputs(inputs,frozen)
    geometry = {}
    for row in geom['rows']:
        key = (row['scene'],row['degradation'])
        case = geometry.setdefault(key,{})
        if row['condition'] in case:
            raise ValueError(f'Duplicate geometry: {key}')
        case[row['condition']] = row
    records = [r for r in inputs['records'] if not selected or r['scene'] in selected]
    if not records or (not selected and len(records)!=len(geometry)):
        raise ValueError('Incomplete geometry set')
    gates = load(gates_path)
    weight_digest = sha256(DEFAULT_WEIGHT)
    if split in ('formal','external') and weight_digest != frozen['weights']['shafe']['sha256']:
        raise ValueError('SHAFE weight differs from the formal lock')
    config = dict(score_code_sha256=sha(__file__),geometry_sha256=sha(geom_path),
        gates_sha256=sha(gates_path),weight_sha256=weight_digest,device=DEVICE,batch_size=batch_size,
        grayscale='clip and round to 8-bit, then Pillow RGB to L',crop=list(CROP),
        **CONFIGURATIONS['paper_shafe'])
    metric = None

    def evaluate(pairs):
        nonlocal metric
        if metric is None:
            metric = load_metric(DEFAULT_WEIGHT,4).to(DEVICE)
            metric.device = DEVICE
            metric.eth_feature.device = DEVICE
            metric.eth_feature.model.to(DEVICE)
        return paper_scores(metric,pairs,DEVICE,batch_size)

    label = 'score_smoke' if selected else 'scores'
    all_rows = []
    unique = 0
    start = time.perf_counter()
    for i,record in enumerate(records,1):
        key = (record['scene'],record['degradation'])
        sr_digest = sha(record['sr'])
        settings = dict(config,sr_sha256=sr_digest)
        cache = base/'score_cache'/f'{key[0]}__{key[1]}.json'
        cached = load(cache) if cache.exists() else {}
        if cached.get('settings') != settings:
            rows,n = score_record(record,geometry[key],gates,evaluate)
            cached = dict(settings=settings,rows=rows,unique_pairs=n)
            temporary = cache.with_suffix('.tmp')
            dump(temporary,cached)
            temporary.replace(cache)
        all_rows.extend(cached['rows'])
        unique += cached['unique_pairs']
        if i%10==0 or i==len(records):
            print(f'{split} SHAFE: {i}/{len(records)} scene-degradations; {unique} unique pairs',flush=True)
    dump(base/f'{label}.json',dict(split=split,methods=list(METHODS),unique_pairs=unique,rows=all_rows,
                                  scoring=config,elapsed_s=time.perf_counter()-start))


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('split',choices=('formal','calibration','external'))
    cli.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    cli.add_argument('--scene',action='append')
    args = cli.parse_args()
    run(args.split,args.batch_size,args.scene)
