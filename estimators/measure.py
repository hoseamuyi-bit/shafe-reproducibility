"""Collect per-case geometry without learning from the evaluation split."""
import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np

from dev_pipeline import HERE, dump, pixel_sha, read, sha
from experiment import CONDITIONS, load, variants
from registration import candidate, direct_candidate


def measure_record(record, reference, include_direct):
    for key in ('lr','hr','source'):
        if sha(record[key]) != record[key+'_sha256']:
            raise ValueError(f'{key} hash changed: {record[key]}')
    sr_path = Path(record['sr'])
    if sha(sr_path) != load(sr_path.with_suffix('.json'))['sr_sha256']:
        raise ValueError(f'SR hash changed: {record["sr"]}')
    lr, hr, sr = [read(record[k]) for k in ('lr','hr','sr')]
    assert lr.shape == (128,128,3) and hr.shape == sr.shape == (512,512,3)
    assert pixel_sha(lr) == record['lr_pixel_sha256']
    external = record['degradation'] == 'camera'
    full = None if external else read(record['source'])
    rows = []
    fits = {}
    def fit(image, pixel_hash):
        if pixel_hash not in fits:
            fits[pixel_hash] = candidate(image,lr,reference=reference)
        return fits[pixel_hash]
    for name, disturbed_hr, disturbed_sr, vh, vs in variants(record,hr,sr,full):
        hhash, shash = pixel_sha(disturbed_hr), pixel_sha(disturbed_sr)
        hf = fit(disturbed_hr,hhash)
        sf = fit(disturbed_sr,shash)
        needs = bool(np.any(vh))
        error = float(np.max(np.abs(np.asarray(hf['vector'])+vh)))
        correct = hf['executable'] and needs and error <= .25
        direct = direct_candidate(disturbed_hr,disturbed_sr) if include_direct else None
        rows.append(dict(scene=record['scene'], group=record['group'], dataset=record['dataset'],
            degradation=record['degradation'], condition=name, global_magnitude=record['magnitude'],
            hr_added=vh.tolist(), sr_added=vs.tolist(), needs_correction=needs,
            candidate_correct=correct, simple_false_correction=not needs and hf['executable'],
            simple_wrong_correction=needs and hf['executable'] and not correct,
            candidate_remaining_linf=error, hr_fit=hf,sr_fit=sf,direct_vector=direct,
            lr_pixel_sha256=record['lr_pixel_sha256'],hr_pixel_sha256=hhash,sr_pixel_sha256=shash))
    index = {r['condition']:r for r in rows}
    for a,b in (('zero','sr_global'),('hr_global','both_global'),('zero','sr_local'),
                ('hr_global','hr2_sr_local')):
        if b not in index or index[a]['hr_added'] != index[b]['hr_added']:
            continue
        assert index[a]['hr_pixel_sha256'] == index[b]['hr_pixel_sha256']
        assert index[a]['hr_fit'] == index[b]['hr_fit']
    assert tuple(index) == (CONDITIONS[:4] if external else CONDITIONS)
    assert pixel_sha(lr) == record['lr_pixel_sha256']
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['degradation'], 'all')].append(r)
        groups[(r['degradation'], r['dataset'])].append(r)
        groups[(r['degradation'], r['condition'])].append(r)
        if r['condition'] in CONDITIONS[:4]:
            groups[(r['degradation'], f"global {r['global_magnitude']:g} px")].append(r)
    result = []
    for (kind,group), subset in groups.items():
        needs = sum(r['needs_correction'] for r in subset)
        correct = sum(r['candidate_correct'] for r in subset)
        false = sum(r['simple_false_correction'] for r in subset)
        wrong = sum(r['simple_wrong_correction'] for r in subset)
        result.append(dict(degradation=kind,stratum=group,n=len(subset),hr_should_move=needs,
            hr_should_hold=len(subset)-needs,candidate_correct=correct,simple_false=false,
            simple_wrong=wrong,simple_false_scenes=len({r['scene'] for r in subset if r['simple_false_correction']}),
            candidate_upper_bound=correct/needs if needs else None))
    return result


def run(split, reference=False, include_direct=True, selected=None, tag=None):
    cv2.setNumThreads(2)
    base = HERE / split
    manifest = base / 'batch_inputs.json'
    input_data = load(manifest)
    if split in ('formal','external'):
        from freeze import verify, verify_inputs
        frozen = verify()
        if selected or tag or reference or not include_direct:
            raise ValueError('Formal and camera evaluation require the complete frozen run')
        if split=='formal':
            verify_inputs(input_data,frozen)
    inputs = input_data['records']
    if selected:
        inputs = [r for r in inputs if r['scene'] in selected]
    label = tag or ('reference' if reference else 'geometry')
    source = HERE/'registration.py'
    settings = dict(measure_sha256=sha(__file__),registration_sha256=sha(source),
                    sources_sha256=sha(manifest),reference=reference,include_direct=include_direct)
    all_rows = []
    tick = time.perf_counter()
    for i,record in enumerate(inputs,1):
        path = base / label / f"{record['scene']}__{record['degradation']}.json"
        sr_sha = sha(record['sr'])
        cfg = dict(settings,sr_sha256=sr_sha)
        saved = load(path) if path.exists() else {}
        if saved.get('settings') == cfg:
            rows = saved['rows']
        else:
            rows = measure_record(record,reference,include_direct)
            dump(path,dict(settings=cfg,rows=rows))
        all_rows.extend(rows)
        if i % 10 == 0 or i == len(inputs):
            print(f'{split} {label}: {i}/{len(inputs)} scene-degradations',flush=True)
    summary = summarize(all_rows)
    dump(base/f'{label}.json',dict(settings=settings,elapsed_s=time.perf_counter()-tick,
         rows=all_rows,summary=summary))
    for row in summary:
        if row['stratum'] == 'all':
            print(row,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('split',choices=('dev','calibration','formal','external'))
    p.add_argument('--reference',action='store_true')
    p.add_argument('--without-direct',action='store_true')
    p.add_argument('--scene',action='append')
    p.add_argument('--tag')
    a = p.parse_args()
    run(a.split,a.reference,not a.without_direct,a.scene,a.tag)
