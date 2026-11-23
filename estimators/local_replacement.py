"""Diagnose whole-image corrections caused by a local HR content replacement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dev_pipeline import dump, read, sha
from experiment import load, variants
from freeze import LOCK, verify
from gate import confidence, row_features
from robust_registration import outside_replacement, tiled_candidates


PERTURBATIONS = ('zero', 'hr_global', 'hr_replacement', 'hr2_replacement')


def local_record(record, original):
    cv2.setNumThreads(1)
    lr, hr, sr, source = [read(record[name]) for name in ('lr', 'hr', 'sr', 'source')]
    results = []
    for condition, disturbed, _, vh, _ in variants(record, hr, sr, source):
        if condition not in PERTURBATIONS:
            continue
        old = original[record['scene'], condition]
        fits = dict(global_mse=old['hr_fit'], **tiled_candidates(disturbed, lr))
        if condition.endswith('replacement'):
            fits['mask_diagnostic'] = outside_replacement(disturbed, lr, *record['hr_replace_xy'], 256)
        results.append(dict(scene=record['scene'], dataset=record['dataset'], condition=condition,
                            added=vh.tolist(), fits=fits))
    return results


def vector_error(fit, added):
    return float(np.max(np.abs(np.asarray(fit['vector']) + added)))


def summarize(rows):
    result = []
    for condition in dict.fromkeys(r['condition'] for r in rows):
        cases = [r for r in rows if r['condition'] == condition]
        for method in cases[0]['fits']:
            fits = [r['fits'][method] for r in cases]
            action_count = sum(f['executable'] for f in fits)
            need_move = bool(any(cases[0]['added']))
            errors = [vector_error(f, r['added']) for r, f in zip(cases, fits)]
            correct = sum(f['executable'] and e <= .25 for e, f in zip(errors, fits))
            result.append(dict(condition=condition, method=method, n=len(cases),
                               correct=correct if need_move else len(cases)-action_count,
                               moved=action_count, valid=sum(f['valid'] for f in fits),
                               median_error=float(np.median(errors))))
    return result


def audit_development(split, workers):
    base = ROOT / split
    manifest = base / 'batch_inputs.json'
    geometry = base / 'geometry.json'
    records = [r for r in load(manifest)['records'] if r['degradation']=='strong']
    original = {(r['scene'], r['condition']): r for r in load(geometry)['rows']
                if r['degradation']=='strong' and r['condition'] in PERTURBATIONS}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for group in pool.map(partial(local_record, original=original), records) for r in group]
    summary = summarize(rows)
    dump(HERE / f'{split}_local_replacement.json', dict(manifest_sha256=sha(manifest),
         geometry_sha256=sha(geometry), implementation_sha256=sha(HERE/'robust_registration.py'),
         rows=rows, summary=summary))
    for row in summary:
        print(split, row, flush=True)


def audit_gates():
    verify()
    gates_path = ROOT / 'calibration/gates.json'
    path = ROOT / 'formal/decisions.json'
    gates = load(gates_path)['models']
    threshold = gates['dual']['calibration']['threshold']
    definitions = load(gates_path)
    names = definitions['hr_features'] + definitions['additional_dual_features']
    selected = [r for r in load(path)['rows'] if (r['degradation'], r['condition'])==
                ('strong', 'hr_replacement') and r['actions']['dual']!=[0, 0]]
    records = {(r['scene'], r['degradation']): r for r in load(ROOT/'formal/batch_inputs.json')['records']}
    result = []
    for row in selected:
        record = records[row['scene'], 'strong']
        original = {(row['scene'], row['condition']): row}
        lr, hr, sr, source = [read(record[name]) for name in ('lr', 'hr', 'sr', 'source')]
        disturbed = next(h for c, h, _, _, _ in variants(record, hr, sr, source) if c==row['condition'])
        model = gates['dual']
        features = np.array(row_features(row, True))
        filled = np.where(np.isfinite(features), features, model['median'])
        terms = (filled - model['mean']) / model['scale'] * model['coef']
        hr_count = len(definitions['hr_features'])
        without_sr = float(model['intercept'] + terms[:hr_count].sum())
        added_sr = float(terms[hr_count:].sum())
        reproduced = float(1 / (1 + np.exp(-without_sr-added_sr)))
        if not np.isclose(reproduced, row['confidence']['dual'], rtol=0, atol=1e-10):
            raise AssertionError(f'Dual score reconstruction drift: {row["scene"]}')
        result.append(dict(scene=row['scene'], global_vector=row['hr_fit']['vector'],
                           tile_agreement=row['hr_fit']['tile_agreement'],
                           sr_vector=row['sr_fit']['vector'], single_confidence=confidence(row, gates['single'], 'single'),
                           dual_confidence=reproduced, dual_threshold=threshold,
                           confidence_at_mean_sr_features=float(1 / (1 + np.exp(-without_sr))),
                           sr_feature_logit_contribution=added_sr, feature_logit_terms=dict(zip(names,terms.tolist())),
                           mask_diagnostic=outside_replacement(disturbed, lr, *record['hr_replace_xy'], 256)))
    dump(HERE/'gate_failure_diagnosis.json', dict(formal_lock_sha256=sha(LOCK), decisions_sha256=sha(path),
         gates_sha256=sha(gates_path), diagnostic='Post hoc. Withholding SR features sets standardized values to zero.',
         cases=result))
    for row in result:
        print({k: v for k, v in row.items() if k != 'feature_logit_terms'}, flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('steps', nargs='+', choices=('dev', 'calibration', 'gates'))
    cli.add_argument('--workers', type=int, default=4)
    args = cli.parse_args()
    for step in args.steps:
        if step=='gates':
            audit_gates()
        else:
            audit_development(step, args.workers)
