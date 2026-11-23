"""Lock the confirmatory evaluation before computing any formal outcomes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from dev_pipeline import CONFIG, HERE, OLD, sha
from experiment import VOSR, VOSR_ARGS, load, plan

LOCK = HERE / 'formal_lock.json'
MANIFEST = HERE / 'data/manifest_frozen.json'
GATES = HERE / 'calibration/gates.json'
PROTOCOL = OLD / 'outputs/SHAFE低清锚定与双侧归因_下一轮实验方案.md'
LOCAL = (
    'freeze.py','dev_pipeline.py','prepare_data.py','scene_audit.py','experiment.py',
    'registration.py','measure.py','gate.py','analysis.py','score.py','environment_verified.json',
    'data/manifest_frozen.json','data/scene_overlap_candidates.json',
    'data/div_source.json','data/urban_source.json','dev/geometry.json','dev/batch_inputs.json',
    'calibration/geometry.json','calibration/batch_inputs.json','calibration/gates.json',
    'calibration/evaluation.json',
)
ARCHIVED = (
    'work/shafe_formal/run_formal.py','work/shafe_pilot/run_shafe.py',
    'work/shafe_pilot/SHAFE.py','work/shafe_pilot/weight_ranges.json',
)


def design():
    manifest = load(MANIFEST)
    if manifest['status'] != 'FROZEN' or load(HERE/'data/scene_overlap_candidates.json')['pairs']:
        raise ValueError('Scene split or overlap audit incomplete')
    splits = {kind:[r for r in manifest['scenes'] if r['split'] == kind]
              for kind in ('dev','calibration','formal')}
    if {kind:len(rows) for kind,rows in splits.items()} != {'dev':20,'calibration':30,'formal':150}:
        raise ValueError('Unexpected scene counts')
    if len({row['group'] for row in manifest['scenes']}) != 200:
        raise ValueError('Scene group collision')
    return list(plan(splits['formal'],'formal'))


def hashes():
    return {
        **{f'local/{name}':sha(HERE/name) for name in LOCAL},
        **{f'archive/{name}':sha(OLD/name) for name in ARCHIVED},
        'protocol':sha(PROTOCOL),
        'vosr_runner':sha(VOSR),
    }


def forecast():
    rows = [r for r in load(HERE/'calibration/geometry.json')['rows'] if r['degradation']=='strong']
    cells = {}
    for row in rows:
        key = row['dataset'],row['group']
        vals = cells.setdefault(key,np.zeros(3,dtype=int))
        vals += [row['candidate_correct'],row['simple_false_correction'],row['needs_correction']]
    rng = np.random.default_rng(20261124)
    projections = np.zeros((5000,3),dtype=int)
    for dataset in sorted({key[0] for key in cells}):
        values = np.array([v for (ds,_),v in cells.items() if ds==dataset])
        projections += values[rng.integers(0,len(values),(5000,75))].sum(axis=1)
    correct,false,move = projections.T
    return dict(purpose='Calibration empirical projection for 75 + 75 new scenes; conditional on observed scene distribution.',
                candidate_ceiling_percentiles=np.quantile(correct/move,[.025,.5,.975]).tolist(),
                ungated_false_rate_percentiles=np.quantile(false/(1200-move),[.025,.5,.975]).tolist(),
                fraction_candidate_below_80=float(np.mean(correct/move < .8)),
                false_events_at_frozen_threshold={'single':0,'dual':0},
                exact_upper_95_scene_event_rate_if_zero_in_30=1-.05**(1/30))


def lock():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    if any((HERE/'formal').iterdir()):
        raise ValueError('Formal files already exist; cannot label this run pre-outcome')
    plans = design()
    provenance = load(HERE/'environment_verified.json')
    gate = load(GATES)
    file_hashes = hashes()
    for expected, actual in (
        (gate['gate_code_sha256'],file_hashes['local/gate.py']),
        (gate['development_geometry_sha256'],file_hashes['local/dev/geometry.json']),
        (gate['calibration_geometry_sha256'],file_hashes['local/calibration/geometry.json']),
        (provenance['code_and_records'][str(VOSR)],file_hashes['vosr_runner']),
    ):
        if expected != actual:
            raise ValueError('Training or inference provenance has drifted')
    frozen = dict(status='LOCKED_BEFORE_FORMAL_OUTCOMES',locked_utc=datetime.now(timezone.utc).isoformat(),
                  formal_cases=2400,primary_cases=1200,files_sha256=file_hashes,
                  config={**CONFIG,'status':'FROZEN_FOR_FORMAL'},injection_plan=plans,
                  gate=gate,vosr_args=VOSR_ARGS,weights=dict(vosr=provenance['vosr_weights'],
                  shafe=provenance['shafe_weight']),calibration_projection=forecast(),
                  analysis=dict(strong_target=.8,execution_hold_max_hr_px=.25,
                                correction_tolerance_hr_px=.25,practical_difference=.05,
                                bootstrap_n=5000,bootstrap_seed=20261123,
                                bootstrap_unreachable_descriptive_threshold=.05),
                  scoring=dict(radius=50,temperature=.01,feature_indices=[1,2],
                               crop=[32,32,480,480],device='cuda',batch_size=2))
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=HERE,delete=False) as stream:
        json.dump(frozen,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
        temporary = stream.name
    try:
        os.link(temporary,LOCK)
    finally:
        Path(temporary).unlink()
    print(json.dumps(dict(lock_sha256=sha(LOCK),precision=forecast()),ensure_ascii=False),flush=True)


def verify():
    frozen = load(LOCK)
    if frozen['status'] != 'LOCKED_BEFORE_FORMAL_OUTCOMES':
        raise ValueError('Invalid formal lock')
    observed = hashes()
    expected = frozen['files_sha256']
    if observed != expected:
        changed = [key for key in sorted(set(observed)|set(expected)) if observed.get(key)!=expected.get(key)]
        raise ValueError(f'Formal lock mismatch: {changed}')
    if frozen['injection_plan'] != design() or frozen['gate'] != load(GATES):
        raise ValueError('Frozen design or gate drift')
    if frozen['vosr_args'] != VOSR_ARGS:
        raise ValueError('Frozen inference command drift')
    return frozen


def verify_inputs(data,frozen):
    if data['design_sha256'] != frozen['files_sha256']['local/data/manifest_frozen.json']:
        raise ValueError('Formal input design hash drift')
    records = data['records']
    expected = [(p,kind) for p in frozen['injection_plan'] for kind in ('normal','strong')]
    if len(records) != len(expected):
        raise ValueError('Formal input count drift')
    for row,(plan,kind) in zip(records,expected):
        if row['degradation']!=kind or any(row[key]!=value for key,value in plan.items()):
            raise ValueError('Formal input plan drift')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('lock','verify'))
    args = parser.parse_args()
    if args.action == 'lock':
        lock()
    else:
        verify()
        print(sha(LOCK))
