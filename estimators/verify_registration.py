"""Numerical and geometric regression checks before the reference search is frozen."""
import itertools
import time

import cv2
import numpy as np

from dev_pipeline import HERE, luminance, project, read, register, shift
from experiment import dump, load
from registration import Projection, candidate, direct_candidate


def verify():
    cv2.setNumThreads(2)
    rng = np.random.default_rng(731)
    render = cv2.GaussianBlur(rng.normal(127,35,(560,560)).astype('float32'),(5,5),.7)
    base = render[24:536,24:536]
    random_lr = project(render)[6:134,6:134]
    records = load(HERE/'dev/inputs_manifest.json')['records']
    cases = [(f'random_{v}',shift(render,v)[24:536,24:536],random_lr)
             for v in [(0,0),(.5,0),(0,-2),(6,0),(-2,6)]]
    cases.extend((f"{r['scene']}__{r['degradation']}",read(r['hr']),read(r['lr'])) for r in records)
    detailed = []
    for name,hr,lr in cases:
        t = time.perf_counter()
        ref = register(hr,lr)
        elapsed_ref = time.perf_counter()-t
        t = time.perf_counter()
        fast = candidate(hr,lr)
        elapsed_fast = time.perf_counter()-t
        assert ref['valid'] and fast['valid']
        for field in ('vector','executable','independent_vector','tile_vectors','hypotheses','boundary'):
            assert ref[field] == fast[field], (name,field,ref,fast)
        fields = ('residual','zero_improvement','independent_gap','tile_agreement')
        max_delta = max(abs(ref[f]-fast[f]) for f in fields)
        assert max_delta < 1e-6, (name,max_delta)
        detailed.append(dict(case=name,vector=fast['vector'],reference_s=elapsed_ref,
                             fast_s=elapsed_fast,max_feature_difference=max_delta))
        print(detailed[-1],flush=True)
    for v in [(0,0),(.5,0),(0,-2),(12,0),(-8,10)]:
        vector=direct_candidate(shift(base,v),base)
        assert np.max(np.abs(np.array(vector)+v))<=.25,(v,vector)
    offsets = (-8,-6.75,-.5,0,.25,3.5,8)
    projection_errors = {}
    for name,hr,_ in (cases[0],cases[6],cases[-1]):
        y = luminance(hr)
        fast = Projection(y)
        diff = max(float(np.max(np.abs(project(shift(y,v))[12:116,12:116]-fast(v))))
                   for v in itertools.product(offsets,repeat=2))
        assert diff < 1.5e-4,(name,diff)
        projection_errors[name] = diff
    dump(HERE/'dev/projection_check.json',dict(cases=detailed,max_projection_pixel_delta=projection_errors,
         direct_displacement_tests_passed=5,projection_tests_per_input=len(offsets)**2))


if __name__ == '__main__':
    verify()
