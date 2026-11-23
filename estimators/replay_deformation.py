"""Recompute the first scene of each frozen smooth-warp cohort from pixels."""

import sys

import cv2

import camera_deformation as camera
import local_deformation as warp
import registration_benchmarks as benchmarks
from holdout_analysis import linf
from holdout_run import OLD, ROOT, atomic_dump, load, sha
from score import BATCH_SIZE, DEVICE


def replay():
    camera_plan = camera.verify_inputs()
    synthetic_plan = load(warp.LOCK)
    for name, digest in synthetic_plan['hashes'].items():
        assert sha(ROOT / name) == digest, name
    warp.self_test()
    cv2.setNumThreads(1)
    sys.path.insert(0, str(OLD / 'work/shafe_formal'))
    from run_formal import paper_scores
    from run_shafe import DEFAULT_WEIGHT, load_metric

    weight = sha(DEFAULT_WEIGHT)
    assert weight == load(benchmarks.coco_confirm.PLAN)['shafe']['weight_sha256']
    metric = load_metric(DEFAULT_WEIGHT, 4).to(DEVICE)
    metric.device = metric.eth_feature.device = DEVICE
    metric.eth_feature.model.to(DEVICE)

    def evaluate(pairs):
        return paper_scores(metric, pairs, DEVICE, BATCH_SIZE)

    checks = []
    for label, folder, manifest, plan in (
        ('COCO', warp.DEST, warp.INPUTS, synthetic_plan),
        ('RealSR', camera.DEST, camera.INPUTS, camera_plan),
    ):
        record = load(manifest)['records'][0]
        scene = record['scene']
        expected = load(folder / 'scenes' / f'{scene}.json')
        if label == 'RealSR':
            record = dict(record, direction=record['hr_direction'])
        actual, pairs = warp.process(record, plan['parameters'][scene], evaluate)
        assert pairs == expected['unique_pairs']
        difference = 0.0
        for row, prior in zip(actual, expected['rows'], strict=True):
            assert all(value == prior[field] for field, value in row.items() if field != 'scores'), scene
            difference = max(difference, linf([v - prior['scores'][m] for m, v in row['scores'].items()]))
        assert difference <= 1e-6, (scene, difference)
        checks.append(dict(cohort=label, scene=scene, conditions=len(actual), pairs=pairs,
                           lock_sha256=sha(folder / 'lock.json'), max_score_error=difference))
        print(f'{label} {scene}: {len(actual)} conditions, {pairs} pairs, max score error {difference:.3g}')
    atomic_dump(warp.DEST.parent / 'deformation_pixel_replay.json', dict(shafe_weight_sha256=weight, checks=checks))


if __name__ == '__main__':
    replay()
