"""Score the frozen ARKitScenes temporal reference and its synchronous control."""

from hashlib import sha256
from pathlib import Path
import sys

import cv2
import numpy as np

from prepare_arkitscenes_temporal import INSET, LR_CONTEXT, OUT, OUTPUT_SCALE, WINDOW, digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shafe_optical'))
from score_pairs import (Scorer, apply, artifact, atomic_json, candidate, direct_candidate,
                         downproject, gray8, infer, load, luminance, lucas_kanade, MODELS, read)
from dev_pipeline import pixel_sha
from ranking_common import LOCK, verify_lock


def verified_path(entry):
    path = OUT / entry['path']
    if digest(path) != entry['sha256']:
        raise ValueError(f'Scoring input has changed: {path}')
    return path


def score_visit(visit, sr_paths, scorer):
    path = {label: verified_path(entry) for label, entry in visit['files'].items()}
    lr = read(path['lr'])
    core_x, core_y, end_x, end_y = visit['score_xyxy']
    native_x, native_y = visit['candidate']['origin_lr']
    context_x, context_y = visit['context_origin_lr']
    if (core_x, core_y) != ((native_x - context_x) * OUTPUT_SCALE + INSET,
                            (native_y - context_y) * OUTPUT_SCALE + INSET):
        raise ValueError('Scoring region changed relative to the LR sensor')
    side = WINDOW * OUTPUT_SCALE - 2 * INSET
    extent = LR_CONTEXT * OUTPUT_SCALE
    if (end_x - core_x != side or end_y - core_y != side or
            not INSET <= core_x < end_x <= extent - INSET or
            not INSET <= core_y < end_y <= extent - INSET):
        raise ValueError('Scoring region must be contained within the audited HR window')

    def region(image):
        if image.shape != (extent, extent, 3):
            raise ValueError('Wrong HR or SR image shape')
        return np.ascontiguousarray(np.clip(np.rint(image[core_y:end_y, core_x:end_x]), 0, 255).astype('uint8'))

    def aligned(image, action):
        if max(map(abs, action)) + 2 >= INSET:
            raise ValueError('Geometric correction would sample outside the verified window')
        return region(apply(image, action))

    restored = {m: read(p) for m, p in sr_paths.items()}
    cores = {m: region(im) for m, im in restored.items()}
    hr = {name: read(path[name]) for name in ('sync', 'moved')}
    references = {label: read(p) for label, p in path.items() if label.startswith('depth_')}
    anchor = gray8(luminance(lr))
    actions, fit_quality, requests = {}, {}, {}
    for condition, image in hr.items():
        whole = candidate(image, lr)
        lk, tracks = lucas_kanade(anchor, gray8(downproject(luminance(image))))
        methods = dict(fixed=[0., 0.], global_lr=whole['vector'] if whole['executable'] else [0., 0.], lk_lr=lk)
        common = {name: aligned(image, v) for name, v in methods.items()}
        fit_quality[condition] = dict(global_lr=whole, lk_tracks=tracks)
        actions[condition] = {}
        for model, sr in restored.items():
            direct = direct_candidate(image, sr)
            actions[condition][model] = dict(methods, direct=direct)
            for name, ref in dict(common, direct=aligned(image, direct)).items():
                requests[(condition, model, name)] = (cores[model], ref)
    for label, reference in references.items():
        if reference.shape != (side, side, 3):
            raise ValueError('Depth reference shape differs from the common scoring support')
        for model in MODELS:
            requests[('moved', model, label)] = (cores[model], reference)

    unique = {}
    keys = {}
    for row, (sr, ref) in requests.items():
        key = (pixel_sha(sr), pixel_sha(ref))
        unique.setdefault(key, (sr, ref))
        keys[row] = key
    scored = dict(zip(unique, scorer.evaluate(list(unique.values())), strict=True))
    conditions = {c: {m: {} for m in MODELS} for c in ('sync', 'moved')}
    for (c, m, method), key in keys.items():
        conditions[c][m][method] = scored[key]
    capture_delta = scorer.evaluate([(region(hr['sync']), region(hr['moved']))] +
                                    [(region(hr['sync']), im) for im in references.values()])
    return dict(video=visit['video'], visit=visit['visit'], score_xyxy=visit['score_xyxy'],
                lr_sha256=visit['files']['lr']['sha256'], actions=actions, fit_quality=fit_quality,
                sr_pixel_sha256={m: pixel_sha(im) for m, im in cores.items()},
                scores=conditions, capture_delta=dict(zip(['fixed', *references], capture_delta, strict=True)))


def run():
    cv2.setNumThreads(1)
    protocol = OUT / 'protocol.json'
    input_manifest = OUT / 'inputs.json'
    protocol_sha, input_sha = (digest(p) for p in (protocol, input_manifest))
    audit, plan = load(input_manifest), load(protocol)
    if audit['protocol_sha256'] != protocol_sha or plan['hashes'][Path(__file__).name] != digest(Path(__file__)):
        raise ValueError('Input manifest or scoring implementation changed')
    _, config = verify_lock()
    selected = [row['selected'] for row in audit['visits'] if row['selected']]
    if not selected:
        raise ValueError('No visits passed the locked temporal substitution checks')
    inputs = [dict(row, scene=row['files']['lr']['sha256'], lr=str(verified_path(row['files']['lr'])),
                   lr_sha256=row['files']['lr']['sha256']) for row in selected]
    infer(inputs, OUT)
    scorer = Scorer()
    all_rows = []
    for row in inputs:
        paths = {model: artifact(OUT, model, row['scene']) for model in MODELS}
        sr_hashes = {}
        for model, path in paths.items():
            sr_hashes[model] = digest(path)
            if load(path.with_suffix('.json')) != dict(config=config, lr_sha256=row['lr_sha256'],
                                                      image_sha256=sr_hashes[model]):
                raise ValueError(f'SR metadata disagrees with input or image: {path}')
        checks = dict(input_manifest_sha256=input_sha, protocol_sha256=protocol_sha,
                      ranking_lock_sha256=digest(LOCK), sr_sha256=sr_hashes)
        result = OUT / 'cases' / (row['video'] + '.json')
        if result.exists() and load(result)['checks'] == checks:
            case = load(result)
        else:
            case = dict(checks=checks, **score_visit(row, paths, scorer))
            atomic_json(result, case)
        all_rows.append(case)
        print('Scored', row['video'], flush=True)
    atomic_json(OUT / 'scores.json', dict(inputs_sha256=input_sha, protocol_sha256=protocol_sha, visits=all_rows))


if __name__ == '__main__':
    run()
