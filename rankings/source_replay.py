"""Rebuild the frozen COCO inputs and replay SR on the preselected audit scenes."""
import argparse
from collections import Counter

import numpy as np
from PIL import Image

from ranking_common import MODELS, ROOT, atomic_json, file_sha, load, verify_lock
from dev_pipeline import degrade, pixel_sha, read, save
from ranking_models import artifact, run_vosr, stamp, torch_outputs


REPLAY = ROOT / 'source_replay'


def rebuild_inputs():
    lock, digest = verify_lock()
    records = load(ROOT / 'inputs/manifest.json')['records']
    audit = set(lock['audit_scenes'])
    rows = []
    checked = Counter()
    for row in records:
        scene = row['scene']
        if file_sha(row['source']) != row['sha256']:
            raise ValueError(f'Source JPEG hash mismatch: {scene}')
        full = read(row['source'])
        origin = ((full.shape[1] - 560) // 2, (full.shape[0] - 560) // 2)
        if list(origin) != row['render_origin']:
            raise ValueError(f'Render crop origin mismatch: {scene}')
        x, y = origin
        render = full[y:y + 560, x:x + 560]
        hr = render[24:536, 24:536]
        lr = degrade(render, scene, strong=True)
        for key, image in (('hr', hr), ('lr', lr)):
            if pixel_sha(image) != row[key + '_pixel_sha256']:
                raise ValueError(f'Rebuilt {key.upper()} pixels differ: {scene}')
            checked[row['split']] += 1
        if scene in audit:
            path = REPLAY / 'inputs' / f'{scene}.png'
            save(path, lr)
            if file_sha(path) != row['lr_sha256']:
                raise ValueError(f'Rebuilt LR PNG differs: {scene}')
            rows.append(dict(row, lr=str(path)))
    if {row['scene'] for row in rows} != audit:
        raise ValueError('Missing audited scenes in the frozen manifest')
    result = dict(lock_sha256=digest, source_jpeg_hashes_checked=len(records),
                  rebuilt_input_pixel_hashes=dict(checked), audited_scenes=sorted(audit))
    return digest, rows, result


def replay_outputs(digest, rows):
    matches = {}
    for model in MODELS:
        paths = [(row, artifact(REPLAY, model, row['scene'])) for row in rows]
        if model == 'VOSR':
            run_vosr(paths, REPLAY, digest)
        elif model == 'Bicubic':
            for row, path in paths:
                with Image.open(row['lr']) as image:
                    enlarged = image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
                    save(path, np.asarray(enlarged))
                stamp(path, row, digest)
        else:
            torch_outputs(model, paths, digest)

        matches[model] = {}
        for row, path in paths:
            scene = row['scene']
            formal = artifact(ROOT / 'formal', model, scene)
            checks = load(formal.with_suffix('.json'))
            if checks != dict(config=digest, lr_sha256=row['lr_sha256'], image_sha256=file_sha(formal)):
                raise ValueError(f'Formal SR provenance mismatch: {model}, {scene}')
            replay_hash = pixel_sha(read(path))
            if replay_hash != pixel_sha(read(formal)):
                raise ValueError(f'Replayed SR pixels differ: {model}, {scene}')
            matches[model][scene] = replay_hash
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    digest, rows, result = rebuild_inputs()
    if args.inputs_only:
        atomic_json(REPLAY / 'inputs_verification.json', result)
        print(result, flush=True)
        return
    result['replayed_sr_pixel_sha256'] = replay_outputs(digest, rows)
    atomic_json(REPLAY / 'verification.json', result)
    print(f'Rebuilt {result["source_jpeg_hashes_checked"]} input groups and matched '
          f'{len(rows) * len(MODELS)} re-inferred SR images pixel for pixel.', flush=True)


if __name__ == '__main__':
    main()
