"""Run the frozen scoring function on a scene range alongside the serial runner."""
import argparse
import os

import cv2

from ranking_common import MODELS, ROOT, atomic_json, file_sha, load, verify_lock
from ranking_models import artifact, pending_outputs
from ranking_scores import Scorer, score_scene
import ranking_scores


def run(first, last):
    _, config = verify_lock()
    manifest = load(ROOT / 'inputs/manifest.json')['records']
    if not 1 <= first <= last <= len(manifest):
        raise ValueError('Expected a valid inclusive one-based scene range')
    records = manifest[first - 1:last]
    base = ROOT / 'formal'
    for model in MODELS:
        if list(pending_outputs(records, base, model, config)):
            raise ValueError(f'{model} inference is incomplete in the selected range')

    cv2.setNumThreads(1)
    scorer = Scorer()
    stage = base / 'shards' / f'{first:03d}-{last:03d}'
    stage.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(records, first):
        for field in ('hr', 'lr'):
            if file_sha(row[field]) != row[field + '_sha256']:
                raise ValueError(f'Input changed: {field} {row["scene"]}')
        images = {model: artifact(base, model, row['scene']) for model in MODELS}
        checks = dict(config=config, scorer_sha256=file_sha(ranking_scores.__file__),
                      hr_sha256=row['hr_sha256'], lr_sha256=row['lr_sha256'],
                      sr_sha256={model: file_sha(path) for model, path in images.items()})
        output = base / 'scores' / f'{row["scene"]}.json'
        if output.exists():
            if load(output)['checks'] != checks:
                raise ValueError(f'Existing score has different provenance: {output}')
            continue
        staging_path = stage / output.name
        fresh = dict(checks=checks, **score_scene(row, images, scorer))
        atomic_json(staging_path, fresh)
        try:
            os.link(staging_path, output)
        except FileExistsError:
            if load(output) != fresh:
                raise ValueError(f'Concurrent scoring disagreement: {output}')
        if (index - first + 1) % 6 == 0 or index == last:
            print(f'Range {first}-{last} reached scene {index}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    args = parser.parse_args()
    run(args.first, args.last)
