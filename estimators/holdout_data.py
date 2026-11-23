"""Prepare independent source scenes before registration outcomes are computed."""
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dev_pipeline import degrade, dump, pixel_sha, read, save, sha
from experiment import DIRECTIONS, load


N = 96
SEED = 20261124
PATCH_SIZES = (128, 256, 320)
BASE = HERE / 'holdout'
SOURCE = HERE / 'holdout_source.json'


def overlap_candidates(original, incoming):
    cv2.setNumThreads(2)
    cv2.setRNGSeed(SEED)
    sift = cv2.SIFT_create(500)
    descriptors, points = [], []
    for row in original + incoming:
        with Image.open(row['source']) as image:
            reduced = ImageOps.contain(image.convert('L'), (600, 600))
        kp, desc = sift.detectAndCompute(np.array(reduced), None)
        descriptors.append(desc if desc is not None else np.empty((0, 128), np.float32))
        points.append(np.array([key.pt for key in kp], dtype='float32'))
    owner = np.repeat(np.arange(len(descriptors)), [len(d) for d in descriptors])
    index = cv2.flann_Index(np.vstack(descriptors), dict(algorithm=1, trees=5))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    flagged = []
    for i in range(len(original), len(descriptors)):
        if not len(descriptors[i]):
            continue
        nn, _ = index.knnSearch(descriptors[i], 12, params=dict(checks=64))
        votes = np.bincount(owner[nn.ravel()], minlength=len(descriptors))
        for j in np.flatnonzero(votes[:i] >= 50):
            if len(descriptors[j]) < 2 or len(descriptors[i]) < 2:
                continue
            forward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                       if a.distance < .8 * b.distance}
            backward = {a.queryIdx: a.trainIdx for a, b in matcher.knnMatch(descriptors[j], descriptors[i], k=2)
                        if a.distance < .8 * b.distance}
            pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
            if len(pairs) < 10:
                continue
            a, b = np.array(pairs).T
            _, mask = cv2.findHomography(points[i][a], points[j][b], cv2.RANSAC, 4.)
            if mask is not None and mask.sum() >= 10:
                other = original[j] if j < len(original) else incoming[j-len(original)]
                flagged.append(dict(scene=incoming[i-len(original)]['id'], other=other['id'],
                                    descriptor_votes=int(votes[j]), homography_inliers=int(mask.sum()),
                                    excluded=bool(mask.sum() >= 20)))
    return flagged


def patch_plan(rng, size, width, height, ox, oy):
    x, y = (rng.randint(0, 512-size) for _ in range(2))
    tx, ty = ox + x, oy + y

    def donor_rank(point):
        sx, sy = point
        overlap = max(0, size-abs(sx-tx)) * max(0, size-abs(sy-ty))
        return overlap, -(sx-tx)**2-(sy-ty)**2

    donors = ((sx, sy) for sx in (0, width-size) for sy in (0, height-size))
    sx, sy = min(donors, key=donor_rank)
    return dict(size=size, target_xy=[x,y], source_xy=[sx,sy], overlap_pixels=donor_rank((sx,sy))[0])


def prepare():
    original_path = ROOT/'data/manifest_frozen.json'
    original = load(original_path)['scenes']
    source = load(SOURCE)
    exact = {row['sha256'] for row in original}
    incoming = []
    dispositions = []
    for row in source['records']:
        scene = 'DIV2K_train_' + Path(row['member']).stem
        if sha(row['path']) != row['sha256']:
            raise ValueError(f'Source hash drift: {scene}')
        if min(row['width'], row['height']) < 560 or row['sha256'] in exact:
            reason = 'too_small' if min(row['width'],row['height']) < 560 else 'exact_duplicate'
            dispositions.append(dict(scene=scene, reason=reason))
            continue
        exact.add(row['sha256'])
        incoming.append(dict(row, id=scene, source=row['path']))
    flags = overlap_candidates(original, incoming)
    overlap = {x['scene'] for x in flags if x['excluded']}
    selected = [r for r in incoming if r['id'] not in overlap][:N]
    if len(selected) != N:
        raise ValueError(f'Only {len(selected)} eligible scenes for a target of {N}')
    selected_ids = {r['id'] for r in selected}
    dispositions += [dict(scene=r['id'], reason='potential_overlap' if r['id'] in overlap else 'reserve')
                     for r in incoming if r['id'] not in selected_ids]

    rng = random.Random(SEED)
    records = []
    for i, row in enumerate(selected):
        scene = row['id']
        full = read(row['source'])
        height, width = full.shape[:2]
        assert (width,height) == (row['width'],row['height'])
        x, y = (width-560)//2, (height-560)//2
        render = full[y:y+560,x:x+560]
        hr_path = BASE/'hr'/f'{scene}.png'
        save(hr_path, render[24:536,24:536])
        patches = [patch_plan(rng, size, width, height, x+24, y+24) for size in PATCH_SIZES]
        direction = list(rng.choice(DIRECTIONS))
        for kind in ('normal','strong'):
            name = f'{scene}__{kind}.png'
            lr_path = BASE/'lr'/name
            lr = degrade(render, scene, kind=='strong')
            save(lr_path, lr)
            records.append(dict(scene=scene, group=scene, dataset='DIV2K_training',
                                source=row['source'], source_sha256=row['sha256'], patches=patches,
                                hr=str(hr_path), hr_sha256=sha(hr_path), lr=str(lr_path),
                                lr_sha256=sha(lr_path), lr_pixel_sha256=pixel_sha(lr),
                                sr=str(BASE/'vosr/images'/name), degradation=kind,
                                magnitude=(.5,2,6)[i%3], direction=direction))
    dump(BASE/'batch_inputs.json', dict(source_sha256=sha(SOURCE), original_scene_manifest_sha256=sha(original_path),
         sample_seed=SEED, n_scenes=N, duplicate_candidates=flags, dispositions=dispositions, records=records))
    print(f'Prepared {len(records)} inputs from {N} new source scenes, {len(flags)} potential overlaps flagged, '
          f'{len(overlap)} excluded', flush=True)


if __name__=='__main__':
    prepare()
