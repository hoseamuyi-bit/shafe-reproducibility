"""Deterministically sample and prepare an independent COCO test2017 cohort."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import random
import time
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

from holdout_data import overlap_candidates, patch_plan
from holdout_run import BASE, ROOT, atomic_dump, load, sha
from dev_pipeline import degrade, pixel_sha, read, save
from experiment import DIRECTIONS


FOLDER = BASE.parent / 'coco_confirm'
S3 = 'https://s3.amazonaws.com/images.cocodataset.org'
SEED = 20261127
N = 144


def fetch(meta):
    filename = meta['file_name']
    url = f'{S3}/test2017/{filename}'
    path = FOLDER / 'source' / filename
    for attempt in range(3):
        try:
            if not path.exists():
                with urlopen(url, timeout=25) as response:
                    data = response.read(5_000_001)
                assert len(data) <= 5_000_000
                with Image.open(io.BytesIO(data)) as image:
                    assert image.size == (meta['width'], meta['height']) and image.format == 'JPEG'
                    image.load()
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(data)
                temporary.replace(path)
            with Image.open(path) as image:
                assert image.size == (meta['width'], meta['height']) and image.format == 'JPEG'
                image.load()
            return dict(meta, id=f'COCO_test2017_{meta["id"]:012d}', url=url,
                        source=str(path.resolve()), sha256=sha(path))
        except (OSError, AssertionError) as error:
            if path.exists():
                path.unlink()
            if attempt == 2:
                return dict(meta, error=f'{type(error).__name__}: {error}', url=url)
            time.sleep(.5 * 2**attempt)


def prepare():
    destination = FOLDER / 'batch_inputs.json'
    if destination.exists():
        raise ValueError('Input manifest already exists')
    metadata = FOLDER / 'image_info_test2017.zip'
    with ZipFile(metadata) as archive:
        index = json.loads(archive.read('annotations/image_info_test2017.json'))
    eligible = sorted((r for r in index['images'] if min(r['width'], r['height']) >= 560), key=lambda r: r['id'])
    random.Random(SEED).shuffle(eligible)
    (FOLDER / 'source').mkdir(exist_ok=True)
    print(f'COCO test2017: {len(eligible)} sources eligible by size', flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        fetched = []
        for i, row in enumerate(pool.map(fetch, eligible[:N + 24]), 1):
            fetched.append(row)
            if i % 24 == 0:
                print(f'Downloaded or checked {i}/{N + 24}', flush=True)
    original_path = ROOT / 'data' / 'manifest_frozen.json'
    holdout_path = BASE / 'batch_inputs.json'
    original = load(original_path)['scenes']
    original += [dict(id=r['scene'], source=r['source'], sha256=r['source_sha256'])
                 for r in load(holdout_path)['records'] if r['degradation'] == 'strong']
    cameras_path = ROOT / 'external' / 'batch_inputs.json'
    original += [dict(id=r['scene'], source=r['source'], sha256=r['source_sha256'])
                 for r in load(cameras_path)['records']]
    exact = {r['sha256'] for r in original}
    candidates, exclusions = [], []
    for row in fetched:
        if 'error' in row or row['sha256'] in exact:
            exclusions.append(dict(row, disposition='unavailable' if 'error' in row else 'exact_duplicate'))
        else:
            candidates.append(row)
            exact.add(row['sha256'])
    flags = overlap_candidates(original, candidates)
    overlaps = {r['scene'] for r in flags if r['excluded']}
    selected = [r for r in candidates if r['id'] not in overlaps][:N]
    if len(selected) != N:
        raise ValueError(f'Only {len(selected)} available after overlap checks')
    selected_ids = {r['id'] for r in selected}
    exclusions += [dict(r, disposition='potential_overlap' if r['id'] in overlaps else 'reserve')
                   for r in candidates if r['id'] not in selected_ids]
    records = []
    rng = random.Random(SEED + 1)
    for i, row in enumerate(selected):
        scene, source = row['id'], row['source']
        full = read(source)
        height, width = full.shape[:2]
        x, y = (width - 560) // 2, (height - 560) // 2
        render = full[y:y+560, x:x+560]
        hr_path, lr_path = FOLDER / 'hr' / f'{scene}.png', FOLDER / 'lr' / f'{scene}.png'
        save(hr_path, render[24:536, 24:536])
        lr = degrade(render, scene, strong=True)
        save(lr_path, lr)
        patch = patch_plan(rng, 256, width, height, x + 24, y + 24)
        records.append(dict(scene=scene, group=scene, dataset='COCO_test2017', degradation='strong',
                            source=source, source_sha256=row['sha256'], source_url=row['url'],
                            source_license=row['license'], render_origin=[x,y], patches=[patch],
                            hr=str(hr_path), hr_sha256=sha(hr_path), lr=str(lr_path), lr_sha256=sha(lr_path),
                            lr_pixel_sha256=pixel_sha(lr), sr=str(FOLDER/'vosr/images'/lr_path.name),
                            magnitude=(.5, 2, 6)[i % 3], direction=rng.choice(DIRECTIONS)))
    atomic_dump(destination, dict(seed=SEED, n_scenes=N, n_size_eligible=len(eligible), source=S3,
                source_index_sha256=sha(metadata), overlap_reference_hashes={p.name + '@' + p.parent.name: sha(p)
                for p in (original_path, holdout_path, cameras_path)}, duplicate_candidates=flags,
                exclusions=exclusions, records=records))
    print(f'Prepared {len(records)} independent scenes after checking {len(flags)} overlap candidates', flush=True)


if __name__ == '__main__':
    prepare()
