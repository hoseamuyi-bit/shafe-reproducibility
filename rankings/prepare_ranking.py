"""Sample COCO scenes before any SR inference or scoring on the new cohort."""
from concurrent.futures import ThreadPoolExecutor
import io
import itertools
import json
import random
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

from ranking_common import ASSETS, ROOT, NEXT, atomic_json, file_sha, load
from dev_pipeline import degrade, pixel_sha, read, save
from holdout_data import overlap_candidates

SEED = 20261129
N_PER_SPLIT = 72
MAGNITUDES = (.5, 2, 6)
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
INDICES = {
    'val2017': (ASSETS / 'annotations_trainval2017.zip', 'instances_val2017.json'),
    'test2017': (NEXT / 'experiment_audit/coco_confirm/image_info_test2017.zip',
                 'image_info_test2017.json'),
}
HISTORY = (
    NEXT / 'data/manifest_frozen.json',
    NEXT / 'experiment_audit/holdout/batch_inputs.json',
    NEXT / 'external/batch_inputs.json',
    NEXT / 'experiment_audit/coco_confirm/batch_inputs.json',
)


def fetch(entry):
    split, row = entry
    filename = row['file_name']
    url = f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}'
    path = ROOT / 'inputs' / 'source' / split / filename
    meta = dict(id=f"COCO_{split}_{row['id']:012d}", coco_id=row['id'], split=split,
                url=url, license=row['license'], width=row['width'], height=row['height'])
    for attempt in range(4):
        try:
            if path.is_file():
                data = path.read_bytes()
            else:
                host = url if attempt % 2 == 0 else f'https://images.cocodataset.org/{split}/{filename}'
                with urlopen(host, timeout=22) as response:
                    data = response.read(8_000_001)
            if len(data) > 8_000_000:
                raise ValueError('JPEG exceeds size limit')
            with Image.open(io.BytesIO(data)) as image:
                if image.size != (row['width'], row['height']) or image.format != 'JPEG':
                    raise ValueError('JPEG dimensions or format differ from COCO metadata')
                image.load()
            if not path.is_file():
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(data)
                tmp.replace(path)
            return dict(meta, source=str(path), sha256=file_sha(path))
        except (OSError, HTTPError, ValueError) as exc:
            if path.exists():
                path.unlink()
            if attempt == 3:
                return dict(meta, error=f'{type(exc).__name__}: {exc}')


def historical_rows():
    rows = {}
    for path in HISTORY:
        obj = load(path)
        for row in obj.get('scenes', obj.get('records', [])):
            identifier = row.get('id', row.get('scene'))
            rows[identifier] = dict(id=identifier, source=row['source'],
                                    sha256=row.get('sha256', row.get('source_sha256')))
    return list(rows.values())


def prepare():
    destination = ROOT / 'inputs/manifest.json'
    if destination.exists():
        return
    history = historical_rows()
    old_ids = {int(r['id'].split('_')[-1]) for r in history if r['id'].startswith('COCO_')}
    used = {r['sha256'] for r in history}
    fetched, exclusions, indices = [], [], {}
    for split, (index, member) in INDICES.items():
        with ZipFile(index) as archive:
            obj = json.loads(archive.read(f'annotations/{member}'))
        size_ok = sorted((r for r in obj['images'] if min(r['width'], r['height']) >= 560),
                         key=lambda row: row['id'])
        eligible = [r for r in size_ok if r['id'] not in old_ids]
        indices[split] = dict(index_sha256=file_sha(index), licenses=obj['licenses'],
                              indexed=len(obj['images']), size_eligible=len(size_ok),
                              prior_id_exclusions=sorted(r['id'] for r in size_ok if r['id'] in old_ids))
        rng = random.Random(f'{SEED}|{split}|sample')
        rng.shuffle(eligible)
        (ROOT / 'inputs' / 'source' / split).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=12) as pool:
            for i, row in enumerate(pool.map(fetch, ((split, r) for r in eligible[:96])), 1):
                if 'error' in row:
                    exclusions.append(dict(row, disposition='unavailable'))
                elif row['sha256'] in used:
                    exclusions.append(dict(row, disposition='exact_duplicate'))
                else:
                    fetched.append(row)
                    used.add(row['sha256'])
                if i % 24 == 0:
                    print(f'{split}: checked {i}/96 downloaded candidates', flush=True)
    flags = overlap_candidates(history, fetched)
    overlapping = {r['scene'] for r in flags if r['excluded']}
    records = []
    for split in INDICES:
        sample = [r for r in fetched if r['split'] == split and r['id'] not in overlapping][:N_PER_SPLIT]
        if len(sample) != N_PER_SPLIT:
            raise ValueError(f'{split}: only {len(sample)} eligible downloads after checks')
        selected_ids = {r['id'] for r in sample}
        exclusions.extend(dict(r, disposition='potential_overlap' if r['id'] in overlapping else 'reserve')
                          for r in fetched if r['split'] == split and r['id'] not in selected_ids)
        cells = list(itertools.product(MAGNITUDES, DIRECTIONS, range(6)))
        random.Random(f'{SEED}|{split}|allocation').shuffle(cells)
        for row, (magnitude, (dx, dy), rep) in zip(sample, cells, strict=True):
            name = row['id']
            full = read(row['source'])
            x, y = (row['width'] - 560) // 2, (row['height'] - 560) // 2
            render = full[y:y + 560, x:x + 560]
            hr = render[24:536, 24:536]
            lr = degrade(render, name, strong=True)
            paths = {key: ROOT / 'inputs' / key / f'{name}.png' for key in ('hr', 'lr')}
            save(paths['hr'], hr)
            save(paths['lr'], lr)
            records.append(dict(row, scene=name, render_origin=[x, y], magnitude=magnitude,
                                sr_added=[magnitude * dx, magnitude * dy],
                                hr_added=[(-1)**rep * magnitude * -dy, (-1)**rep * magnitude * dx],
                                hr=str(paths['hr']), lr=str(paths['lr']),
                                hr_sha256=file_sha(paths['hr']), lr_sha256=file_sha(paths['lr']),
                                hr_pixel_sha256=pixel_sha(hr), lr_pixel_sha256=pixel_sha(lr)))
    atomic_json(destination, dict(seed=SEED, indices=indices, historical_n=len(history),
                                  historical_manifest_sha256={str(p): file_sha(p) for p in HISTORY},
                                  exclusions=exclusions, overlap_flags=flags, records=records))
    print(f'Prepared {len(records)} new scenes. Checked {len(history)} historical references.', flush=True)


if __name__ == '__main__':
    prepare()
