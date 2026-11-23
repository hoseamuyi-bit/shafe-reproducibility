"""Populate formal geometry caches as verified SR batches become available."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import monotonic, sleep

import cv2

from dev_pipeline import HERE, dump, sha
from experiment import load
from freeze import verify, verify_inputs
from measure import measure_record


def cache_record(record, settings):
    cv2.setNumThreads(2)
    sr_sha = sha(record['sr'])
    key = f"{record['scene']}__{record['degradation']}"
    path = HERE / 'formal' / 'geometry' / f'{key}.json'
    expected = dict(settings, sr_sha256=sr_sha)
    if path.exists() and load(path)['settings'] == expected:
        return
    rows = measure_record(record, reference=False, include_direct=True)
    temporary = path.with_suffix('.tmp')
    dump(temporary, dict(settings=expected, rows=rows))
    temporary.replace(path)


def run():
    base = HERE / 'formal'
    inputs_path = base / 'batch_inputs.json'
    inputs = load(inputs_path)
    verify_inputs(inputs, verify())
    settings = dict(measure_sha256=sha(HERE / 'measure.py'),
                    registration_sha256=sha(HERE / 'registration.py'),
                    sources_sha256=sha(inputs_path), reference=False, include_direct=True)
    pending = inputs['records']
    futures = set()
    completed = 0
    last_progress = monotonic()
    with ProcessPoolExecutor(max_workers=3) as pool:
        while pending or futures:
            ready = [r for r in pending if Path(r['sr']).with_suffix('.json').exists()]
            for record in ready:
                futures.add(pool.submit(cache_record, record, settings))
            pending = [r for r in pending if r not in ready]
            if futures:
                done, futures = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                    last_progress = monotonic()
                    if completed % 20 == 0 or completed == len(inputs['records']):
                        print(f'formal verified geometry caches: {completed}/{len(inputs["records"])}', flush=True)
            elif pending:
                if monotonic() - last_progress > 300:
                    raise TimeoutError('Waiting for formal SR inference to make progress')
                sleep(5)


if __name__ == '__main__':
    run()
