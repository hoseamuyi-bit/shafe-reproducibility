"""Regenerate all formal and RealSR VOSR inputs and verify their pixels."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import cv2

from replay_from_source import checked, stage
from replay_frozen import COHORTS, ROOT, checkpoint, read_cohort
from dev_pipeline import pixel_sha, read, sha, shift
from experiment import VOSR, VOSR_ARGS, load
from freeze import LOCK, verify


def replay(folder):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    frozen = verify()
    cohorts = {split: read_cohort(split, frozen) for split in COHORTS}
    environment = load(ROOT / 'environment_verified.json')
    runner_hash = environment['code_and_records'][str(VOSR)]
    checked(VOSR, runner_hash)
    weights = {item['path']: item['sha256'] for item in environment['vosr_weights']}
    for path, digest in weights.items():
        checked(path, digest)

    provenance = dict(
        script_sha256=sha(__file__), staging_sha256=sha(Path(__file__).with_name('replay_from_source.py')),
        lock_sha256=sha(LOCK), runner_sha256=runner_hash, weights=weights, args=VOSR_ARGS,
        inputs={split: cohort[2] for split, cohort in cohorts.items()},
    )
    output = folder / 'full_vosr_replay.json'
    result = load(output) if output.exists() else dict(provenance=provenance, batches={})
    if result['provenance'] != provenance:
        raise ValueError('The replay inputs or code have changed since the checkpoint')

    for split, (records, archived, _, _) in cohorts.items():
        for start in range(0, len(records), 100):
            rows = records[start:start + 100]
            label = f'{split}_{start:03d}'
            if label in result['batches']:
                continue
            attempt = 0
            while (batch := folder / f'{label}_{attempt}').exists():
                attempt += 1
            staged = [stage(split, row, batch) for row in rows]
            command = [sys.executable, '-B', str(VOSR), '--input', str(batch / 'input'),
                       '--output', str(batch / 'vosr'), *VOSR_ARGS]
            subprocess.run(command, check=True)
            run = load(batch / 'vosr/run.json')
            if (run['input_count'], run['upscale'], run['precision'], run['seed_per_image']) != (len(rows), 4, 'fp32', 42):
                raise ValueError(f'{label}: VOSR runtime configuration differs')

            checks = []
            for prior, fresh in zip(rows, staged, strict=True):
                key = (prior['scene'], prior['degradation'])
                prior_path = Path(prior['sr'])
                sidecar = load(prior_path.with_suffix('.json'))
                if sidecar != dict(lr_sha256=prior['lr_sha256'], runner_sha256=runner_hash,
                                   args=VOSR_ARGS, weights=weights, sr_sha256=sha(prior_path)):
                    raise ValueError(f'{key}: archival VOSR input or weight fingerprint differs')
                sr = read(fresh['sr'])
                fresh_hash = pixel_sha(sr)
                old_hash = pixel_sha(read(prior_path))
                zero_hash = pixel_sha(shift(sr, (0, 0)))
                if fresh_hash != old_hash or zero_hash != archived[0][key]['zero']['sr_pixel_sha256']:
                    raise ValueError(f'{key}: regenerated VOSR pixels differ')
                checks.append(dict(scene=key[0], degradation=key[1], dataset=prior['dataset'],
                                   source_sha256=prior['source_sha256'], sr_pixel_sha256=fresh_hash,
                                   zero_sr_pixel_sha256=zero_hash, lr_sha256=sha(fresh['lr']),
                                   hr_sha256=sha(fresh['hr'])))
            result['batches'][label] = dict(split=split, run=run, checks=checks)
            checkpoint(output, result)
            print(f'{label}: {len(checks)} source inputs and VOSR outputs match', flush=True)

    for split, (records, _, _, _) in cohorts.items():
        checks = [row for batch in result['batches'].values() if batch['split'] == split for row in batch['checks']]
        if [(r['scene'], r['degradation']) for r in checks] != [(r['scene'], r['degradation']) for r in records]:
            raise ValueError(f'{split}: incomplete replay')
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    checkpoint(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    replay(parser.parse_args().output.resolve())
