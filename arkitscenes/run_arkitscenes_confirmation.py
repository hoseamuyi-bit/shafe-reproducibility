"""Freeze and run a visit-disjoint temporal reference comparison."""

from argparse import ArgumentParser
from collections import defaultdict
import csv
from pathlib import Path
from shutil import disk_usage
from unittest.mock import patch

import cv2

import run_arkitscenes_holdout as holdout
from resume_arkitscenes_holdout import apple_missing_trajectories


train, screen, ROOT = holdout.train, holdout.screen, holdout.ROOT
OUT = ROOT / 'arkitscenes_temporal_confirmation'
PARENT = holdout.OUT / 'protocol.json'
RADIUS_MATCHED = 50 * 88 / 448
MARGINS = (0.0005, 0.001, 0.002, 0.005)
MODELS = ('VOSR', 'Real-ESRGAN', 'SwinIR-S', 'Bicubic')
BOOTSTRAP = dict(unit='visit', draws=10000, seed=20261122, interval='percentile 95%')


def history(rows):
    video_visits = {row['video_id']: row['visit_id'] for row in rows}
    excluded = defaultdict(set)
    for row in rows:
        if row['fold'] == 'Validation':
            excluded[row['visit_id']].add('official Validation')

    def visit_records(node):
        if isinstance(node, list):
            for value in node:
                yield from visit_records(value)
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('visit', 'visit_id') and isinstance(value, (str, int)):
                    yield str(value)
                if key in ('video', 'video_id') and isinstance(value, (str, int)):
                    if str(value) in video_visits:
                        yield video_visits[str(value)]
                if isinstance(value, (dict, list)):
                    yield from visit_records(value)

    sources = set(ROOT.glob('arkitscenes_*.json'))
    for directory in ROOT.glob('arkitscenes_temporal*'):
        if directory != OUT and directory.is_dir():
            sources.update(path for path in directory.glob('*.json')
                           if path.name in ('inputs.json', 'protocol.json', 'sampling_protocol.json'))
    hashes = {}
    for path in sorted(sources):
        name = path.relative_to(ROOT).as_posix()
        hashes[name] = screen.digest(path)
        for visit in visit_records(screen.read(path)):
            excluded[visit].add(name)
    sites = sorted(path for path in ROOT.rglob('arkit_*')
                   if path.is_dir() and not path.is_relative_to(OUT))
    for path in sites:
        visit = video_visits.get(path.name.removeprefix('arkit_'))
        if visit is not None:
            excluded[visit].add(path.relative_to(ROOT).as_posix())
    return hashes, {visit: sorted(sources) for visit, sources in sorted(excluded.items())}


def freeze():
    parent = screen.read(PARENT)
    index_hash = screen.digest(train.INDEX)
    if index_hash != parent['index_sha256']:
        raise ValueError('The official index differs from the frozen Apple index')
    with train.INDEX.open(encoding='utf-8', newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['video_id'])
    history_hashes, excluded = history(rows)
    available = {}
    for row in rows:
        if row['fold'] == train.FOLD and row['visit_id'] not in excluded:
            available.setdefault(row['visit_id'], row)
    sample = sorted(available.values(), key=lambda row: row['video_id'])
    original_path = OUT / 'sampling_protocol_pre_smoke_fix.json'
    original = screen.read(original_path) if original_path.exists() else None
    if original is not None and original['sample'] != sample:
        raise ValueError('Smoke-test fix must preserve the original roster')
    missing, missing_hash = apple_missing_trajectories()
    hashes = dict(parent['hashes'])
    scripts = (
        Path(__file__).name, 'score_arkitscenes_confirmation.py', 'run_vosr_pread.py',
        'verify_arkitscenes_temporal_acquisition.py', 'arkitscenes_roster.py',
        'audit_arkitscenes_reference_controls.py', 'check_arkitscenes_temporal_metrics.py',
        'check_temporal_shafe_radius.py', 'compare_temporal_reference_controls.py',
        'resume_arkitscenes_holdout.py', 'probe_arkitscenes_outside_pose.py',
        '../shafe_optical/score_pairs.py', '../shafe_ranking/ranking_common.py',
        '../shafe_ranking/ranking_scores.py', '../shafe_ranking/ranking_models.py',
        '../shafe_ranking/formal_lock.json',
    )
    hashes.update({name: screen.digest(ROOT / name) for name in scripts})
    if any(screen.digest(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('The acquisition or scoring sources differ from the archived versions')
    endpoint = dict(
        baseline='Fixed synchronous HR, fixed LR and same four model outputs on 88x88 HR support.',
        models=list(MODELS),
        definition='A pair reverses if both absolute SHAFE gaps strictly exceed the margin and '
                   'their directions differ. Divide each count by six and weight each visit equally.',
        primary=dict(radius=50, pose='linear', practical_tie=MARGINS[0],
                     contrast='depth-corrected delayed HR minus same-frame pose round trip'),
        support='Exactly one first qualifying window per visit. Primary and sensitivity contrasts '
                'use the same windows with full same-frame round-trip support in both interpolants.',
        sensitivity=dict(radii=[50, RADIUS_MATCHED], poses=list(screen.POSE_NAMES), margins=list(MARGINS)),
        uncertainty=BOOTSTRAP,
        criterion='At least two common-support visits, lower bound of the primary 95% visit bootstrap '
                  'interval strictly above zero, and exact two-sided visit sign-flip p < 0.05 under '
                  'the paired exchangeability null. Report all runs regardless of direction.',
    )
    if original is not None and original['endpoint'] != endpoint:
        raise ValueError('Smoke-test fix must preserve the original endpoint')
    return train.locked('sampling_protocol.json', dict(
        official_index_url='https://raw.githubusercontent.com/apple/ARKitScenes/main/depth_upsampling/upsampling_train_val_splits.csv',
        official_archive_root=train.BASE, index_sha256=index_hash,
        official_missing_script_sha256=missing_hash,
        official_missing_video_ids=sorted(missing), parent_sha256=screen.digest(PARENT),
        history_sha256=history_hashes, excluded_visits=excluded, hashes=hashes,
        settings=dict(parent['settings'], scope='Visit-disjoint confirmation of temporal reference substitution.'),
        selection='Use all remaining Training visits, lowest video ID per visit. Keep the archived '
                  'pose, depth, RGB order and gates, with no score-adaptive exclusions.',
        sample=sample, endpoint=endpoint, expansion=parent['expansion'],
        pre_scoring_amendment=dict(original_sampling_sha256=screen.digest(original_path),
                                   reason='Archived positive control smoke test found an optional RGB-bias '
                                          'diagnostic key was absent. Handle the absent key; roster, data, '
                                          'controls and endpoint are unchanged.') if original is not None else None))


def clear_tails(row):
    train.keep_scored_frames(row)
    site = OUT / ('arkit_' + row['video'])
    if site.resolve().parent != OUT.resolve():
        raise ValueError(f'Unexpected trial cache location: {site}')
    for asset in ('highres_depth', 'lowres_wide', 'wide'):
        tail = site / f'{asset}_tail.bin'
        if not tail.exists():
            continue
        if tail.resolve().parent != site.resolve():
            raise ValueError(f'Unexpected trial cache entry: {tail}')
        header = tail.with_suffix('.headers')
        receipt = site / f'{asset}_tail_receipt.json'
        evidence = dict(
            url=f'{train.BASE}/raw/{train.FOLD}/{row["video"]}/{asset}.zip',
            bytes=tail.stat().st_size, sha256=screen.digest(tail),
            headers=header.read_text(encoding='ascii'))
        if receipt.exists() and screen.read(receipt) != evidence:
            raise ValueError(f'Tail receipt changed: {tail}')
        screen.write(receipt, evidence)
        tail.unlink()


def acquire(sampling_path):
    sample = screen.read(sampling_path)
    missing = set(sample['official_missing_video_ids'])

    def priority(row):
        if row['video_id'] in missing:
            return dict(row, priority_pairs=0, priority_centers=0,
                        availability='Apple lists lowres_wide.traj as missing')
        return train.priority(row)

    ranking = holdout.collect(sample['sample'], priority, 12, 'pose')
    ranking.sort(key=lambda row: (-row['priority_centers'], -row['priority_pairs'], row['video_id']))
    candidates = [row for row in ranking if row['priority_pairs']]
    protocol_path = train.locked('protocol.json', dict(
        sampling_protocol_sha256=screen.digest(sampling_path),
        ranking=ranking, candidates=candidates, hashes=sample['hashes'],
        settings=sample['settings'], endpoint=sample['endpoint'],
        expansion=dict(sample['expansion'], max_visits=len(candidates))))
    protocol_sha = screen.digest(protocol_path)
    print(f'{len(candidates)} pose-qualified visits for depth/RGB screening', flush=True)

    def acquire_visit(row):
        if disk_usage(OUT).free < 256 * 1024**2:
            raise OSError('Less than 256 MiB free before starting another visit')
        result = screen.run_visit(row, protocol_sha)
        if not result.get('error'):
            clear_tails(result)
            if result['selected']:
                print('Selected', result['video'], 'visit', result['visit'], flush=True)
        return result

    with patch.multiple(screen, OUT=OUT, FrameArchive=train.train_archive,
                        acquire_geometry=train.acquire):
        final = holdout.collect(candidates, acquire_visit, 6, 'frames')
    screen.write(OUT / 'inputs.json', dict(protocol_sha256=protocol_sha, visits=final))
    print(f'{sum(bool(row["selected"]) for row in final)} eligible visits', flush=True)


def main(stage):
    cv2.setNumThreads(1)
    for name in ('pose_priority', 'screened', 'visit_plans'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    with patch.object(train, 'OUT', OUT), patch.object(holdout, 'OUT', OUT):
        # Use the already verified official source script from the preceding download.
        with patch('resume_arkitscenes_holdout.holdout.OUT', PARENT.parent):
            sampling_path = freeze()
        print(f'Frozen {len(screen.read(sampling_path)["sample"])} disjoint visits '
              f'{screen.digest(sampling_path)}', flush=True)
        if stage in ('acquire', 'all') and not (OUT / 'inputs.json').exists():
            acquire(sampling_path)
    if stage in ('score', 'all'):
        from score_arkitscenes_confirmation import run
        run(sampling_path)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'acquire', 'score', 'all'))
    main(parser.parse_args().stage)
