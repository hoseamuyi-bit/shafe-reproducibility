"""Resume the frozen draw, recording the trajectories Apple lists as missing."""

import ast
from unittest.mock import patch

import run_arkitscenes_holdout as holdout


def apple_missing_trajectories():
    source = holdout.OUT / 'apple_download_data.py'
    nodes = ast.parse(source.read_text(encoding='utf-8')).body
    matches = [node.value for node in nodes if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'missing_3dod_assets_video_ids'
                       for target in node.targets)]
    if len(matches) != 1:
        raise ValueError('Apple download script does not contain one missing-asset roster')
    return set(ast.literal_eval(matches[0])), holdout.screen.digest(source)


def run():
    if holdout.screen.digest(holdout.OUT / 'apple_official_split.csv') != holdout.screen.digest(holdout.train.INDEX):
        raise ValueError('Official split index differs from the frozen split')
    missing, source_hash = apple_missing_trajectories()
    original = holdout.train.priority

    def priority(row):
        if row['video_id'] in missing:
            return dict(row, priority_pairs=0, priority_centers=0,
                        availability='Apple lists lowres_wide.traj as missing',
                        official_roster_sha256=source_hash)
        return original(row)

    with patch.object(holdout.train, 'priority', priority):
        holdout.run()


if __name__ == '__main__':
    run()
