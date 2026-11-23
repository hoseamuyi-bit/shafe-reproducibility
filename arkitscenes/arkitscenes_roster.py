"""Resolve the visit roster from the locked acquisition design."""

from hashlib import file_digest
import json
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def registered_candidates(directory, plan):
    if 'candidates' in plan:
        return plan['candidates']

    root = Path(__file__).resolve().parent
    parent_path = root / 'arkitscenes_temporal_training' / 'protocol.json'
    script_path = root / 'extend_arkitscenes_training_temporal.py'
    count = plan['expansion']['max_visits']
    if (count != 24 or
            plan['selection'] != 'The next 24 pose-priority visits from the archived Training draw.' or
            plan['parent_protocol_sha256'] != sha256(parent_path) or
            plan['extension_script_sha256'] != sha256(script_path)):
        raise ValueError('Could not resolve the locked Training extension roster')
    parent = json.loads(parent_path.read_bytes())
    if (any(plan[k] != parent[k] for k in ('index_sha256', 'hashes', 'settings')) or
            plan['expansion'] != dict(parent['expansion'], max_visits=count)):
        raise ValueError('Training extension settings differ from the parent acquisition')

    used = {row['visit_id'] for row in parent['candidates']}
    ordered = [row for row in parent['ranking'] if row['priority_pairs'] and row['visit_id'] not in used]
    candidates = ordered[:count]
    if len({row['visit_id'] for row in candidates}) != count:
        raise ValueError(f'Invalid visit roster in {directory}')
    return candidates
