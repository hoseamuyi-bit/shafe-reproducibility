"""Quantify rank disagreement on RealSR and the response to known added shifts."""
import numpy as np

from analyze_ranking import EPSILONS, NEAR_TIE, SIGNS, bootstrap_weights, effect, pair_delta, sign, view
from ranking_common import CONDITIONS, METRICS, METHODS, MODELS, ROOT, file_sha, load


def analyze_camera():
    folder = ROOT / 'camera_ranking'
    lock = load(folder / 'lock.json')
    digest = file_sha(folder / 'lock.json')
    records = lock['records']
    camera = np.array([row['dataset'] for row in records])
    magnitude = np.array([row['magnitude'] for row in records])
    cube = np.empty((len(records), len(CONDITIONS), len(METHODS), len(MODELS), len(METRICS)))
    direct_moves, lk_moves, global_moves = [], [], []
    for index, row in enumerate(records):
        scene = load(folder / 'scores' / (row['scene'] + '.json'))
        if scene['checks']['config'] != digest or scene['lr_pixel_sha256'] != row['lr_pixel_sha256']:
            raise ValueError(f"Camera score provenance mismatch: {row['scene']}")
        cases = {(r['condition'], r['model']): r for r in scene['cases']}
        if len(cases) != len(CONDITIONS) * len(MODELS) or len(cases) != len(scene['cases']):
            raise ValueError(f"Incomplete camera scene: {row['scene']}")
        native = [cases['zero', model] for model in MODELS]
        direct_moves.append(sum(any(r['actions']['direct']) for r in native))
        lk_moves.append(any(native[0]['actions']['lk_lr']))
        global_moves.append(any(native[0]['actions']['global_lr']))
        for c, condition in enumerate(CONDITIONS):
            for m, model in enumerate(MODELS):
                for a, alignment in enumerate(METHODS):
                    scores = cases[condition, model]['scores'][alignment]
                    cube[index, c, a, m] = [scores[metric] for metric in METRICS]
    if not np.isfinite(cube).all():
        raise ValueError('Camera score cube has non-finite values')
    oriented = cube * SIGNS
    pool = {}
    for name, use in [('pooled', np.ones(len(records), dtype=bool)),
                      *[(name, camera == name) for name in sorted(set(camera))]]:
        groups = list(zip(camera[use], magnitude[use]))
        weights = bootstrap_weights(groups, np.random.default_rng(lock['bootstrap_seed']))
        native = oriented[use, CONDITIONS.index('zero')]
        delta = pair_delta(native)
        discordance = {}
        for label, tolerance in [('strict', EPSILONS), ('wider_ties', NEAR_TIE)]:
            order = sign(delta, np.asarray(tolerance))
            discordance[label] = {}
            for left, right in [('direct', 'lk_lr'), ('direct', 'fixed'), ('lk_lr', 'fixed'),
                                ('global_lr', 'lk_lr')]:
                agreement = order[:, METHODS.index(left)] * order[:, METHODS.index(right)]
                discordance[label][f'{left} vs {right}'] = {
                    metric: dict(opposite=int((agreement[:, :, q] == -1).sum()),
                                 both_ordered=int((agreement[:, :, q] != 0).sum()),
                                 total=agreement[:, :, q].size,
                                 rate_pct=effect((agreement[:, :, q] == -1).mean(axis=1), weights, 100))
                    for q, metric in enumerate(METRICS)}
        pool[name] = dict(scenes=int(use.sum()), native_discordance=discordance,
                          native_hr_moves=dict(direct=int(np.asarray(direct_moves)[use].sum()),
                                               lk_lr=int(np.asarray(lk_moves)[use].sum()),
                                               global_lr=int(np.asarray(global_moves)[use].sum())),
                          relative_to_published_geometry={
                              condition: view(oriented[use, c], weights, np.asarray(EPSILONS))
                              for c, condition in enumerate(CONDITIONS)})
    return pool
