"""Commit the new-scene inputs and analysis before formal model inference."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import __version__ as pillow_version
import scipy
import torch
import torchvision

from analyze_ranking import BOOT_REPS, BOOT_SEED, EPSILONS, MIN_PP, NEAR_TIE
from dev_pipeline import CONFIG, OLD
from ranking_common import ASSETS, CONDITIONS, LOCK, METRICS, METHODS, MODELS, NEXT, ROOT
from ranking_common import atomic_json, file_sha, load, verify_lock
from ranking_models import REAL_WEIGHT, SWIN_WEIGHT, VOSR, VOSR_ARGS
from ranking_scores import SHAFE_WEIGHT, ST_RUNTIME, ST_WEIGHT


def freeze():
    if LOCK.exists():
        _, digest = verify_lock()
        print(f'Existing formal lock verified: {digest}', flush=True)
        return
    if (ROOT / 'formal').exists():
        raise RuntimeError('Formal output folder already exists before the lock')
    manifest = ROOT / 'inputs/manifest.json'
    records = load(manifest)['records']
    if len({row['scene'] for row in records}) != 144:
        raise ValueError('Expected 144 distinct new scenes')
    cells = Counter((row['split'], row['magnitude'], tuple(np.sign(row['sr_added']))) for row in records)
    if len(cells) != 24 or set(cells.values()) != {6}:
        raise ValueError('The split, magnitude and direction cells must each have six scenes')
    verification = ROOT / 'development_verification.json'
    if load(verification)['repeated_rgb_outputs'] != 12:
        raise ValueError('Development repeatability check is missing')

    files = [*sorted(ROOT.glob('*.py')), manifest, verification, VOSR, REAL_WEIGHT,
             SWIN_WEIGHT, ST_WEIGHT, SHAFE_WEIGHT, ASSETS / 'SwinIR.commit',
             ASSETS / 'ShiftTolerant-LPIPS.commit', ASSETS / 'SwinIR/models/network_swinir.py',
             OLD / 'work/shafe_formal/run_formal.py', OLD / 'work/shafe_pilot/run_shafe.py',
             OLD / 'work/shafe_pilot/SHAFE.py', VOSR.parent / 'upstream/inference_vosr_onestep.py']
    files += sorted((ST_RUNTIME / 'stlpips_pytorch').rglob('*.py'))
    files += [NEXT / name for name in ('dev_pipeline.py', 'registration.py', 'experiment.py',
                                      'realesrgan_x4.py', 'experiment_audit/holdout_data.py',
                                      'experiment_audit/robust_controls.py',
                                      'experiment_audit/holdout_analysis.py',
                                      'experiment_audit/pixel_metric_audit.py')]
    # The historical runner checks these files against its published hash list.
    vosr_weights = load(ROOT / 'vosr_weight_verification.json')
    weight_paths = {VOSR.parent / 'ckpts' / name: item['sha256']
                    for name, item in vosr_weights.items()}
    hashes = {str(p): file_sha(p) for p in files}
    hashes.update({str(p): digest for p, digest in weight_paths.items()})

    audit_rng = random.Random(20261201)
    audit_scenes = [audit_rng.choice([row['scene'] for row in records
                                     if (row['split'], row['magnitude']) == (split, magnitude)])
                    for split in ('val2017', 'test2017') for magnitude in (.5, 2, 6)]
    ranking = dict(
        primary='SHAFE in sr_global. Percentage-point flip rate of direct minus lk_lr.',
        reversal='Order of a model pair is strictly opposite its oracle order on the same scene, '
                 'condition and metric. Ties never count as reversals. Denominator includes all six '
                 'pairs per scene. Also record ties and both-ordered denominators.',
        aggregate='Average absolute metric values across the same scenes, compute each model-pair '
                  'mean difference and compare signs against oracle. Scene-stratified percentile '
                  'intervals and paired bootstrap reversal fractions for each model pair.',
        secondary='Other metrics and conditions, split and magnitude strata, wider tie sensitivity, '
                  'shifting one SR model at a time, action residuals and resampling sensitivity. '
                  'Secondary intervals are descriptive and unadjusted.',
        success=dict(min_effect_pp=MIN_PP, ci95_lower_strictly_above=0),
        resampling='Paired source scene bootstrap within COCO split × magnitude. Preserve all model '
                   'outputs, metrics, perturbations, and methods for a scene in each draw.',
        seed=BOOT_SEED, replicates=BOOT_REPS, epsilon=dict(zip(METRICS, EPSILONS)),
        wider_tie_sensitivity=dict(zip(METRICS, NEAR_TIE)))
    atomic_json(LOCK, dict(
        locked_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent images for a follow-up to the earlier single-model score audit. '
              'The COCO sampling and all endpoints were locked before new-scene inference and scoring.',
        scene_counts={'val2017': 72, 'test2017': 72}, audit_scenes=audit_scenes,
        inputs=dict(lr=(128, 128), source_render=(560, 560), hr=(512, 512),
                    degradation={key: CONFIG[key] for key in ('scale', 'blur_strong', 'noise_sigma',
                                 'jpeg_quality', 'jpeg_subsampling', 'noise_seed', 'downsample')},
                    roi_xyxy=(32, 32, 480, 480), magnitudes=(.5, 2, 6),
                    vectors='Axial SR translation with HR orthogonal in both_global, equally balanced '
                            'by split, magnitude and SR direction. All models share an injected vector.',
                    interpolation='OpenCV bicubic. SR is never moved after condition injection.'),
        models=MODELS, vosr_cli=VOSR_ARGS, metrics=METRICS, conditions=CONDITIONS, methods=METHODS,
        score_details=dict(PSNR='8-bit PIL L, MAX=255', SSIM='8-bit PIL L, Gaussian 11, σ=1.5, valid',
                           SHAFE='8-bit PIL L, paper parameters r=50, T=0.01, layers [1,2]',
                           ST_LPIPS='8-bit RGB, official shift-tolerant AlexNet, normalized from [0,1]'),
        controls=dict(fixed='Leave HR in place', direct='Map HR to each SR, ±16 HR px with 0.25 px refinement',
                      global_lr='Map HR to LR using uniform normalized residual, ±8 HR px',
                      lk_lr='LR to projected HR and back, median of ≥8 accepted LK tracks',
                      oracle='Undo only the known HR translation and retain actual SR',
                      clean_hr='Original HR crop, for interpolation sensitivity'),
        ranking=ranking, environment=dict(python=sys.version.split()[0], torch=torch.__version__,
                                         torchvision=torchvision.__version__, cuda=torch.version.cuda,
                                         gpu=torch.cuda.get_device_name(), numpy=np.__version__,
                                         scipy=scipy.__version__, pillow=pillow_version, opencv=cv2.__version__),
        sha256=hashes))
    _, digest = verify_lock()
    print(f'Formal lock verified: {digest}', flush=True)


if __name__ == '__main__':
    freeze()
