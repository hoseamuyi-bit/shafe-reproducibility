"""Paths and integrity helpers shared by the ranking stages."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
NEXT = ROOT.parent / 'shafe_next'
LOCK = ROOT / 'formal_lock.json'
MODELS = ('VOSR', 'Real-ESRGAN', 'SwinIR-S', 'Bicubic')
METRICS = ('PSNR', 'SSIM', 'SHAFE', 'ST-LPIPS')
CONDITIONS = ('zero', 'hr_global', 'sr_global', 'both_global')
METHODS = ('fixed', 'direct', 'global_lr', 'lk_lr', 'oracle', 'clean_hr')

sys.path[:0] = [str(NEXT), str(NEXT / 'experiment_audit')]


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def verify_lock():
    lock = load(LOCK)
    for path, expected in lock['sha256'].items():
        if file_sha(path) != expected:
            raise ValueError(f'Frozen file changed: {path}')
    return lock, file_sha(LOCK)
