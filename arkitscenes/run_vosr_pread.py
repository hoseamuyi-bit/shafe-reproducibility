"""Run the frozen VOSR entry point with positional reads for the DiT checkpoint."""

import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[4] / '2026-11-19/ji-x/work/vosr/run_baseline.py'
DIT_WEIGHTS = RUNNER.parent / 'ckpts/VOSR_0.5B_os/checkpoints/ema_model.safetensors'


def main():
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    import safetensors.torch

    load_file = safetensors.torch.load_file

    def load_checkpoint(filename, device='cpu', *, backend='mmap'):
        if Path(filename).resolve() == DIT_WEIGHTS.resolve():
            backend = 'pread'
            print('Reading the frozen LightningDiT checkpoint with pread', flush=True)
        return load_file(filename, device=device, backend=backend)

    sys.path.insert(0, str(RUNNER.parent))
    with patch.object(safetensors.torch, 'load_file', load_checkpoint):
        runpy.run_path(str(RUNNER), run_name='__main__')


if __name__ == '__main__':
    main()
