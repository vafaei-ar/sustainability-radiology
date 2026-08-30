#!/usr/bin/env python3
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

OUT = Path('results/wp3/runtime_readiness.md')
OUT.parent.mkdir(parents=True, exist_ok=True)

packages = ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'huggingface_hub', 'pillow']
rows = []
for name in packages:
    try:
        version = importlib.metadata.version(name)
        rows.append((name, version))
    except importlib.metadata.PackageNotFoundError:
        rows.append((name, 'NOT_INSTALLED'))


def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return p.returncode, (p.stdout or '').strip(), (p.stderr or '').strip()
    except Exception as exc:
        return -1, '', f'{type(exc).__name__}: {exc}'

nvidia_smi = shutil.which('nvidia-smi')
gpu_lines = []
if nvidia_smi:
    rc, out, err = run([nvidia_smi, '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
    if rc == 0:
        gpu_lines = [line.strip() for line in out.splitlines() if line.strip()]
    else:
        gpu_lines = [f'ERROR: {err or out}']
else:
    gpu_lines = ['nvidia-smi NOT_FOUND']

cuda_available = 'UNKNOWN'
torch_gpu_name = 'UNKNOWN'
torch_cuda_version = 'UNKNOWN'
try:
    import torch
    cuda_available = str(torch.cuda.is_available())
    torch_cuda_version = str(torch.version.cuda)
    if torch.cuda.is_available():
        torch_gpu_name = torch.cuda.get_device_name(0)
except Exception as exc:
    cuda_available = f'ERROR: {type(exc).__name__}: {exc}'

stat = shutil.disk_usage('.')
free_gib = stat.free / (1024 ** 3)

auth_env = {
    'HF_TOKEN_present': bool(os.environ.get('HF_TOKEN')),
    'HUGGING_FACE_HUB_TOKEN_present': bool(os.environ.get('HUGGING_FACE_HUB_TOKEN')),
}

lines = [
    '# WP3 runtime readiness audit',
    '',
    f'- Host: `{platform.node()}`',
    f'- Python: `{platform.python_version()}`',
    f'- Working directory: `{Path.cwd()}`',
    f'- Free disk space: `{free_gib:.1f} GiB`',
    f'- torch.cuda.is_available(): `{cuda_available}`',
    f'- torch CUDA version: `{torch_cuda_version}`',
    f'- torch GPU 0: `{torch_gpu_name}`',
    '',
    '## NVIDIA devices',
]
for line in gpu_lines:
    lines.append(f'- `{line}`')
lines += ['', '## Python packages']
for name, version in rows:
    lines.append(f'- {name}: `{version}`')
lines += ['', '## Authentication visibility']
for key, value in auth_env.items():
    lines.append(f'- {key}: `{value}`')
lines += [
    '',
    '## Readiness interpretation',
    '- This audit does not download models, alter the environment, or expose token values.',
    '- A publication-grade prospective run requires one visible RTX 6000 Ada GPU, functional CUDA/PyTorch, sufficient disk space, compatible Transformers stack, and any model-access authorization required by the selected repositories.',
]

OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'WROTE {OUT}')
