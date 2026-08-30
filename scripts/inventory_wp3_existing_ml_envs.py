#!/usr/bin/env python3
"""Read-only inventory of existing local Python ML environments for WP3."""
from pathlib import Path
import os, subprocess, json

ROOT = Path('/home/asadr/works')
OUT = Path('results/wp3/existing_ml_env_inventory.txt')
OUT.parent.mkdir(parents=True, exist_ok=True)

candidates = []
for p in ROOT.rglob('python'):
    s = str(p)
    if '/bin/python' not in s:
        continue
    if any(x in s for x in ('/.cache/', '/node_modules/')):
        continue
    candidates.append(p)

# Deduplicate and cap pathological scans.
candidates = sorted(set(candidates))[:200]
lines = [f'ROOT\t{ROOT}', f'PYTHON_CANDIDATES\t{len(candidates)}']
probe = (
    "import json,sys; d={'python':sys.version.split()[0]}; "
    "mods=['torch','transformers','accelerate','bitsandbytes','huggingface_hub']; "
    "import importlib; "
    "\nfor m in mods:\n"
    "  try:\n"
    "    x=importlib.import_module(m); d[m]=getattr(x,'__version__','installed')\n"
    "  except Exception as e: d[m]='MISSING'\n"
    "try:\n"
    " import torch; d['cuda_available']=bool(torch.cuda.is_available()); d['cuda_version']=getattr(torch.version,'cuda',None); d['gpu0']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
    "except Exception: d['cuda_available']=False\n"
    "print(json.dumps(d,sort_keys=True))"
)

for py in candidates:
    try:
        r = subprocess.run([str(py), '-c', probe], text=True, capture_output=True, timeout=20)
        out = (r.stdout or '').strip().replace('\n',' ')
        err = (r.stderr or '').strip().replace('\n',' ')[:300]
        if r.returncode == 0:
            lines.append(f'ENV\t{py}\t{out}')
        else:
            lines.append(f'ENV_ERROR\t{py}\trc={r.returncode}\t{err}')
    except Exception as e:
        lines.append(f'ENV_ERROR\t{py}\t{type(e).__name__}:{e}')

# Report Hugging Face cache presence and top-level model directory names only.
for cache in [Path.home()/'.cache/huggingface/hub', Path('/home/asadr/.cache/huggingface/hub')]:
    if cache.exists():
        models = sorted(p.name for p in cache.iterdir() if p.is_dir() and p.name.startswith('models--'))
        lines.append(f'HF_CACHE\t{cache}\tmodels={len(models)}')
        for name in models[:100]:
            lines.append(f'HF_MODEL\t{name}')

OUT.write_text('\n'.join(lines) + '\n')
print(f'WP3_EXISTING_ML_ENV_INVENTORY environments={len(candidates)}')
