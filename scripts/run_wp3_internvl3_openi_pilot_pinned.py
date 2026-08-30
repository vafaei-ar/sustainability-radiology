from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv-wp3" / "bin" / "python"
PILOT_SCRIPT = ROOT / "scripts" / "run_wp3_internvl3_openi_pilot.py"
MODEL_ID = "OpenGVLab/InternVL3-8B"
MODEL_REVISION = "dab7194eaadae9ff191fef49b961847a18b4c822"
MODEL_CACHE = ROOT / ".wp3-models" / "InternVL3-8B"
EXTRA_REQUIREMENTS = ["einops==0.8.1", "timm==1.0.19"]


def ensure_runtime_dependencies() -> None:
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check", *EXTRA_REQUIREMENTS],
        check=True,
        cwd=ROOT,
    )


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("wp3_internvl3_pilot", PILOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pilot script: {PILOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ensure_runtime_dependencies()
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE),
    )
    pilot = load_pilot_module()
    pilot.MODEL_ID = str(pathlib.Path(snapshot_path).resolve())
    pilot.MODEL_CACHE = MODEL_CACHE
    pilot.main()


if __name__ == "__main__":
    main()
