from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-wp3"
BOOTSTRAP = ROOT / ".wp3-bootstrap"
REQ = ROOT / "config" / "wp3_vlm_requirements.txt"
OUT = ROOT / "results" / "wp3" / "runtime_setup_report.json"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def working_pip(py: pathlib.Path) -> bool:
    if not py.exists():
        return False
    probe = subprocess.run(
        [str(py), "-m", "pip", "--version"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def create_project_venv() -> str:
    py = VENV / "bin" / "python"
    if working_pip(py):
        return "existing"

    if VENV.exists():
        shutil.rmtree(VENV)

    stdlib = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if stdlib.returncode == 0 and working_pip(py):
        return "stdlib_venv"

    if VENV.exists():
        shutil.rmtree(VENV)

    BOOTSTRAP.mkdir(parents=True, exist_ok=True)
    pip_cmd = [sys.executable, "-m", "pip"]
    pip_probe = subprocess.run(
        pip_cmd + ["--version"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if pip_probe.returncode != 0:
        external_pip = shutil.which("pip3") or shutil.which("pip")
        if not external_pip:
            raise RuntimeError(
                "Cannot create the project runtime: stdlib venv lacks ensurepip and no usable pip command is available."
            )
        pip_cmd = [external_pip]

    run(pip_cmd + ["install", "--target", str(BOOTSTRAP), "virtualenv==20.35.3"])
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(BOOTSTRAP) if not prior else str(BOOTSTRAP) + os.pathsep + prior
    run([sys.executable, "-m", "virtualenv", str(VENV)], env=env)
    if not working_pip(py):
        raise RuntimeError("Project-local virtualenv bootstrap completed but pip is still unavailable.")
    return "project_local_virtualenv_bootstrap"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    creation_method = create_project_venv()
    py = VENV / "bin" / "python"

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([
        str(py), "-m", "pip", "install",
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "torch==2.6.0", "torchvision==0.21.0",
    ])
    run([
        str(py), "-m", "pip", "install",
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "huggingface_hub==0.34.4",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "pillow==11.3.0",
        "qwen-vl-utils==0.0.14",
    ])

    probe = subprocess.run(
        [
            str(py), "-c",
            "import json,torch,transformers,accelerate,huggingface_hub; "
            "print(json.dumps({'python':__import__('sys').version.split()[0],"
            "'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),"
            "'cuda_version':torch.version.cuda,'gpu0':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
            "'transformers':transformers.__version__,'accelerate':accelerate.__version__,"
            "'huggingface_hub':huggingface_hub.__version__}))"
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(probe.stdout.strip())
    report["venv"] = str(VENV)
    report["creation_method"] = creation_method
    report["requirements_file"] = str(REQ.relative_to(ROOT))
    report["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WP3_RUNTIME_SETUP_OK")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
