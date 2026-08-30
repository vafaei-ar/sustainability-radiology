from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-wp3"
REQ = ROOT / "config" / "wp3_vlm_requirements.txt"
OUT = ROOT / "results" / "wp3" / "runtime_setup_report.json"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not VENV.exists():
        run([sys.executable, "-m", "venv", str(VENV)])

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
    report["requirements_file"] = str(REQ.relative_to(ROOT))
    report["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WP3_RUNTIME_SETUP_OK")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
