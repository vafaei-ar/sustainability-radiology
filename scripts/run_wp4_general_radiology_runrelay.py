#!/usr/bin/env python3
"""RunRelay entry point for adjudicated WP4 disease-based clinical volume.

The wrapper resolves frozen inputs, ensures a pandas-capable project-local
analysis runtime, runs the adjudicated 2018-2019 TriNetX extraction, and holds
GBD scaling until mapping and missingness QC are reviewed. No patient-level
artifact is written.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = Path("/home/asadr/datasets/trinetx/66350692f55db9228fba3206_20240514_224202103_Control")
UTIL_OUT = ROOT / "results" / "wp4" / "general_radiology"
RUNTIME_DIR = ROOT / ".venv-wp4"
REQUIREMENTS = ROOT / "requirements-wp4.txt"
ZIP_NAME = "zip_code_database.csv"
GBD_NAME = "IHME-GBD_2021_DATA-80d29511-1.csv"
SEARCH_ROOTS = (Path("/home/asadr/works"), Path("/home/asadr/datasets"))
PRUNE_NAMES = {
    ".git", ".cache", ".conda", ".venv", "venv", "node_modules",
    "site-packages", "__pycache__", CONTROL_DIR.name,
}


def progress(phase: str, message: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "phase": phase,
        "message": message,
        "unit": "WP4 stage",
        "updated_at_epoch": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_file(name: str) -> Path:
    candidates: list[Path] = []
    direct = (
        Path("/home/asadr/works/radio") / name,
        Path("/home/asadr/works") / name,
        Path("/home/asadr/datasets") / name,
        Path("/home/asadr/datasets/geodata") / name,
    )
    candidates.extend(p.resolve() for p in direct if p.is_file())
    if not candidates:
        for root in SEARCH_ROOTS:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in PRUNE_NAMES]
                if name in filenames:
                    candidates.append((Path(dirpath) / name).resolve())
    unique = sorted(set(candidates))
    if not unique:
        raise FileNotFoundError(f"Required input not found under authorized roots: {name}")
    if len(unique) == 1:
        return unique[0]
    hashes = {p: sha256_file(p) for p in unique}
    if len(set(hashes.values())) == 1:
        return unique[0]
    raise RuntimeError(
        "Ambiguous non-identical input files for " + name + ": " + ", ".join(str(p) for p in unique)
    )


def find_optional_file(name: str) -> Path | None:
    try:
        return find_exact_file(name)
    except FileNotFoundError:
        return None


def python_has_pandas(python: Path) -> bool:
    if not python.is_file():
        return False
    try:
        result = subprocess.run(
            [str(python), "-c", "import pandas as pd; print(pd.__version__)"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode == 0:
        print(f"WP4 runtime candidate OK: {python} pandas={result.stdout.strip()}", flush=True)
        return True
    return False


def bootstrap_wp4_runtime() -> Path:
    progress("runtime_setup", "Creating project-local WP4 Python runtime with pinned pandas")
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"WP4 requirements file not found: {REQUIREMENTS}")
    runtime_python = RUNTIME_DIR / "bin" / "python"
    if not runtime_python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(RUNTIME_DIR)], cwd=ROOT, check=True)
    subprocess.run(
        [
            str(runtime_python), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "-r", str(REQUIREMENTS),
        ],
        cwd=ROOT,
        check=True,
    )
    if not python_has_pandas(runtime_python):
        raise RuntimeError("Project-local WP4 runtime was created but pandas import still fails")
    return runtime_python


def resolve_analysis_python() -> Path:
    candidates = [
        RUNTIME_DIR / "bin" / "python",
        ROOT / ".venv-wp3" / "bin" / "python",
        Path("/home/asadr/miniconda3/bin/python"),
        Path("/home/asadr/anaconda3/bin/python"),
        Path("/home/asadr/miniforge3/bin/python"),
        Path("/home/asadr/mambaforge/bin/python"),
        Path(sys.executable),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if python_has_pandas(candidate):
            return candidate
    return bootstrap_wp4_runtime()


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    UTIL_OUT.mkdir(parents=True, exist_ok=True)

    progress("preflight", "Resolving frozen workstation inputs")
    if not CONTROL_DIR.is_dir():
        raise FileNotFoundError(f"TriNetX Control directory not found: {CONTROL_DIR}")
    zip_map = find_exact_file(ZIP_NAME)
    gbd_file = find_optional_file(GBD_NAME)
    analysis_python = resolve_analysis_python()

    resolved = {
        "trinetx_control_dir": str(CONTROL_DIR),
        "zip_map": str(zip_map),
        "zip_map_sha256": sha256_file(zip_map),
        "analysis_python": str(analysis_python),
        "wp4_requirements_sha256": sha256_file(REQUIREMENTS),
        "mapping_profile": "adjudicated_pre_gbd_v2",
        "gbd_file": str(gbd_file) if gbd_file else None,
        "gbd_file_sha256": sha256_file(gbd_file) if gbd_file else None,
        "gbd_input_status": "available" if gbd_file else "not_found_on_execution_host",
        "gbd_expected_filename": GBD_NAME,
    }
    (UTIL_OUT / "resolved_inputs.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    progress("trinetx_extraction", "Running adjudicated WP4 mapping with year-aware breast MRI and missingness analysis")
    run([
        str(analysis_python),
        "scripts/run_wp4_general_radiology_adjudicated.py",
        "--trinetx-dir", str(CONTROL_DIR),
        "--zip-map", str(zip_map),
        "--output-dir", str(UTIL_OUT),
    ])

    run_meta = json.loads((UTIL_OUT / "run_metadata.json").read_text(encoding="utf-8"))
    summary = {
        "status": "WP4_GENERAL_RADIOLOGY_ADJUDICATED_TRINETX_OK_GBD_HELD",
        "primary_estimand": "annual disease patient-year imaging utilization",
        "sensitivity_estimand": "imaging within +/-31 days of qualifying diagnosis",
        "mapping_profile": "adjudicated_pre_gbd_v2",
        "breast_mapping": "mammography/tomosynthesis 77063,77065,77066,77067,G0279; breast MRI 77058/77059 in 2018 and 77046-77049 in 2019; 0633T-0638T excluded",
        "copd_icd9_mapping": "491,492,496",
        "missingness_selection_analysis": run_meta.get("missingness_sensitivity"),
        "zero_imaging_patient_years_in_denominator": True,
        "multi_target_disease_patients_retained": True,
        "cross_stratum_fallback": False,
        "patient_level_artifacts": False,
        "gbd_scaling_status": "held_pending_adjudicated_qc",
        "analysis_python": str(analysis_python),
        "resolved_inputs": resolved,
    }
    (UTIL_OUT / "runrelay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress("complete", "Adjudicated WP4 TriNetX run complete; GBD scaling held for QC")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
