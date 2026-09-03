#!/usr/bin/env python3
"""RunRelay entry point for corrected WP4 disease-based clinical volume.

The task resolves the legacy ZIP-to-state input, runs synthetic validation, and
extracts aggregate TriNetX utilization. If the original IHME GBD CSV is present
on the bound workstation, the task also performs GBD prevalence scaling. If the
GBD file is absent, TriNetX extraction still completes successfully and the
summary records GBD scaling as pending. No patient-level artifact is written.
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
GBD_OUT = ROOT / "results" / "wp4" / "general_radiology_gbd"
ZIP_NAME = "zip_code_database.csv"
GBD_NAME = "IHME-GBD_2021_DATA-80d29511-1.csv"
SEARCH_ROOTS = (Path("/home/asadr/works"), Path("/home/asadr/datasets"))
PRUNE_NAMES = {
    ".git", ".cache", ".conda", ".venv", "venv", "node_modules",
    "site-packages", "__pycache__", CONTROL_DIR.name,
}


def progress(phase: str, message: str) -> None:
    """Write phase-only trusted progress. No fabricated percentage/ETA."""
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

    resolved = {
        "trinetx_control_dir": str(CONTROL_DIR),
        "zip_map": str(zip_map),
        "zip_map_sha256": sha256_file(zip_map),
        "gbd_file": str(gbd_file) if gbd_file else None,
        "gbd_file_sha256": sha256_file(gbd_file) if gbd_file else None,
        "gbd_input_status": "available" if gbd_file else "not_found_on_execution_host",
        "gbd_expected_filename": GBD_NAME,
    }
    (UTIL_OUT / "resolved_inputs.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    progress("synthetic_validation", "Running corrected WP4 synthetic regression tests")
    run([sys.executable, "scripts/test_wp4_general_radiology_clinical_volume.py"])

    progress("trinetx_extraction", "Extracting aggregate disease-by-modality utilization from TriNetX")
    run([
        sys.executable,
        "scripts/run_wp4_general_radiology_clinical_volume.py",
        "--trinetx-dir", str(CONTROL_DIR),
        "--zip-map", str(zip_map),
        "--output-dir", str(UTIL_OUT),
    ])

    if gbd_file is not None:
        GBD_OUT.mkdir(parents=True, exist_ok=True)
        progress("gbd_scaling", "Scaling observed TriNetX strata to GBD 2021 prevalence")
        run([
            sys.executable,
            "scripts/scale_wp4_general_radiology_gbd.py",
            "--utilization-dir", str(UTIL_OUT),
            "--gbd-file", str(gbd_file),
            "--output-dir", str(GBD_OUT),
        ])
        gbd_scaling_status = "completed"
        overall_status = "WP4_GENERAL_RADIOLOGY_RUNRELAY_OK"
    else:
        gbd_scaling_status = "pending_external_gbd_input"
        overall_status = "WP4_GENERAL_RADIOLOGY_TRINETX_OK_GBD_PENDING"
        print(
            "GBD input was not found on the execution host. "
            "TriNetX aggregate extraction completed; GBD scaling is pending.",
            flush=True,
        )

    summary = {
        "status": overall_status,
        "primary_estimand": "annual disease patient-year imaging utilization",
        "sensitivity_estimand": "imaging within +/-31 days of qualifying diagnosis",
        "zero_imaging_patient_years_in_denominator": True,
        "multi_target_disease_patients_retained": True,
        "cross_stratum_fallback": False,
        "patient_level_artifacts": False,
        "gbd_scaling_status": gbd_scaling_status,
        "copd_icd9_status": "provisional source definition 490-496; adjudication required before analysis freeze",
        "resolved_inputs": resolved,
    }
    (UTIL_OUT / "runrelay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress("complete", "WP4 corrected TriNetX clinical-volume run complete")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
