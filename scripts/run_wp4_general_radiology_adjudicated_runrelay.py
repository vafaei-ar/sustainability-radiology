#!/usr/bin/env python3
"""RunRelay wrapper for adjudicated WP4 TriNetX extraction before GBD scaling."""
from __future__ import annotations

import json
from pathlib import Path

import run_wp4_general_radiology_runrelay as rr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp4" / "general_radiology_adjudicated"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rr.progress("preflight", "Resolving frozen workstation inputs for adjudicated WP4 run")
    if not rr.CONTROL_DIR.is_dir():
        raise FileNotFoundError(f"TriNetX Control directory not found: {rr.CONTROL_DIR}")
    zip_map = rr.find_exact_file(rr.ZIP_NAME)
    analysis_python = rr.resolve_analysis_python()

    resolved = {
        "trinetx_control_dir": str(rr.CONTROL_DIR),
        "zip_map": str(zip_map),
        "zip_map_sha256": rr.sha256_file(zip_map),
        "analysis_python": str(analysis_python),
        "wp4_requirements_sha256": rr.sha256_file(rr.REQUIREMENTS),
        "mapping_profile": "adjudicated_pre_gbd_v1",
        "gbd_scaling_performed": False,
    }
    (OUT / "resolved_inputs.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    rr.progress("adjudicated_extraction", "Running adjudicated mapping and missingness analysis")
    rr.run([
        str(analysis_python),
        "scripts/run_wp4_general_radiology_adjudicated.py",
        "--trinetx-dir", str(rr.CONTROL_DIR),
        "--zip-map", str(zip_map),
        "--output-dir", str(OUT),
    ])

    summary = {
        "status": "WP4_GENERAL_RADIOLOGY_ADJUDICATED_RUNRELAY_OK",
        "mapping_profile": "adjudicated_pre_gbd_v1",
        "breast_mapping": "remove 0633T-0638T for 2018-2019; add mammography/tomosynthesis 77063,77065,77066,77067,G0279",
        "copd_icd9_mapping": "491,492,496",
        "missingness_selection_analysis": True,
        "gbd_scaling_status": "held_pending_adjudicated_qc",
        "patient_level_artifacts": False,
        "resolved_inputs": resolved,
    }
    (OUT / "runrelay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rr.progress("complete", "Adjudicated WP4 extraction complete")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
