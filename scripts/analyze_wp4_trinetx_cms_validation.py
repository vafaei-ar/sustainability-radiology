#!/usr/bin/env python3
"""Export a compact aggregate-only WP4 payload for local GBD scaling.

This task reads only the already-produced adjudicated_pre_gbd_v2 aggregate
TriNetX utilization table and the completed final-QC summary. It adds a gzip +
base64 encoded state-by-sex-by-age utilization payload to the already-declared
summary.json artifact. No patient-level data are read or written.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TRI = ROOT / "results" / "wp4" / "general_radiology"
OUT = ROOT / "results" / "wp4" / "trinetx_cms_validation"
ANNUAL = TRI / "trinetx_imaging_utilization_annual_long.csv"
META = TRI / "run_metadata.json"
SUMMARY = OUT / "summary.json"


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "WP4 GBD export stages",
        "updated_at_epoch": time.time(),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, p)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    progress(0, 3, "Checking frozen aggregate inputs")
    for p in (ANNUAL, META, SUMMARY):
        if not p.is_file():
            raise FileNotFoundError(f"Required aggregate artifact missing: {p.relative_to(ROOT)}")
    meta = json.loads(META.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if meta.get("mapping_profile") != "adjudicated_pre_gbd_v2":
        raise RuntimeError("Expected mapping profile adjudicated_pre_gbd_v2")
    if summary.get("status") != "WP4_FINAL_MAPPING_AND_MISSINGNESS_QC_OK":
        raise RuntimeError("Expected completed final WP4 QC summary")

    progress(1, 3, "Encoding compact GBD-compatible stratum rates")
    rows = []
    with ANNUAL.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"disease", "year", "state", "sex", "age_group", "modality", "procedures_per_patient"}
        if not r.fieldnames or not required.issubset(r.fieldnames):
            raise RuntimeError("Annual utilization table has unexpected schema")
        for x in r:
            rows.append([
                x["disease"], int(x["year"]), x["state"], x["sex"], x["age_group"],
                x["modality"], float(x["procedures_per_patient"]),
            ])
    if not rows:
        raise RuntimeError("Annual utilization table is empty")
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    payload = base64.b64encode(packed).decode("ascii")

    progress(2, 3, "Publishing safe compact payload in declared summary")
    summary["gbd_scaling_input"] = {
        "status": "ready",
        "source_mapping_profile": meta.get("mapping_profile"),
        "source_annual_sha256": sha256(ANNUAL),
        "row_count": len(rows),
        "columns": ["disease", "year", "state", "sex", "age_group", "modality", "procedures_per_patient"],
        "encoding": "base64(gzip(json_rows))",
        "payload": payload,
        "patient_level_data_read": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(3, 3, "WP4 compact GBD scaling payload ready")
    print(json.dumps({
        "status": "WP4_GBD_SCALING_INPUT_READY",
        "mapping_profile": meta.get("mapping_profile"),
        "row_count": len(rows),
        "raw_bytes": len(raw),
        "gzip_bytes": len(packed),
        "source_annual_sha256": sha256(ANNUAL),
    }, indent=2))


if __name__ == "__main__":
    main()
