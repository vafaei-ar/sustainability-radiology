#!/usr/bin/env python3
"""Scale frozen WP4 aggregate TriNetX rates with the exact IHME GBD file when available.

This task reads only aggregate adjudicated_pre_gbd_v2 TriNetX outputs plus the exact
GBD prevalence CSV if it exists on the bound workstation. It reports national primary
complete-case scaling, a prespecified missingness-corrected sensitivity, and a national
all-patient-rate benchmark. No patient-level data are read or written.
"""
from __future__ import annotations

import csv
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
REPORT = OUT / "validation_report.md"
GBD_FILENAME = "IHME-GBD_2021_DATA-80d29511-1.csv"
SEARCH_ROOTS = (Path("/home/asadr/works"), Path("/home/asadr/datasets"))
CAUSE_TO_DISEASE = {
    "Breast cancer": "BC",
    "Chronic obstructive pulmonary disease": "COPD",
    "Chronic kidney disease": "CKD",
    "Colon and rectum cancer": "CRC",
    "Ischemic heart disease": "IHD",
}
AGE_GROUPS = {"0-14 years", "15-49 years", "50-74 years", "75+ years"}
SEX_TO_CODE = {"Male": "M", "Female": "F"}
YEARS = {2018, 2019}


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
        "unit": "WP4 GBD scaling stages",
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


def find_exact_gbd() -> list[Path]:
    hits: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            hits.extend(p for p in root.rglob(GBD_FILENAME) if p.is_file())
        except (OSError, PermissionError):
            continue
    return sorted(set(hits))


def add(acc: dict, key: tuple, val: float, lower: float, upper: float) -> None:
    x = acc.setdefault(key, {"val": 0.0, "lower": 0.0, "upper": 0.0})
    x["val"] += val
    x["lower"] += lower
    x["upper"] += upper


def main() -> None:
    progress(0, 5, "Checking frozen aggregate inputs")
    for p in (ANNUAL, META, SUMMARY):
        if not p.is_file():
            raise FileNotFoundError(f"Required aggregate artifact missing: {p.relative_to(ROOT)}")
    meta = json.loads(META.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if meta.get("mapping_profile") != "adjudicated_pre_gbd_v2":
        raise RuntimeError("Expected mapping profile adjudicated_pre_gbd_v2")
    if summary.get("status") != "WP4_FINAL_MAPPING_AND_MISSINGNESS_QC_OK":
        raise RuntimeError("Expected completed final WP4 QC summary")

    progress(1, 5, "Reading frozen state-sex-age utilization rates")
    rates: dict[tuple, float] = {}
    base_strata: set[tuple] = set()
    modalities: dict[str, set[str]] = {}
    states: set[str] = set()
    with ANNUAL.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"disease", "year", "state", "sex", "age_group", "modality", "procedures_per_patient"}
        if not r.fieldnames or not required.issubset(r.fieldnames):
            raise RuntimeError("Annual utilization table has unexpected schema")
        for x in r:
            disease = x["disease"]
            year = int(x["year"])
            state = x["state"]
            sex = x["sex"]
            age = x["age_group"]
            modality = x["modality"]
            key = (disease, year, state, sex, age, modality)
            if key in rates:
                raise RuntimeError(f"Duplicate annual utilization key: {key}")
            rates[key] = float(x["procedures_per_patient"])
            base_strata.add((disease, year, state, sex, age))
            modalities.setdefault(disease, set()).add(modality)
            states.add(state)
    if not rates:
        raise RuntimeError("Annual utilization table is empty")

    progress(2, 5, "Locating exact IHME GBD prevalence file")
    hits = find_exact_gbd()
    gbd_inventory = [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in hits
    ]
    summary["gbd_workstation_inventory"] = {
        "filename": GBD_FILENAME,
        "search_roots": [str(x) for x in SEARCH_ROOTS],
        "match_count": len(hits),
        "matches": gbd_inventory,
    }
    if not hits:
        summary["gbd_scaling"] = {
            "status": "blocked_exact_gbd_file_not_found_on_workstation",
            "patient_level_data_read": False,
        }
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(
            "# WP4 GBD scaling\n\nExact GBD file was not found under the approved workstation search roots. "
            "No scaling was performed.\n",
            encoding="utf-8",
        )
        progress(5, 5, "Exact GBD file not found; scaling held")
        print(json.dumps(summary["gbd_workstation_inventory"], indent=2))
        return
    if len(hits) > 1:
        hashes = {x["sha256"] for x in gbd_inventory}
        if len(hashes) != 1:
            raise RuntimeError("Multiple exact-filename GBD files have different SHA256 values")
    gbd_path = hits[0]

    progress(3, 5, "Joining exact GBD prevalence to available TriNetX strata")
    primary: dict[tuple, dict[str, float]] = {}
    prevalence_total: dict[tuple, dict[str, float]] = {}
    prevalence_matched: dict[tuple, dict[str, float]] = {}
    filtered_rows = 0
    seen_gbd: set[tuple] = set()
    with gbd_path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"measure_name", "location_name", "sex_name", "age_name", "cause_name", "metric_name", "year", "val", "lower", "upper"}
        if not r.fieldnames or not required.issubset(r.fieldnames):
            raise RuntimeError("GBD CSV has unexpected schema")
        for x in r:
            if x["measure_name"] != "Prevalence" or x["metric_name"] != "Number":
                continue
            cause = x["cause_name"]
            if cause not in CAUSE_TO_DISEASE:
                continue
            year = int(x["year"])
            if year not in YEARS or x["age_name"] not in AGE_GROUPS or x["sex_name"] not in SEX_TO_CODE:
                continue
            state = x["location_name"]
            if state not in states:
                continue
            disease = CAUSE_TO_DISEASE[cause]
            sex = SEX_TO_CODE[x["sex_name"]]
            age = x["age_name"]
            gkey = (disease, year, state, sex, age)
            if gkey in seen_gbd:
                raise RuntimeError(f"Duplicate GBD stratum after filtering: {gkey}")
            seen_gbd.add(gkey)
            filtered_rows += 1
            val = float(x["val"]); lower = float(x["lower"]); upper = float(x["upper"])
            add(prevalence_total, (disease, year), val, lower, upper)
            if gkey in base_strata:
                add(prevalence_matched, (disease, year), val, lower, upper)
            for modality in sorted(modalities[disease]):
                rate = rates.get((disease, year, state, sex, age, modality))
                if rate is None:
                    continue
                add(primary, (disease, year, modality), val * rate, lower * rate, upper * rate)

    expected_gbd_rows = len(states) * len(YEARS) * 2 * len(AGE_GROUPS) * len(CAUSE_TO_DISEASE)
    if filtered_rows != expected_gbd_rows:
        raise RuntimeError(f"Expected {expected_gbd_rows} filtered GBD rows, found {filtered_rows}")

    progress(4, 5, "Computing missingness sensitivities and national benchmark")
    miss_cells = summary.get("missingness_selection", {}).get("cells", [])
    miss_map = {(x["disease"], int(x["year"]), x["modality"]): x for x in miss_cells}
    rows_out = []
    for key in sorted(primary):
        disease, year, modality = key
        p = primary[key]
        m = miss_map.get(key)
        if not m:
            raise RuntimeError(f"Missing missingness cell for {key}")
        nc = float(m["n_patient_years_gbd_compatible"])
        ni = float(m["n_patient_years_incomplete"])
        rc = float(m["procedures_per_patient_year_gbd_compatible"])
        ri = float(m["procedures_per_patient_year_incomplete"])
        all_rate = (nc * rc + ni * ri) / (nc + ni)
        factor = all_rate / rc if rc > 0 else 1.0
        pt = prevalence_total[(disease, year)]
        pm = prevalence_matched.get((disease, year), {"val": 0.0, "lower": 0.0, "upper": 0.0})
        coverage = pm["val"] / pt["val"] if pt["val"] > 0 else None
        rows_out.append({
            "disease": disease,
            "year": year,
            "modality": modality,
            "gbd_prevalence_total": pt["val"],
            "gbd_prevalence_matched": pm["val"],
            "gbd_prevalence_coverage": coverage,
            "primary_procedures": p["val"],
            "primary_gbd_lower": p["lower"],
            "primary_gbd_upper": p["upper"],
            "all_patient_rate": all_rate,
            "missingness_correction_factor": factor,
            "sensitivity_a_procedures": p["val"] * factor,
            "sensitivity_a_gbd_lower": p["lower"] * factor,
            "sensitivity_a_gbd_upper": p["upper"] * factor,
            "sensitivity_b_national_procedures": pt["val"] * all_rate,
            "sensitivity_b_gbd_lower": pt["lower"] * all_rate,
            "sensitivity_b_gbd_upper": pt["upper"] * all_rate,
        })

    coverage_rows = []
    for key in sorted(prevalence_total):
        pt = prevalence_total[key]
        pm = prevalence_matched.get(key, {"val": 0.0, "lower": 0.0, "upper": 0.0})
        coverage_rows.append({
            "disease": key[0], "year": key[1],
            "gbd_prevalence_total": pt["val"],
            "gbd_prevalence_matched": pm["val"],
            "gbd_prevalence_coverage": pm["val"] / pt["val"] if pt["val"] else None,
        })

    summary["gbd_scaling"] = {
        "status": "WP4_GBD_SCALING_COMPLETE",
        "source_mapping_profile": meta.get("mapping_profile"),
        "source_annual_sha256": sha256(ANNUAL),
        "gbd_path": str(gbd_path),
        "gbd_sha256": sha256(gbd_path),
        "gbd_filtered_row_count": filtered_rows,
        "gbd_expected_row_count": expected_gbd_rows,
        "method_primary": "GBD prevalence multiplied by exact state-sex-age disease-modality complete-case TriNetX annual rates; strata without an observed TriNetX denominator are excluded, not treated as zero or backfilled.",
        "method_sensitivity_a": "Primary geographically resolved total multiplied within disease-year-modality by the national all-patient-year rate divided by the complete-case rate.",
        "method_sensitivity_b": "National GBD prevalence multiplied by the national all-patient-year TriNetX rate; geographic utilization variation is intentionally removed.",
        "uncertainty_note": "Lower and upper values propagate only the IHME GBD prevalence interval with utilization rates held fixed; they are not full uncertainty intervals.",
        "coverage": coverage_rows,
        "national_by_disease_year_modality": rows_out,
        "patient_level_data_read": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WP4 GBD scaling",
        "",
        f"Status: WP4_GBD_SCALING_COMPLETE",
        f"Mapping profile: {meta.get('mapping_profile')}",
        f"Exact GBD SHA256: {sha256(gbd_path)}",
        f"Filtered GBD strata: {filtered_rows}",
        "",
        "Primary uses exact state-sex-age complete-case TriNetX rates with no cross-stratum fallback. Missing TriNetX strata are excluded rather than assigned zero utilization. Sensitivity A preserves the primary geographic pattern but corrects each disease-year-modality total by the observed all-patient versus complete-case rate ratio. Sensitivity B applies the all-patient national rate to total national GBD prevalence and therefore does not preserve geographic utilization variation.",
        "",
        "## GBD prevalence coverage of observed TriNetX strata",
        "",
        "| Disease | Year | Coverage |",
        "|---|---:|---:|",
    ]
    for x in coverage_rows:
        lines.append(f"| {x['disease']} | {x['year']} | {100*x['gbd_prevalence_coverage']:.1f}% |")
    lines += [
        "",
        "## National procedure volumes",
        "",
        "| Disease | Year | Modality | Primary | Sensitivity A | Sensitivity B |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for x in rows_out:
        lines.append(
            f"| {x['disease']} | {x['year']} | {x['modality']} | {x['primary_procedures']:.0f} | "
            f"{x['sensitivity_a_procedures']:.0f} | {x['sensitivity_b_national_procedures']:.0f} |"
        )
    lines += [
        "",
        "IHME lower/upper prevalence bounds are propagated in summary.json with TriNetX utilization held fixed. These are not full uncertainty intervals.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(5, 5, "WP4 GBD scaling complete")
    print(json.dumps({
        "status": "WP4_GBD_SCALING_COMPLETE",
        "gbd_path": str(gbd_path),
        "gbd_sha256": sha256(gbd_path),
        "filtered_gbd_rows": filtered_rows,
        "n_output_cells": len(rows_out),
        "minimum_prevalence_coverage": min(x["gbd_prevalence_coverage"] for x in coverage_rows),
        "maximum_prevalence_coverage": max(x["gbd_prevalence_coverage"] for x in coverage_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
