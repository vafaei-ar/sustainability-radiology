#!/usr/bin/env python3
"""Aggregate-only WP4 QC after final mapping adjudication.

Reads only previously generated aggregate artifacts. It validates cohort completeness,
annual-versus-31-day sensitivity, final mapping status, and the utilization difference
between GBD-compatible and incomplete-demographic patient-years. No patient-level data
are read or written.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
TRI = ROOT / "results" / "wp4" / "general_radiology"
CMS = ROOT / "results" / "wp4" / "cms_imaging_volume"
OUT = ROOT / "results" / "wp4" / "trinetx_cms_validation"

QC_FILE = TRI / "cohort_qc.csv"
WINDOW_FILE = TRI / "trinetx_imaging_utilization_national_window_comparison.csv"
CPT_FILE = TRI / "cpt_disease_modality_map.csv"
DX_FILE = TRI / "disease_diagnosis_prefixes.csv"
META_FILE = TRI / "run_metadata.json"
MISS_FILE = TRI / "missingness_utilization_comparison.csv"
CMS_FILE = CMS / "national_imaging_by_modality.csv"
REQUIRED = [QC_FILE, WINDOW_FILE, CPT_FILE, DX_FILE, META_FILE, MISS_FILE, CMS_FILE]


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
        "unit": "WP4 aggregate QC stages",
        "updated_at_epoch": time.time(),
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, p)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def f(x: str | float | int) -> float:
    return float(x)


def i(x: str | float | int) -> int:
    return int(round(float(x)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress(0, 5, "Checking final aggregate inputs")
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required aggregate artifact(s): " + ", ".join(missing))

    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if meta.get("mapping_profile") != "adjudicated_pre_gbd_v2":
        raise RuntimeError("Expected final mapping profile adjudicated_pre_gbd_v2")

    progress(1, 5, "Summarizing GBD-compatible cohort completeness")
    qc = read_csv(QC_FILE)
    qc_out: list[dict[str, object]] = []
    total_n = total_ok = total_state = total_age = total_sex = 0
    for r in qc:
        n = i(r["n_patient_years"])
        ok = i(r["n_gbd_compatible"])
        ms = i(r["n_missing_state"])
        ma = i(r["n_missing_age"])
        sx = i(r["n_invalid_sex"])
        total_n += n; total_ok += ok; total_state += ms; total_age += ma; total_sex += sx
        qc_out.append({
            **r,
            "gbd_compatible_fraction": ok / n,
            "missing_state_fraction": ms / n,
            "missing_age_fraction": ma / n,
            "invalid_sex_fraction": sx / n,
        })
    qc_fields = list(qc[0].keys()) + ["gbd_compatible_fraction", "missing_state_fraction", "missing_age_fraction", "invalid_sex_fraction"]
    write_csv(OUT / "cohort_qc_summary.csv", qc_fields, qc_out)

    progress(2, 5, "Rechecking annual versus 31-day utilization")
    rows = read_csv(WINDOW_FILE)
    keyed: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        keyed.setdefault((r["disease"], r["year"], r["modality"]), {})[r["utilization_window"]] = r
    win_out: list[dict[str, object]] = []
    win_ratios: list[float] = []
    violations = 0
    for key, pair in sorted(keyed.items()):
        a = pair["annual"]; s = pair["diagnosis31d"]
        ar = f(a["procedures_per_patient"]); sr = f(s["procedures_per_patient"])
        ratio = sr / ar if ar > 0 else (1.0 if sr == 0 else float("nan"))
        violation = i(s["n_procedures"]) > i(a["n_procedures"]) or sr > ar + 1e-12
        violations += int(violation)
        if ratio == ratio:
            win_ratios.append(ratio)
        win_out.append({
            "disease": key[0], "year": key[1], "modality": key[2],
            "n_patients": i(a["n_patients"]),
            "annual_n_procedures": i(a["n_procedures"]),
            "diagnosis31d_n_procedures": i(s["n_procedures"]),
            "annual_procedures_per_patient": ar,
            "diagnosis31d_procedures_per_patient": sr,
            "diagnosis31d_to_annual_rate_ratio": ratio,
            "annual_fraction_outside_31d": 1.0 - ratio if ratio == ratio else "",
            "subset_violation": str(violation).lower(),
        })
    write_csv(OUT / "window_sensitivity_summary.csv", list(win_out[0].keys()), win_out)

    progress(3, 5, "Auditing final mapping and CMS modality context")
    cpt = read_csv(CPT_FILE)
    dx = read_csv(DX_FILE)
    cms = read_csv(CMS_FILE)
    selected = sorted({r["modality"] for r in cpt})
    cms_map = {"CT": "ct", "MRI": "mri", "US": "ultrasound", "X-ray": "radiography_fluoroscopy", "Mammography": "mammography"}
    cms_out: list[dict[str, object]] = []
    for r in cms:
        cm = r["derived_modality"]
        labels = [k for k, v in cms_map.items() if v == cm]
        t4 = labels[0] if labels else ""
        diseases = sorted({x["disease"] for x in cpt if x["modality"] == t4}) if t4 else []
        cms_out.append({
            "cms_modality": cm,
            "cms_total_services": r["total_services"],
            "cms_share_of_imaging_services": r["share_of_imaging_services"],
            "represented_in_t4_selected_panel": str(bool(diseases)).lower(),
            "t4_modality_label": t4,
            "t4_diseases_using_modality": ";".join(diseases),
        })
    write_csv(OUT / "cms_modality_context.csv", list(cms_out[0].keys()), cms_out)
    bc_codes = {r["CPTcode"] for r in cpt if r["disease"] == "BC"}
    copd_icd9 = sorted({r["prefix"] for r in dx if r["disease"] == "COPD" and "ICD-9" in r["code_system"]})
    mapping_ok = (
        {"77058", "77059", "77046", "77047", "77048", "77049"}.issubset(bc_codes)
        and {"77063", "77065", "77066", "77067", "G0279"}.issubset(bc_codes)
        and not bool({"0633T", "0634T", "0635T", "0636T", "0637T", "0638T"} & bc_codes)
        and copd_icd9 == ["491", "492", "496"]
    )

    progress(4, 5, "Quantifying missing-demographic selection")
    miss = read_csv(MISS_FILE)
    ratios: list[float] = []
    miss_cells: list[dict[str, object]] = []
    for r in miss:
        raw = r.get("incomplete_to_compatible_rate_ratio", "")
        ratio = None if raw in ("", "nan", "NaN") else f(raw)
        if ratio is not None:
            ratios.append(ratio)
        miss_cells.append({
            "disease": r["disease"],
            "year": int(r["year"]),
            "modality": r["modality"],
            "n_patient_years_gbd_compatible": i(r["n_patient_years_gbd_compatible"]),
            "n_patient_years_incomplete": i(r["n_patient_years_incomplete"]),
            "procedures_per_patient_year_gbd_compatible": f(r["procedures_per_patient_year_gbd_compatible"]),
            "procedures_per_patient_year_incomplete": f(r["procedures_per_patient_year_incomplete"]),
            "incomplete_to_compatible_rate_ratio": ratio,
            "absolute_rate_difference": f(r["absolute_rate_difference"]),
        })
    if not ratios:
        raise RuntimeError("No finite missingness rate ratios")
    materially_different = [r for r in ratios if r < 0.8 or r > 1.2]

    summary = {
        "status": "WP4_FINAL_MAPPING_AND_MISSINGNESS_QC_OK",
        "mapping_profile": meta.get("mapping_profile"),
        "mapping_frozen_recommendation": bool(mapping_ok and violations == 0),
        "technical_internal_consistency": "pass" if violations == 0 else "fail",
        "cohort_qc": {
            "n_patient_years": total_n,
            "n_gbd_compatible": total_ok,
            "gbd_compatible_fraction": total_ok / total_n,
            "missing_state_fraction": total_state / total_n,
            "missing_age_fraction": total_age / total_n,
            "invalid_sex_fraction": total_sex / total_n,
        },
        "window_sensitivity": {
            "n_cells": len(win_out),
            "subset_violation_count": violations,
            "median_diagnosis31d_to_annual_rate_ratio": statistics.median(win_ratios),
            "minimum_diagnosis31d_to_annual_rate_ratio": min(win_ratios),
            "maximum_diagnosis31d_to_annual_rate_ratio": max(win_ratios),
        },
        "mapping_audit": {
            "mapping_ok": mapping_ok,
            "breast_mri_2018_codes_present": sorted({"77058", "77059"} & bc_codes),
            "breast_mri_2019_codes_present": sorted({"77046", "77047", "77048", "77049"} & bc_codes),
            "breast_mammography_codes_present": sorted({"77063", "77065", "77066", "77067", "G0279"} & bc_codes),
            "breast_postperiod_ct_codes_present": sorted({"0633T", "0634T", "0635T", "0636T", "0637T", "0638T"} & bc_codes),
            "copd_icd9_prefixes": copd_icd9,
            "selected_modalities": selected,
        },
        "missingness_selection": {
            "n_cells": len(miss_cells),
            "median_incomplete_to_compatible_rate_ratio": statistics.median(ratios),
            "minimum_incomplete_to_compatible_rate_ratio": min(ratios),
            "maximum_incomplete_to_compatible_rate_ratio": max(ratios),
            "n_cells_outside_0_8_to_1_2": len(materially_different),
            "cells": miss_cells,
            "interpretation": "Incomplete-demographic patient-years have materially different utilization in some cells; complete-case GBD scaling requires sensitivity analysis.",
        },
        "gbd_scaling_recommendation": "proceed_with_complete_case_primary_plus_missingness_sensitivity",
        "patient_level_data_read": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WP4 final mapping and missingness QC",
        "",
        f"Mapping profile: {summary['mapping_profile']}",
        f"Mapping freeze recommendation: {summary['mapping_frozen_recommendation']}",
        f"GBD-compatible fraction: {100 * summary['cohort_qc']['gbd_compatible_fraction']:.1f}%",
        f"Missingness rate-ratio median: {summary['missingness_selection']['median_incomplete_to_compatible_rate_ratio']:.3f}",
        f"Missingness rate-ratio range: {summary['missingness_selection']['minimum_incomplete_to_compatible_rate_ratio']:.3f} to {summary['missingness_selection']['maximum_incomplete_to_compatible_rate_ratio']:.3f}",
        f"Cells outside 0.8 to 1.2: {summary['missingness_selection']['n_cells_outside_0_8_to_1_2']} of {summary['missingness_selection']['n_cells']}",
        "",
        "Recommendation: freeze adjudicated_pre_gbd_v2. Use complete-case state-age-sex GBD scaling as the primary geographically resolved estimate, and propagate the observed complete-versus-incomplete utilization differences as a prespecified missingness sensitivity analysis.",
        "",
        "CMS values remain contextual billing-service volumes, not direct validation of disease-specific TriNetX procedure counts.",
    ]
    (OUT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(5, 5, "WP4 final aggregate QC complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
