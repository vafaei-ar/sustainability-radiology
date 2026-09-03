#!/usr/bin/env python3
"""Aggregate-only cross-validation of WP4 TriNetX utilization and CMS imaging context.

Reads only previously produced aggregate WP4 artifacts. It does not access patient-level
TriNetX data and does not treat CMS billing services as unique examinations or as a
full-US population denominator.
"""
from __future__ import annotations

import csv
import hashlib
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
RUN_META_FILE = TRI / "run_metadata.json"
CMS_MODALITY_FILE = CMS / "national_imaging_by_modality.csv"
CMS_SUMMARY_FILE = CMS / "summary.json"

REQUIRED = [QC_FILE, WINDOW_FILE, CPT_FILE, DX_FILE, RUN_META_FILE, CMS_MODALITY_FILE, CMS_SUMMARY_FILE]
T4_TO_CMS = {
    "CT": "ct",
    "MRI": "mri",
    "US": "ultrasound",
    "X-ray": "radiography_fluoroscopy",
}


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "WP4 aggregate validation stages",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require_columns(path: Path, rows: list[dict[str, str]], cols: set[str]) -> None:
    if not rows:
        raise RuntimeError(f"Required aggregate table is empty: {path}")
    missing = cols - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")


def as_int(value: str) -> int:
    return int(round(float(value)))


def as_float(value: str) -> float:
    return float(value)


def f6(x: float | None) -> str:
    return "" if x is None else f"{x:.6f}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def qc_analysis(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    require_columns(
        QC_FILE,
        rows,
        {"disease", "year", "n_patient_years", "n_gbd_compatible", "n_missing_state", "n_missing_age", "n_invalid_sex"},
    )
    out: list[dict[str, object]] = []
    totals = {k: 0 for k in ["n_patient_years", "n_gbd_compatible", "n_missing_state", "n_missing_age", "n_invalid_sex"]}
    for r in rows:
        vals = {k: as_int(r[k]) for k in totals}
        n = vals["n_patient_years"]
        if n <= 0:
            raise RuntimeError(f"Non-positive patient-year denominator in QC row: {r}")
        for k, v in vals.items():
            if v < 0:
                raise RuntimeError(f"Negative QC count {k}: {r}")
            totals[k] += v
        if vals["n_gbd_compatible"] > n:
            raise RuntimeError(f"GBD-compatible count exceeds denominator: {r}")
        out.append({
            "disease": r["disease"],
            "year": r["year"],
            **vals,
            "gbd_compatible_fraction": f6(vals["n_gbd_compatible"] / n),
            "missing_state_fraction": f6(vals["n_missing_state"] / n),
            "missing_age_fraction": f6(vals["n_missing_age"] / n),
            "invalid_sex_fraction": f6(vals["n_invalid_sex"] / n),
        })
    n = totals["n_patient_years"]
    summary = {
        **totals,
        "gbd_compatible_fraction": totals["n_gbd_compatible"] / n,
        "missing_state_fraction": totals["n_missing_state"] / n,
        "missing_age_fraction": totals["n_missing_age"] / n,
        "invalid_sex_fraction": totals["n_invalid_sex"] / n,
        "minimum_cell_gbd_compatible_fraction": min(float(r["gbd_compatible_fraction"]) for r in out),
    }
    return out, summary


def window_analysis(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    require_columns(
        WINDOW_FILE,
        rows,
        {"utilization_window", "disease", "year", "modality", "n_patients", "n_procedures", "procedures_per_patient"},
    )
    keyed: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        key = (r["disease"], r["year"], r["modality"])
        keyed.setdefault(key, {})[r["utilization_window"]] = r
    out: list[dict[str, object]] = []
    subset_violations: list[dict[str, object]] = []
    ratios: list[float] = []
    for (disease, year, modality), pair in sorted(keyed.items()):
        if "annual" not in pair or "diagnosis31d" not in pair:
            raise RuntimeError(f"Missing annual/diagnosis31d pair for {(disease, year, modality)}")
        a = pair["annual"]
        s = pair["diagnosis31d"]
        an = as_int(a["n_patients"])
        sn = as_int(s["n_patients"])
        ap = as_int(a["n_procedures"])
        sp = as_int(s["n_procedures"])
        ar = as_float(a["procedures_per_patient"])
        sr = as_float(s["procedures_per_patient"])
        if an != sn:
            raise RuntimeError(f"Sensitivity denominator differs from annual denominator for {(disease, year, modality)}: {an} vs {sn}")
        if min(an, ap, sp, ar, sr) < 0:
            raise RuntimeError(f"Negative utilization value for {(disease, year, modality)}")
        violation = sp > ap or sr > ar + 1e-12
        if violation:
            subset_violations.append({"disease": disease, "year": year, "modality": modality})
        ratio = (sr / ar) if ar > 0 else (1.0 if sr == 0 else None)
        if ratio is not None:
            ratios.append(ratio)
        out.append({
            "disease": disease,
            "year": year,
            "modality": modality,
            "n_patients": an,
            "annual_n_procedures": ap,
            "diagnosis31d_n_procedures": sp,
            "annual_procedures_per_patient": f6(ar),
            "diagnosis31d_procedures_per_patient": f6(sr),
            "diagnosis31d_to_annual_rate_ratio": f6(ratio),
            "annual_fraction_outside_31d": f6((1.0 - ratio) if ratio is not None else None),
            "subset_violation": str(violation).lower(),
        })
    summary = {
        "n_disease_year_modality_cells": len(out),
        "subset_violation_count": len(subset_violations),
        "subset_violations": subset_violations,
        "median_diagnosis31d_to_annual_rate_ratio": statistics.median(ratios) if ratios else None,
        "minimum_diagnosis31d_to_annual_rate_ratio": min(ratios) if ratios else None,
        "maximum_diagnosis31d_to_annual_rate_ratio": max(ratios) if ratios else None,
        "cells_with_less_than_half_annual_rate_in_31d": sum(r < 0.5 for r in ratios),
    }
    return out, summary


def mapping_and_cms(cpt_rows: list[dict[str, str]], dx_rows: list[dict[str, str]], cms_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    require_columns(CPT_FILE, cpt_rows, {"disease", "modality", "CPTcode"})
    require_columns(DX_FILE, dx_rows, {"disease", "code_system", "prefix"})
    require_columns(CMS_MODALITY_FILE, cms_rows, {"derived_modality", "total_services", "share_of_imaging_services"})

    modality_diseases: dict[str, set[str]] = {}
    for r in cpt_rows:
        modality_diseases.setdefault(r["modality"], set()).add(r["disease"])
    represented_cms = {T4_TO_CMS[m] for m in modality_diseases if m in T4_TO_CMS}
    reverse = {v: k for k, v in T4_TO_CMS.items()}

    out: list[dict[str, object]] = []
    cms_map: dict[str, dict[str, str]] = {}
    for r in cms_rows:
        cm = r["derived_modality"]
        cms_map[cm] = r
        t4 = reverse.get(cm)
        diseases = sorted(modality_diseases.get(t4 or "", set()))
        out.append({
            "cms_modality": cm,
            "cms_total_services": f"{as_float(r['total_services']):.3f}",
            "cms_share_of_imaging_services": f6(as_float(r["share_of_imaging_services"])),
            "represented_in_t4_selected_panel": str(cm in represented_cms).lower(),
            "t4_modality_label": t4 or "",
            "t4_diseases_using_modality": ";".join(diseases),
        })

    bc_modalities = {r["modality"] for r in cpt_rows if r["disease"] == "BC"}
    bc_codes = {r["CPTcode"] for r in cpt_rows if r["disease"] == "BC"}
    breast_category_iii_codes = sorted(c for c in bc_codes if c in {"0633T", "0634T", "0635T", "0636T", "0637T", "0638T"})
    mammography_cms = cms_map.get("mammography")
    nuclear_cms = cms_map.get("nuclear_medicine_pet")

    copd_icd9 = sorted({r["prefix"] for r in dx_rows if r["disease"] == "COPD" and "ICD-9" in r["code_system"]})

    summary = {
        "t4_modalities": sorted(modality_diseases),
        "represented_cms_modalities": sorted(represented_cms),
        "cms_modalities_not_in_t4_selected_panel": sorted(set(cms_map) - represented_cms),
        "breast_cancer_t4_modalities": sorted(bc_modalities),
        "breast_cancer_has_mammography_modality": "mammography" in {m.lower() for m in bc_modalities},
        "breast_category_iii_ct_codes": breast_category_iii_codes,
        "cms_mammography_services": as_float(mammography_cms["total_services"]) if mammography_cms else None,
        "cms_mammography_share": as_float(mammography_cms["share_of_imaging_services"]) if mammography_cms else None,
        "cms_nuclear_pet_services": as_float(nuclear_cms["total_services"]) if nuclear_cms else None,
        "cms_nuclear_pet_share": as_float(nuclear_cms["share_of_imaging_services"]) if nuclear_cms else None,
        "copd_icd9_prefixes": copd_icd9,
        "copd_icd9_definition_requires_adjudication": copd_icd9 == ["490", "491", "492", "493", "494", "495", "496"],
    }
    return out, summary


def pct(x: float | None) -> str:
    return "NA" if x is None else f"{100*x:.1f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress(0, 5, "Checking aggregate inputs")
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required aggregate WP4 artifacts: " + ", ".join(missing))

    inputs = {str(p.relative_to(ROOT)): sha256_file(p) for p in REQUIRED}
    run_meta = json.loads(RUN_META_FILE.read_text(encoding="utf-8"))
    cms_summary = json.loads(CMS_SUMMARY_FILE.read_text(encoding="utf-8"))

    progress(1, 5, "Summarizing TriNetX cohort QC")
    qc_rows, qc_summary = qc_analysis(read_csv(QC_FILE))
    write_csv(
        OUT / "cohort_qc_summary.csv",
        ["disease", "year", "n_patient_years", "n_gbd_compatible", "n_missing_state", "n_missing_age", "n_invalid_sex", "gbd_compatible_fraction", "missing_state_fraction", "missing_age_fraction", "invalid_sex_fraction"],
        qc_rows,
    )

    progress(2, 5, "Comparing annual and diagnosis-window estimands")
    window_rows, window_summary = window_analysis(read_csv(WINDOW_FILE))
    write_csv(
        OUT / "window_sensitivity_summary.csv",
        ["disease", "year", "modality", "n_patients", "annual_n_procedures", "diagnosis31d_n_procedures", "annual_procedures_per_patient", "diagnosis31d_procedures_per_patient", "diagnosis31d_to_annual_rate_ratio", "annual_fraction_outside_31d", "subset_violation"],
        window_rows,
    )

    progress(3, 5, "Crosswalking selected TriNetX modalities to CMS context")
    cms_context_rows, mapping_summary = mapping_and_cms(read_csv(CPT_FILE), read_csv(DX_FILE), read_csv(CMS_MODALITY_FILE))
    write_csv(
        OUT / "cms_modality_context.csv",
        ["cms_modality", "cms_total_services", "cms_share_of_imaging_services", "represented_in_t4_selected_panel", "t4_modality_label", "t4_diseases_using_modality"],
        cms_context_rows,
    )

    technical_pass = window_summary["subset_violation_count"] == 0
    mapping_review_required = (
        not mapping_summary["breast_cancer_has_mammography_modality"]
        or bool(mapping_summary["breast_category_iii_ct_codes"])
        or bool(mapping_summary["copd_icd9_definition_requires_adjudication"])
    )

    summary = {
        "status": "WP4_TRINETX_CMS_AGGREGATE_VALIDATION_OK",
        "technical_internal_consistency": "pass" if technical_pass else "fail",
        "mapping_review": "required" if mapping_review_required else "no_automatic_flag",
        "gbd_freeze_recommendation": "hold_for_mapping_adjudication" if mapping_review_required else "eligible_for_next_review",
        "input_sha256": inputs,
        "trinetx_run_metadata": {
            "years": run_meta.get("years"),
            "n_disease_patient_years": run_meta.get("n_disease_patient_years"),
            "n_unique_patients": run_meta.get("n_unique_patients"),
            "primary_window": run_meta.get("primary_window"),
            "sensitivity_window": run_meta.get("sensitivity_window"),
        },
        "cohort_qc": qc_summary,
        "window_sensitivity": window_summary,
        "mapping_and_cms_context": mapping_summary,
        "cms_context": {
            "cms_year": cms_summary.get("cms_year"),
            "imaging_total_services": cms_summary.get("imaging_total_services"),
            "population_scope": "Original Medicare fee-for-service Part B national billing services",
            "comparison_rule": "Use CMS only as an external plausibility and coding-coverage context. Do not interpret CMS billing-service totals as unique examinations, full-US imaging volume, or a direct denominator for TriNetX disease-specific rates.",
        },
        "patient_level_data_read": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    mamm_share = mapping_summary.get("cms_mammography_share")
    nuc_share = mapping_summary.get("cms_nuclear_pet_share")
    report = f"""# WP4 TriNetX-CMS aggregate validation

## Scope

This analysis reads only previously generated aggregate WP4 artifacts. The primary TriNetX estimand is annual disease patient-year selected-imaging utilization for 2018-2019. The +/-31-day diagnosis window is a structural sensitivity analysis. CMS 2024 data represent national Original Medicare fee-for-service Part B billing services and are used only for external plausibility and coding-coverage context.

## Cohort QC

- Disease patient-years in aggregate QC: {qc_summary['n_patient_years']:,}.
- GBD-compatible patient-years: {qc_summary['n_gbd_compatible']:,} ({pct(qc_summary['gbd_compatible_fraction'])}).
- Missing state: {qc_summary['n_missing_state']:,} ({pct(qc_summary['missing_state_fraction'])}).
- Missing age: {qc_summary['n_missing_age']:,} ({pct(qc_summary['missing_age_fraction'])}).
- Invalid/non-MF sex for the GBD crosswalk: {qc_summary['n_invalid_sex']:,} ({pct(qc_summary['invalid_sex_fraction'])}).
- Minimum disease-year GBD-compatible fraction: {pct(qc_summary['minimum_cell_gbd_compatible_fraction'])}.

## Annual versus +/-31-day sensitivity

- Disease-year-modality cells compared: {window_summary['n_disease_year_modality_cells']}.
- Subset-consistency violations: {window_summary['subset_violation_count']}.
- Median diagnosis31d/annual rate ratio: {window_summary['median_diagnosis31d_to_annual_rate_ratio']:.3f}.
- Minimum ratio: {window_summary['minimum_diagnosis31d_to_annual_rate_ratio']:.3f}.
- Maximum ratio: {window_summary['maximum_diagnosis31d_to_annual_rate_ratio']:.3f}.
- Cells with diagnosis31d rate below half the annual rate: {window_summary['cells_with_less_than_half_annual_rate_in_31d']}.

The annual estimator should remain primary because it aligns with annual GBD prevalence. A low diagnosis31d/annual ratio is not automatically an error. It quantifies how much selected imaging occurs outside the source notebook's episode-like window.

## CMS external context

CMS imaging billing services in 2024 totaled {float(cms_summary.get('imaging_total_services', 0)):,.0f}. This is not a count of unique examinations and is not a full-US denominator.

The selected TriNetX panel maps to CMS CT, MRI, ultrasound, and radiography/fluoroscopy categories. CMS mammography accounts for {pct(mamm_share)} of imaging billing services, but the current breast-cancer T4 mapping has no mammography modality. CMS nuclear medicine/PET accounts for {pct(nuc_share)}, but the selected T4 panel has no nuclear/PET modality. These differences are acceptable only if the manuscript explicitly defines WP4 as a selected AI-relevant imaging panel rather than total general-radiology utilization.

## Mapping review before GBD freeze

1. Breast cancer: mammography is absent from the selected T4 mapping. The mapping includes category III breast CT codes {', '.join(mapping_summary['breast_category_iii_ct_codes']) or 'none detected'}. Clinical and temporal adjudication is required before scaling.
2. COPD: ICD-9 prefixes are {', '.join(mapping_summary['copd_icd9_prefixes'])}. The broad 490-496 source definition requires adjudication before analysis freeze.
3. Nuclear medicine/PET: absent from the selected T4 panel. State clearly whether this is an intentional scope restriction.

## Decision

Technical internal consistency: **{summary['technical_internal_consistency'].upper()}**. Mapping review: **{summary['mapping_review'].upper()}**. Recommended GBD freeze status: **{summary['gbd_freeze_recommendation']}**.

Do not compare absolute TriNetX procedure counts directly with CMS service totals as a validation ratio because the population, year, modality scope, and billing unit differ.
"""
    (OUT / "validation_report.md").write_text(report, encoding="utf-8")
    progress(5, 5, "WP4 aggregate TriNetX-CMS validation complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
