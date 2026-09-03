#!/usr/bin/env python3
"""Adjudicated WP4 TriNetX extraction before GBD scaling.

This run changes only two source mappings that failed the pre-GBD review:
1. Breast cancer: remove breast CT Category III codes 0633T-0638T because they
   became effective in 2021, after the 2018-2019 analytic period. Add standard
   mammography/tomosynthesis codes used during 2018-2019.
2. COPD ICD-9-CM: restrict the broad 490-496 source range to 491, 492 and 496.

The script writes only aggregate outputs. It also compares imaging utilization
between GBD-compatible and incomplete-demographic disease patient-years to
quantify selection risk before national scaling.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

import run_wp4_general_radiology_clinical_volume as base

# Freeze the adjudicated mappings for the 2018-2019 analysis.
base.DX = {
    **base.DX,
    "COPD": {"ICD-9-CM": ("491", "492", "496"), "ICD-10-CM": ("J44",)},
}
base.CPT = {
    **base.CPT,
    "BC": {
        "Mammography": ("77063", "77065", "77066", "77067", "G0279"),
        "MRI": ("77046", "77047", "77048", "77049"),
        "US": ("76641", "76642"),
    },
}

MAPPING_DECISIONS = {
    "analysis_years": [2018, 2019],
    "breast_cancer": {
        "removed_codes": ["0633T", "0634T", "0635T", "0636T", "0637T", "0638T"],
        "reason": "Breast CT Category III codes 0633T-0638T became effective 2021-01-01 and cannot represent 2018-2019 utilization.",
        "added_mammography_codes": ["77063", "77065", "77066", "77067", "G0279"],
        "counting_rule": "Same patient+disease+year+date+modality is one event, so add-on tomosynthesis does not double-count same-day mammography.",
        "evidence_basis": [
            "CMS Medicare Claims Processing Manual: 77065/77066/77067 used for mammography on/after 2018-01-01; 77063 and G0279 available during the analytic period.",
            "ACR 2021 Breast Imaging FAQ and CMS 2021 coding update: 0633T-0638T effective 2021-01-01.",
        ],
    },
    "copd": {
        "source_icd9_prefixes": ["490", "491", "492", "493", "494", "495", "496"],
        "adjudicated_icd9_prefixes": ["491", "492", "496"],
        "icd10_prefixes": ["J44"],
        "reason": "491 chronic bronchitis, 492 emphysema and 496 chronic airway obstruction align with COPD; 493 asthma, 494 bronchiectasis and 495 extrinsic allergic alveolitis are distinct disease groups, while 490 is nonspecific bronchitis.",
    },
}


def missingness_utilization(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Compare annual selected-imaging rates by GBD compatibility, aggregate only."""
    keys = ["disease", "year", "gbd_compatible"]
    den = (
        cohort.groupby(keys, dropna=False)["patient_id"]
        .nunique()
        .rename("n_patient_years")
        .reset_index()
    )
    mods = base.cpt_frame()[["disease", "modality"]].drop_duplicates()
    grid = den.merge(mods, on="disease", how="inner")
    if events.empty:
        num = pd.DataFrame(columns=keys + ["modality", "n_procedures"])
    else:
        num = (
            events.merge(
                cohort[["patient_id", "disease", "year", "gbd_compatible"]],
                on=["patient_id", "disease", "year"],
                how="inner",
            )
            .groupby(keys + ["modality"], dropna=False)
            .size()
            .rename("n_procedures")
            .reset_index()
        )
    out = grid.merge(num, on=keys + ["modality"], how="left")
    out["n_procedures"] = out["n_procedures"].fillna(0).astype(int)
    out["procedures_per_patient_year"] = out["n_procedures"] / out["n_patient_years"]
    return out.sort_values(["disease", "year", "modality", "gbd_compatible"])


def missingness_comparison(long: pd.DataFrame) -> pd.DataFrame:
    x = long.copy()
    x["compatibility"] = x["gbd_compatible"].map({True: "gbd_compatible", False: "incomplete"})
    p = x.pivot_table(
        index=["disease", "year", "modality"],
        columns="compatibility",
        values=["n_patient_years", "n_procedures", "procedures_per_patient_year"],
        aggfunc="first",
    )
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    for col in [
        "n_patient_years_gbd_compatible", "n_patient_years_incomplete",
        "n_procedures_gbd_compatible", "n_procedures_incomplete",
        "procedures_per_patient_year_gbd_compatible", "procedures_per_patient_year_incomplete",
    ]:
        if col not in p:
            p[col] = 0
    a = p["procedures_per_patient_year_gbd_compatible"].astype(float)
    b = p["procedures_per_patient_year_incomplete"].astype(float)
    p["incomplete_to_compatible_rate_ratio"] = b / a.replace(0, pd.NA)
    p["absolute_rate_difference"] = b - a
    return p


def main() -> None:
    a = base.args()
    o: Path = a.output_dir
    o.mkdir(parents=True, exist_ok=True)
    base.prog(0, 8, "resolve input tables")
    dxfile = base.table(a.trinetx_dir, ["diagnosis.csv"], ["patient_id", "code_system", "code", "principal_diagnosis_indicator", "date"])
    pxfile = base.table(a.trinetx_dir, ["procedure.csv", "procedures.csv"], ["patient_id", "code", "date"])
    pfile = base.table(a.trinetx_dir, ["patient.csv"], ["patient_id", "sex", "year_of_birth", "postal_code"])

    base.prog(1, 8, "extract adjudicated target principal diagnoses")
    dxe = base.read_dx(dxfile, a.chunksize)
    cohort = dxe[["patient_id", "disease", "year"]].drop_duplicates()

    base.prog(2, 8, "attach GBD-compatible strata")
    cs, amb, qc = base.strata(cohort, pfile, a.zip_map)
    amb.to_csv(o / "ambiguous_zip3_prefixes.csv", index=False)
    qc.to_csv(o / "cohort_qc.csv", index=False)

    base.prog(3, 8, "extract adjudicated disease-relevant imaging")
    ev = base.read_px(pxfile, cohort, a.chunksize)

    base.prog(4, 8, "aggregate adjudicated annual utilization")
    annual = base.aggregate(cs, ev, "annual")
    annual.to_csv(o / "trinetx_imaging_utilization_annual_long.csv", index=False)

    base.prog(5, 8, "aggregate adjudicated +/-31-day sensitivity")
    d31 = base.aggregate(cs, base.window31(ev, dxe), "diagnosis31d")
    d31.to_csv(o / "trinetx_imaging_utilization_diagnosis31d_long.csv", index=False)

    base.prog(6, 8, "quantify missing-demographic selection sensitivity")
    miss_long = missingness_utilization(cs, ev)
    miss_long.to_csv(o / "missingness_utilization_long.csv", index=False)
    miss_cmp = missingness_comparison(miss_long)
    miss_cmp.to_csv(o / "missingness_utilization_comparison.csv", index=False)

    base.cpt_frame().to_csv(o / "cpt_disease_modality_map.csv", index=False)
    pd.DataFrame(
        [(d, s, p) for d, v in base.DX.items() for s, ps in v.items() for p in ps],
        columns=["disease", "code_system", "prefix"],
    ).to_csv(o / "disease_diagnosis_prefixes.csv", index=False)

    nat = pd.concat([annual, d31]).groupby(
        ["utilization_window", "disease", "year", "modality"], as_index=False
    ).agg(n_patients=("n_patients", "sum"), n_procedures=("n_procedures", "sum"))
    nat["procedures_per_patient"] = nat.n_procedures / nat.n_patients
    nat.to_csv(o / "trinetx_imaging_utilization_national_window_comparison.csv", index=False)

    overlap = cohort.groupby("patient_id").disease.nunique().value_counts().sort_index()
    ratios = pd.to_numeric(miss_cmp["incomplete_to_compatible_rate_ratio"], errors="coerce").dropna()
    meta = {
        "status": "WP4_GENERAL_RADIOLOGY_ADJUDICATED_OK",
        "years": list(base.YEARS),
        "primary_window": "annual",
        "sensitivity_window": "diagnosis31d",
        "mapping_profile": "adjudicated_pre_gbd_v1",
        "mapping_decisions": MAPPING_DECISIONS,
        "zero_imaging_patient_years_in_denominator": True,
        "multi_target_disease_patients_excluded": False,
        "same_day_deduplication": "patient+disease+year+date+modality",
        "cross_stratum_fallback": False,
        "patient_level_files_written": False,
        "n_disease_patient_years": int(len(cohort)),
        "n_unique_patients": int(cohort.patient_id.nunique()),
        "target_disease_count_per_patient": {str(int(k)): int(v) for k, v in overlap.items()},
        "missingness_sensitivity": {
            "n_cells": int(len(miss_cmp)),
            "median_incomplete_to_compatible_rate_ratio": float(ratios.median()) if len(ratios) else None,
            "min_incomplete_to_compatible_rate_ratio": float(ratios.min()) if len(ratios) else None,
            "max_incomplete_to_compatible_rate_ratio": float(ratios.max()) if len(ratios) else None,
        },
        "inputs": {
            "diagnosis": str(dxfile),
            "procedure": str(pxfile),
            "patient": str(pfile),
            "zip_map": str(a.zip_map),
        },
    }
    (o / "mapping_decisions.json").write_text(json.dumps(MAPPING_DECISIONS, indent=2) + "\n", encoding="utf-8")
    (o / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    base.prog(8, 8, "complete")
    print(json.dumps(meta, indent=2))
    print(f"Wrote adjudicated aggregate outputs to {o}")


if __name__ == "__main__":
    main()
