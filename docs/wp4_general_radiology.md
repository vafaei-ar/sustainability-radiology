# WP4 general-radiology clinical-volume plan

## Status

The original general-radiology logic is recovered from `sus_radio.ipynb`. The recovery-only task is retained for provenance but is superseded for analysis by the corrected WP4 pipeline.

WP4 has two independent clinical-volume tracks:

1. disease-based volume: TriNetX utilization scaled to GBD disease prevalence;
2. procedure-based volume: observed CMS imaging services used for external validation and a Medicare-specific analysis.

The two tracks will not be forced to agree. Disagreement is a validation result that must be explained before integrated national modeling.

## Disease-based estimand

Target diseases are breast cancer (BC), COPD, chronic kidney disease (CKD), colorectal cancer (CRC), and ischemic heart disease (IHD). The source notebook used principal diagnoses in 2018-2019 and disease-specific CPT definitions from `per_cond.csv`.

The corrected primary TriNetX estimand is annual disease-specific imaging utilization among disease patient-years with a qualifying principal diagnosis in that calendar year:

```text
utilization[disease, year, state, sex, age, modality]
    = unique patient-day-modality imaging events
      / unique disease patient-years
```

Every eligible disease patient-year remains in the denominator, including patients with zero selected imaging. A patient with more than one target disease can contribute independently to each applicable cohort.

## Primary and sensitivity windows

`annual` is the primary window. It counts prespecified disease-relevant imaging in the same calendar year as the qualifying diagnosis. This aligns more directly with annual GBD prevalence scaling.

`diagnosis31d` is the source-compatibility sensitivity. It counts imaging within +/-31 days of a qualifying diagnosis. It preserves the source notebook's episode-window concept while fixing its denominator, year matching, same-day deduplication, and missing-stratum problems.

The difference between these windows is structural uncertainty. It will be reported, not hidden.

## Frozen source definitions and intentional corrections

The CPT mapping in `scripts/run_wp4_general_radiology_clinical_volume.py` is a frozen transcription of the rows selected by the source notebook from `per_cond.csv`. The pipeline writes the exact disease-modality-CPT map used in each run.

Diagnosis definitions preserve the notebook families with one explicit correction: CRC ICD-10 includes `C18`, `C19`, and `C20`, because the source condition is colon/rectal cancer. Exact GBD age groups are 0-14, 15-49, 50-74, and >=75 years.

The source COPD ICD-9 range `490-496` remains provisional. It includes diagnoses beyond strict COPD. COPD results cannot be treated as final until this definition is clinically/coding adjudicated and frozen.

Other corrections:

- do not exclude patients with multiple target diseases;
- include zero-imaging patient-years in denominators;
- estimate 2018 and 2019 separately;
- do not fall back from missing state-sex-age-year cells to pooled disease patients;
- deduplicate same-day procedures to one event per patient, disease, modality, and day;
- treat ambiguous ZIP3-to-state mappings as unavailable rather than assigning a state arbitrarily;
- select GBD US states by numeric location ID, preventing the country Georgia from contaminating the US state Georgia;
- write aggregate artifacts only, with no patient-level rows.

## GBD scaling

`scripts/scale_wp4_general_radiology_gbd.py` uses GBD prevalence (`measure_id=5`) expressed as number (`metric_id=1`) for 2018-2019, stratified by US state, sex, age group, and disease.

For observed TriNetX strata:

```text
estimated procedures = procedures_per_patient x GBD prevalence
```

GBD lower and upper prevalence bounds are scaled separately. Missing TriNetX strata remain missing. No national or pooled fallback rate is substituted. Stratum coverage is an explicit output.

This is not the final uncertainty model. WP7 will propagate utilization, prevalence, workload, benchmark-energy, adoption, and carbon-intensity uncertainty jointly.

## Structural limitation

The TriNetX denominator is a diagnosis-in-year patient-year cohort. GBD represents prevalent disease cases. These are not identical populations. Scaling assumes that annual utilization among TriNetX patients with a recorded principal diagnosis represents utilization among prevalent disease cases in the corresponding stratum.

That assumption is a model component, not an observed fact. WP4 must compare the resulting volume magnitude and modality distribution against the independent CMS procedure-based track. Material disagreement must trigger sensitivity analysis or model revision before WP7 integration.

## RunRelay execution contract

The authoritative named task is:

```text
wp4_general_radiology_clinical_volume
```

It runs `scripts/run_wp4_general_radiology_runrelay.py` at an exact repository commit on the fixed project workstation. The wrapper:

1. verifies the frozen TriNetX Control directory;
2. resolves `zip_code_database.csv` and `IHME-GBD_2021_DATA-80d29511-1.csv` under authorized workstation roots and records SHA256 checksums;
3. runs the synthetic regression suite;
4. runs aggregate TriNetX extraction;
5. runs GBD scaling;
6. writes only the safe aggregate artifacts declared in `.runrelay/project.yaml`.

If multiple non-identical copies of an auxiliary source file are found, the task fails rather than choosing one silently. RunRelay progress is phase-only because a trustworthy total row/chunk denominator is not known before reading the large clinical files. No percentage or ETA is fabricated by the project code.

## Aggregate outputs

TriNetX extraction writes:

```text
results/wp4/general_radiology/
  resolved_inputs.json
  cohort_qc.csv
  ambiguous_zip3_prefixes.csv
  disease_diagnosis_prefixes.csv
  cpt_disease_modality_map.csv
  trinetx_imaging_utilization_annual_long.csv
  trinetx_imaging_utilization_diagnosis31d_long.csv
  trinetx_imaging_utilization_national_window_comparison.csv
  run_metadata.json
  runrelay_summary.json
```

GBD scaling writes:

```text
results/wp4/general_radiology_gbd/
  gbd_us_state_prevalence_2018_2019.csv
  gbd_scaled_imaging_by_stratum.csv
  gbd_scaled_imaging_national.csv
  gbd_strata_missing_trinetx_rates.csv
  gbd_scaling_metadata.json
```

## Validation gates

WP4 disease-based volume is not ready for manuscript integration until:

1. synthetic tests pass;
2. TriNetX extraction completes without patient-level artifacts;
3. cohort QC and missing state/sex/age rates are reviewed;
4. annual versus +/-31-day sensitivity is quantified;
5. COPD ICD-9 definition is adjudicated;
6. GBD stratum coverage is acceptable or missingness is explicitly modeled;
7. disease-based volume is compared with the independent CMS procedure-based results;
8. major disagreement is explained rather than averaged away.

Only after these gates should WP7 combine clinical volume with AI energy and grid carbon intensity.

## Direct debugging commands

The RunRelay task is preferred. The component scripts remain runnable for debugging:

```bash
python3 scripts/test_wp4_general_radiology_clinical_volume.py

python3 scripts/run_wp4_general_radiology_clinical_volume.py \
  --trinetx-dir /home/asadr/datasets/trinetx/66350692f55db9228fba3206_20240514_224202103_Control \
  --zip-map /path/to/zip_code_database.csv \
  --output-dir results/wp4/general_radiology

python3 scripts/scale_wp4_general_radiology_gbd.py \
  --utilization-dir results/wp4/general_radiology \
  --gbd-file /path/to/IHME-GBD_2021_DATA-80d29511-1.csv \
  --output-dir results/wp4/general_radiology_gbd
```
