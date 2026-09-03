# Sustainability of AI in Radiology

This repository contains the reproducible modeling framework for estimating the operational energy use and CO2-equivalent emissions of AI inference in radiology.

## Study concept

The model links five components: disease-specific radiology utilization, national disease or procedure volume, modality-specific AI workload, measured model/GPU energy use, and geographic electricity carbon intensity.

```text
estimated examinations = utilization rate x population/procedure volume
AI energy = examinations x AI workload per examination x measured energy per inference
CO2eq = AI energy x grid carbon intensity
```

The study extends prior sustainability modeling in digital pathology and stroke imaging. It explicitly evaluates task/model complexity, 2D versus 3D workload, GPU hardware, VLM size/optimization, AI adoption, and carbon-aware processing.

## Current analysis status

WP1 benchmark normalization, WP3 prospective VLM benchmarking, WP4 clinical-volume reconstruction, and WP5 annual grid-emissions extraction are active.

For WP4, the original `sus_radio.ipynb` logic is recovered. It is provenance, not executable source of truth. The corrected disease-based estimator now uses all eligible disease patient-years, including zero-imaging patient-years. The primary utilization window is the full diagnosis calendar year. The notebook's +/-31-day window is retained as a structural sensitivity. Patients with multiple target diseases remain eligible for each disease cohort. Missing TriNetX strata are not replaced by pooled disease rates.

The detailed WP4 design, corrections, limitations, validation gates, and run commands are in [`docs/wp4_general_radiology.md`](docs/wp4_general_radiology.md). Method decisions are in [`docs/decisions.md`](docs/decisions.md). The complete study plan remains in [`PLAN.md`](PLAN.md).

## Important benchmark audit

Historical MONAI scripts execute 5,000 forward-pass iterations per recorded run. Energy normalization must therefore use the total number of processed samples:

```text
n_samples = n_iterations x batch_size
energy_per_1000_inferences = total_energy / n_samples x 1000
```

The legacy `Kg_carbon(batch=1000)` field is not used as a normalized inference endpoint.

## WP4 execution

Run synthetic validation first:

```bash
python3 scripts/test_wp4_general_radiology_clinical_volume.py
```

Run the aggregate TriNetX extraction on the bound workstation:

```bash
python3 scripts/run_wp4_general_radiology_clinical_volume.py \
  --trinetx-dir /home/asadr/datasets/TriNetX/66350692f55db9228fba3206_20240514_224202103_Control \
  --zip-map /home/asadr/datasets/geodata/zip_code_database.csv \
  --output-dir results/wp4/general_radiology
```

When the original IHME GBD CSV is available on the execution host, scale the aggregate utilization separately:

```bash
python3 scripts/scale_wp4_general_radiology_gbd.py \
  --utilization-dir results/wp4/general_radiology \
  --gbd-file /path/to/IHME-GBD_2021_DATA-80d29511-1.csv \
  --output-dir results/wp4/general_radiology_gbd
```

The GBD file is intentionally not committed to this public repository.

## Planned workflow

- normalize historical MONAI/GPU measurements and preserve architecture-matched comparisons;
- benchmark prospective radiology VLMs under a controlled protocol;
- derive disease-based TriNetX imaging utilization with the corrected WP4 estimator;
- scale disease-based utilization to GBD prevalence with explicit stratum coverage;
- maintain an independent CMS procedure-volume track for external validation;
- derive clinical workload multipliers rather than equating one benchmark tensor with one examination;
- model explicit AI adoption scenarios;
- use EPA eGRID as the primary annual grid-emissions source;
- propagate uncertainty across clinical volume, workload, energy, adoption, and grid intensity;
- compare mitigation through model/hardware efficiency, quantization/workload reduction, and carbon-aware deployment.

Raw proprietary clinical data and patient-level outputs must never be committed to this public repository.
