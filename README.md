# Sustainability of AI in Radiology

This repository contains the reproducible modeling framework for estimating the operational energy use and CO2-equivalent emissions of AI inference in radiology.

## Study concept

The model links five components:

1. disease-specific radiology utilization,
2. national disease or procedure volume,
3. modality-specific AI workload,
4. measured model and GPU energy use,
5. geographic electricity carbon intensity.

The intended core calculation is:

```text
estimated examinations = utilization rate x population/procedure volume
AI energy = examinations x AI workload per examination x energy per inference
CO2eq = AI energy x grid carbon intensity
```

The study extends prior sustainability modeling in digital pathology and stroke imaging. The new radiology model will explicitly evaluate model/task complexity, 2D versus 3D workload, GPU hardware, AI adoption, and carbon-aware processing.

## Important benchmark audit

The historical MONAI benchmark scripts execute 5,000 forward-pass iterations for each recorded run. Therefore, energy normalization must use the total number of processed samples:

```text
n_samples = n_iterations x batch_size
energy_per_1000_inferences = total_energy_kWh / n_samples x 1000
```

Some legacy CSV files contain a column named `Kg_carbon(batch=1000)` calculated as `1000 * Kg_carbon / n_batch`. That expression does not include the 5,000 benchmark iterations and must not be used directly for the new analysis. The new pipeline will recalculate normalized energy from raw run totals and benchmark metadata.

This distinction is critical. Population extrapolation based on the legacy normalized carbon column would overestimate per-inference emissions by approximately the number of benchmark iterations if interpreted as emissions for 1,000 individual inferences.

## Planned workflow

- audit and normalize MONAI/GPU benchmark data
- define one inference and one radiology examination computational workload
- estimate disease-specific utilization from TriNetX
- scale to national burden using GBD and/or observed CMS procedure volumes
- characterize available radiology AI tasks using FDA-authorized AI devices
- model adoption scenarios
- convert energy to CO2eq using geographic carbon intensity
- quantify uncertainty and sensitivity
- compare mitigation strategies: efficient model, efficient GPU, and carbon-aware processing

## Repository structure

```text
config/        model assumptions and scenario definitions
data/          local/raw data placeholders and data documentation
notebooks/     exploratory and reporting notebooks
scripts/       command-line analysis entry points
src/           reusable modeling code
tests/         unit tests
results/       generated tables and figures
```

Raw proprietary or large clinical data must not be committed to this public repository.
