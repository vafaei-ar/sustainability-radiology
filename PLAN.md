# Analysis plan

## Primary objective

Estimate the operational energy use and CO2-equivalent emissions of scaling AI inference across radiology in the United States, and quantify how model architecture, GPU hardware, clinical adoption, imaging workload, and electricity carbon intensity change the result.

## Study arms

### Arm 1: task-specific radiology AI

Use measured MONAI classification and segmentation benchmarks as the narrow-model reference.

Factors:
- 2D classification
- 3D classification
- 2D segmentation
- 3D segmentation
- GPU platform
- modality-specific workload
- clinical adoption

### Arm 2: medical VLM/foundation-model AI

Benchmark representative open medical or radiology-oriented VLMs using the same energy accounting framework.

Primary comparisons:
- narrow single-task model
- multi-model narrow workflow
- medical VLM workflow
- optimized medical VLM workflow

Optimization experiments:
- FP16/BF16 reference
- INT8
- INT4
- model-size reduction where suitable
- context-length reduction
- output-token reduction

We will not infer energy savings from memory reduction alone. Energy and wall-clock time must be measured directly when models are runnable.

## Core model

For stratum s, disease d, modality m, AI task k, hardware h, and deployment scenario c:

```text
exams[d,m,s] = utilization[d,m,s] * population[d,s]
AI_exams[d,m,s,c] = exams[d,m,s] * adoption[m,c]
energy[d,m,s,k,h,c] = AI_exams[d,m,s,c] * workload[m,k] * energy_per_unit[k,h]
CO2eq[d,m,s,k,h,c] = energy[d,m,s,k,h,c] * carbon_intensity[s]
```

For VLMs, workload additionally depends on number of images/volumes, visual tokens, text context, and generated output length.

## Data strategy

### Existing study data

- TriNetX: disease-specific imaging utilization
- GBD: disease burden scaling
- historical MONAI/GPU benchmark runs

### Open external data

1. CMS Medicare Physician & Other Practitioners, Geography and Service
   - observed HCPCS procedure counts by geography
   - use for national/procedure-volume validation and sensitivity analyses
   - Medicare FFS population is not equivalent to the full US population and will not be treated as such

2. FDA AI-Enabled Medical Devices
   - characterize the available radiology AI ecosystem by modality, product code, task, and authorization year
   - device counts will not be interpreted as clinical adoption rates

3. EPA eGRID
   - primary open annual electricity CO2e intensity source
   - state and eGRID-subregion rates
   - historical eGRID versions permit temporal sensitivity analyses

4. EIA Open Data
   - optional hourly balancing-authority generation data
   - use for temporal carbon-aware scheduling scenarios if a defensible marginal/average intensity calculation is implemented

5. NCI Imaging Data Commons
   - public DICOM imaging and metadata
   - estimate empirical study dimensions, instance counts, series counts, and modality-specific computational workload distributions

## Validation strategy

- Compare TriNetX/GBD-derived radiology volumes with CMS observed procedure distributions where populations overlap conceptually.
- Validate assumed CT/MR/X-ray workload multipliers against DICOM metadata from public datasets.
- Recalculate all historical benchmark energy values from total measured energy, batch size, and benchmark iterations.
- Do not use legacy `Kg_carbon(batch=1000)` values as normalized inference estimates.

## Uncertainty

Use Monte Carlo propagation rather than confidence intervals on only one component.

Uncertain inputs should include, where available:
- utilization rate
- disease burden/procedure volume
- images/slices/volumes per examination
- AI adoption
- benchmark energy variation
- workload multiplier
- grid carbon intensity

Report medians/means and 95% simulation intervals.

## Sensitivity analyses

Quantify the contribution of:
- AI adoption
- 2D versus 3D workload
- model family
- GPU
- quantization
- VLM context/output length
- geographic carbon intensity

## Scope

Primary scope is operational AI inference. Training, scanner acquisition energy, PACS storage, embodied hardware emissions, building energy, network transmission, and water use are excluded from the primary model unless analyzed in a clearly separate sensitivity analysis.

## Work packages

### WP1. Benchmark audit
- normalize legacy MONAI/GPU results
- identify measured versus simulated rows
- quantify run-to-run variation
- create clean model x task x GPU energy table

### WP2. Imaging workload
- define computational unit for each modality and task
- derive empirical workload distributions from public DICOM metadata

### WP3. Clinical volume
- reconstruct TriNetX utilization pipeline
- map CPT/HCPCS to modality and disease
- integrate GBD
- obtain CMS procedure-volume validation data

### WP4. Grid emissions
- build reproducible EPA eGRID import
- optional EIA temporal analysis

### WP5. VLM benchmark
- select representative runnable open models
- benchmark precision/quantization/context/output scenarios
- separate 2D and 3D-capable models

### WP6. Simulation
- integrate all components
- run adoption, hardware, model, geography, and optimization scenarios
- perform uncertainty and sensitivity analysis

### WP7. Publication outputs
- tables and figures generated from code
- machine-readable processed data
- reproducible environment and exact data-source versions
