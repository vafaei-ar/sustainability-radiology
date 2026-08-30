# Analysis plan

## Study objective

Estimate the operational energy use and CO2-equivalent emissions of scaling AI inference across radiology in the United States, and determine which deployment choices most strongly change environmental burden while preserving clinical utility.

The study is designed as a reproducible systems-modeling analysis with measured compute benchmarks, empirical imaging workload distributions, observed or modeled clinical volumes, open electricity data, uncertainty propagation, and pre-specified mitigation analyses.

## Primary research questions

1. What is the operational energy and CO2eq burden of task-specific radiology AI at different levels of clinical adoption?
2. How does a multi-model narrow-AI workflow compare with a medical VLM/foundation-model workflow that delivers broader clinical functionality?
3. How much do GPU choice, model size, quantization, workload reduction, and carbon-aware processing reduce energy and CO2eq?
4. Which uncertain inputs dominate the national estimates?
5. Under what conditions can a smaller or quantized medical VLM provide similar clinical utility with substantially lower energy than a larger general VLM or a multi-model workflow?

## Pre-specified hypotheses

H1. Energy per clinical examination will vary substantially across AI task, model family, and hardware even after matching workload.

H2. 3D and multi-image workloads will consume materially more energy per examination than single-image 2D workloads.

H3. Smaller domain-specific VLMs and quantized VLMs will reduce energy per case relative to larger full-precision VLMs, but the magnitude of reduction will depend on hardware and workload.

H4. The ranking of mitigation levers will differ between urgent and deferrable radiology AI. Hardware/model efficiency should dominate for urgent workflows, whereas carbon-aware time/location shifting can additionally reduce emissions for deferrable workflows.

H5. A model's environmental efficiency cannot be inferred reliably from parameter count, memory footprint, or theoretical FLOPs alone; measured energy will remain the primary benchmark endpoint.

## Study arms

### Arm 1: task-specific radiology AI

Use measured MONAI classification and segmentation benchmarks as the narrow-model reference.

Factors:
- 2D classification
- 3D classification
- 2D segmentation
- 3D segmentation
- model architecture
- GPU platform
- modality-specific workload
- clinical adoption

### Arm 2: multi-model narrow-AI workflow

Represent workflows in which multiple specialized models are applied to one examination or episode of care.

The total energy of the workflow is the sum of the measured, workload-adjusted energy of each component model. Scenarios must be clinically justified rather than defined only by arbitrary model counts.

### Arm 3: medical VLM/foundation-model AI

Benchmark representative open medical or radiology-oriented VLMs with the same energy-accounting principles.

Target contemporary model classes:
- small medical VLM
- medium medical VLM
- contemporary general-purpose VLM comparator
- 3D-capable medical/radiology VLM where technically feasible

Historical InternVL2 and Florence-2 measurements are retained as pilot/scaling data only. Publication-grade VLM results will use a new standardized protocol.

### Arm 4: optimized VLM

Repeat selected VLM workloads with optimization strategies where supported:
- BF16/FP16 reference
- INT8
- INT4
- smaller model size or distilled model
- reduced context length
- reduced output length
- optional optimized attention/kernels as a secondary engineering sensitivity analysis

We will not infer energy savings from lower VRAM use alone. Energy and wall-clock time must be measured directly.

## Evidence tiers

### Tier 1: measured historical narrow-model data

MONAI classification and segmentation runs on available GPUs. These require normalization from total run energy using the exact number of benchmark samples.

### Tier 2: historical VLM pilot data

Recovered InternVL2 and Florence-2 experiments on RTX 6000 Ada. These are useful for exploratory scaling because the original scripts and tracker outputs are available. They will not serve as the primary VLM comparison because the image/task protocol was not designed for clinical radiology benchmarking and actual generated-token counts were not systematically stored.

### Tier 3: prospective publication-grade VLM benchmarks

New experiments using public radiology images, fixed prompts, recorded generated-token counts, controlled decoding, explicit quantization settings, exact model revisions, and complete software/hardware metadata.

## Core population model

For stratum s, disease d, modality m, AI task k, hardware h, and deployment scenario c:

```text
exams[d,m,s] = utilization[d,m,s] * population[d,s]
AI_exams[d,m,s,c] = exams[d,m,s] * adoption[m,c]
energy[d,m,s,k,h,c] = AI_exams[d,m,s,c] * workload[m,k] * energy_per_unit[k,h]
CO2eq[d,m,s,k,h,c] = energy[d,m,s,k,h,c] * carbon_intensity[s]
```

For procedure-based analyses, observed procedure counts can replace modeled `utilization * population`.

For VLMs, workload additionally depends on number of images or volumes, visual tokens/patches, text context, generated output length, decoding method, and precision.

## Unit of analysis and benchmark normalization

### Narrow models

Historical MONAI scripts execute 5,000 forward-pass iterations. If each iteration processes `batch_size` samples:

```text
n_samples = 5000 * batch_size
energy_per_sample_kWh = total_run_energy_kWh / n_samples
energy_per_1000_samples_kWh = energy_per_sample_kWh * 1000
```

Legacy columns labeled as carbon per 1,000 batch items will not be treated as normalized inference values unless independently reconstructed from raw measurements.

### VLMs

The primary clinical benchmark unit is one completed case request, including visual encoding, prompt prefill, and autoregressive decoding.

Primary measured endpoints:
- kWh per case
- kWh per 1,000 cases
- joules or Wh per generated token
- runtime per case
- generated tokens per second
- peak VRAM when available

Where technically feasible, prefill and decode energy will be separated.

## Clinical workload conversion

A benchmark tensor is not automatically one clinical examination. Workload multipliers will be derived empirically.

Examples:
- radiograph: images/views per examination
- mammography: views per examination
- CT: images, voxels, volumes, or model patches per examination
- MRI: sequences, volumes, voxels, or model patches per examination
- ultrasound: selected still images or clips/frames depending on modeled AI task
- PET/PET-CT: volume(s) per examination

For patch-based 3D inference, approximate patch counts must account for model input dimensions and overlap rather than assuming one forward pass per study.

## VLM publication benchmark protocol

### Model selection principles

Models should be open-weight or otherwise reproducibly runnable, clinically relevant, and feasible on available hardware. Model selection will be frozen before the final benchmark run and recorded with exact repository/model revision.

The main panel should remain compact enough to permit repeated controlled experiments rather than maximize the number of models.

### Public radiology test set

Use openly accessible radiology data with stable identifiers and licenses. Prefer datasets that allow direct download or reproducible programmatic access and include labels/reports or established benchmark tasks.

Primary 2D task candidates:
- chest radiograph finding identification
- diagnostic question answering
- short findings generation
- impression/report-like generation

A separate 3D-capable experiment may use open CT or MRI data if the model and task are technically mature enough for meaningful evaluation.

### Controlled workload factors

For each selected VLM, record and vary only pre-specified factors:

```text
precision: BF16/FP16, INT8, INT4 where supported
input: single image, multi-image study, optional sampled volume
context: short and longer clinical context
output: short diagnostic output and report-like output
```

### Required benchmark metadata

Every run must record:
- timestamp
- model name and exact revision
- parameter scale when known
- GPU name and count
- GPU memory
- CPU/RAM metadata if tracker supports it
- CUDA/cuDNN versions
- Python, PyTorch, Transformers, Accelerate, quantization-library versions
- precision/quantization method
- prompt template
- decoding settings
- image dimensions and number of images/patches
- input token count
- requested maximum output tokens
- actual generated token count
- batch size
- warm-up iterations
- measured iterations
- runtime
- total measured energy
- component energy where available
- peak VRAM where available

### Repetition and warm-up

Each configuration must include warm-up calls outside the measured interval. Publication runs should use enough repeated cases/runs to characterize variability without unnecessarily consuming energy. The exact repetition count will be set after a pilot variance analysis and then frozen for the main experiment.

### Determinism

Use deterministic or near-deterministic decoding for the core energy comparison. Stochastic decoding can be studied only as a secondary sensitivity analysis.

## Clinical utility / performance

Energy comparisons alone are insufficient when models perform different functions.

Where labels or reference reports permit evaluation, collect clinically relevant performance metrics alongside energy. Depending on task these may include:
- classification AUROC, sensitivity, specificity, F1
- exact/semantic answer accuracy for VQA
- report factuality/clinical error metrics
- RadGraph- or CheXbert-style metrics where appropriate
- expert or established benchmark scores when reproducibly available

Primary interpretation will emphasize the performance-energy Pareto frontier rather than assuming that the lowest-energy model is preferable.

A secondary efficiency metric may be defined as clinical utility per kWh only when the performance metric is comparable across models and tasks.

## Data strategy

### Existing study data

- TriNetX: disease-specific imaging utilization
- GBD: disease burden scaling
- historical MONAI/GPU benchmark runs
- historical recovered multimodal/VLM pilot runs

### Open external data

1. CMS Medicare Physician & Other Practitioners, Geography and Service
   - observed HCPCS procedure counts by geography
   - use for procedure-volume validation and sensitivity analyses
   - Medicare fee-for-service volume is not equivalent to the full US population and will not be treated as such

2. FDA AI-Enabled Medical Devices
   - characterize the available radiology AI ecosystem by modality, product code, task, and authorization year
   - FDA authorization counts will not be interpreted as clinical adoption rates

3. EPA eGRID
   - primary open annual electricity CO2e intensity source
   - use state and/or eGRID-subregion output-emission rates
   - archive exact release/version used

4. EIA Open Data
   - optional temporal grid analysis
   - use only if a defensible average or marginal carbon-intensity method can be implemented and documented

5. NCI Imaging Data Commons
   - public DICOM imaging and metadata
   - estimate empirical study dimensions, instance counts, series counts, and modality-specific computational workload distributions

6. Additional open radiology benchmarks/datasets
   - may be used for VLM performance/energy benchmarking if licenses permit reproducible research use
   - exact version, subset selection, exclusions, and identifiers must be recorded

## Clinical volume strategy

Two complementary volume models will be maintained rather than forcing them into one denominator.

### Disease-based model

```text
TriNetX disease-specific imaging utilization * GBD disease burden
```

Use for disease-specific estimates and demographic/geographic analyses.

### Procedure-based model

Use observed CMS HCPCS service counts for modality/procedure analyses and external validation.

CMS will not be naively scaled to all US residents unless a defensible population-standardization method is explicitly specified.

Agreement/disagreement between the two approaches will be reported rather than hidden.

## Adoption scenarios

Clinical adoption is uncertain and must remain explicit.

Primary generic adoption scenarios:
- 10%
- 25%
- 50%
- 75%
- 100%

Where defensible external adoption estimates exist, they may be added as an empirical scenario but will not replace the full adoption curve.

FDA authorization counts are not adoption estimates.

## Urgency classes

AI use cases will be categorized where clinically defensible as:

### Immediate/latency-sensitive
Examples include acute triage or time-critical findings. Mitigation focuses on model and hardware efficiency; geographic or temporal shifting is constrained.

### Deferrable/non-urgent
Examples may include opportunistic screening, retrospective quantification, selected longitudinal analysis, or batch post-processing. These may additionally support carbon-aware temporal or geographic routing.

This distinction will be scenario-based and not used to claim that every application in a category is safely deferrable.

## Grid-emissions modeling

Primary annual analysis will use an open, versioned electricity-emissions source such as EPA eGRID.

CO2eq will be calculated from measured electricity consumption and geographic grid intensity:

```text
CO2eq_kg = energy_kWh * carbon_intensity_kgCO2e_per_kWh
```

A temporal analysis using EIA data is secondary and should be performed only if the time-resolved carbon-intensity derivation is methodologically defensible.

## Validation strategy

- Compare disease-based TriNetX/GBD radiology volumes with CMS procedure distributions where conceptually comparable.
- Validate assumed CT/MR/X-ray workload multipliers against DICOM metadata from open datasets.
- Recalculate all historical benchmark energy values from total measured energy, batch size, and benchmark iterations.
- Match model architecture when comparing GPUs to avoid confounding hardware with model composition.
- Inspect run-to-run benchmark variability and outliers before aggregating.
- Validate the prospective VLM measurement pipeline on one small model/configuration before scaling to the full model panel.

## Uncertainty

Use Monte Carlo propagation for population estimates rather than attaching confidence intervals to only one component.

Uncertain inputs should include where possible:
- utilization rate
- disease burden/procedure volume
- images/slices/volumes/patches per examination
- AI adoption
- benchmark energy variation
- workload multiplier
- grid carbon intensity

Report point estimates plus 95% simulation intervals.

Structural uncertainty will also be addressed through explicit scenario comparisons rather than being disguised as parametric uncertainty.

## Sensitivity analysis

Quantify the contribution of:
- AI adoption
- 2D versus 3D workload
- modality workload distribution
- model family/size
- GPU
- quantization
- VLM context length
- VLM output length
- clinical volume model
- geographic carbon intensity

Use one-way sensitivity plots plus a global sensitivity method when input distributions permit it.

## Primary outcomes

1. Operational kWh per 1,000 AI-processed radiology examinations/cases.
2. Annual US operational AI electricity consumption under defined deployment scenarios.
3. Annual kg or metric tons CO2eq under defined geographic/grid scenarios.
4. Relative energy and CO2eq reductions from model/hardware/quantization/carbon-aware mitigation.
5. Performance-energy Pareto position for VLM comparisons where common clinical performance metrics are available.

## Secondary outcomes

- runtime and throughput
- peak VRAM
- energy per generated token for VLMs
- per-examination CO2eq
- disease/modality/state contributions
- urgent versus deferrable mitigation potential

## Scope

Primary scope is operational AI inference.

Excluded from the primary estimate:
- model training and pretraining
- scanner acquisition energy
- PACS storage
- embodied hardware emissions
- building construction/maintenance
- network transmission
- water use

These may be discussed or analyzed only in clearly labeled secondary/sensitivity analyses if suitable data become available. The study will not present itself as a full life-cycle assessment.

## Reproducibility requirements

- analysis code in this repository is the source of truth
- exploratory notebooks call reusable functions rather than duplicate calculations
- raw proprietary data are never committed
- every open dataset has a version/date and retrieval script or documented manual download
- every processed table has provenance metadata
- random seeds are fixed and logged
- simulation parameters are config-driven
- generated figures/tables are reproducible from scripts
- dependencies are pinned for the publication release
- final manuscript results should be regenerable from one documented analysis command/workflow after required source data are present

## Work packages and gates

### WP1. Benchmark audit

Tasks:
- normalize legacy MONAI/GPU results
- identify measured versus simulated rows
- quantify run-to-run variation
- match architectures across GPUs
- create clean model x task x GPU energy table

Gate to proceed: normalization rules and model-matching logic pass unit tests and produce auditable tables.

### WP2. Historical VLM pilot reconstruction

Tasks:
- parse InternVL2 and Florence-2 scripts/tracker outputs
- reconstruct exact case counts and run metadata
- calculate energy per case where defensible
- mark fields that cannot be recovered

Gate to proceed: historical values are labeled clearly as pilot results and are not mixed with prospective benchmark estimates.

### WP3. Imaging workload

Tasks:
- define the clinical computational unit for each modality/task
- derive empirical workload distributions from public DICOM metadata
- estimate patch/volume multipliers for 3D models

Gate to proceed: every population-model workload multiplier has documented provenance and uncertainty.

### WP4. Clinical volume

Tasks:
- reconstruct TriNetX utilization pipeline
- map CPT/HCPCS to modality and disease
- integrate GBD
- obtain CMS procedure-volume validation data

Gate to proceed: disease-based and procedure-based estimates are generated independently and validation discrepancies are characterized.

### WP5. Grid emissions

Tasks:
- build reproducible EPA eGRID import
- harmonize geographic units
- optionally implement EIA temporal analysis

Gate to proceed: carbon-intensity units and geographic joins are unit-tested.

### WP6. Prospective VLM benchmark

Tasks:
- freeze compact model panel
- freeze public test set and prompts
- pilot one model/configuration
- determine repetition count from variance
- benchmark full-precision reference
- benchmark INT8/INT4 where supported
- record performance and energy together

Gate to proceed: all primary runs have complete metadata and pass quality-control checks.

### WP7. Integrated simulation

Tasks:
- integrate clinical volume, workload, benchmark energy, adoption, and grid intensity
- run narrow-AI, multi-model, VLM, and optimized-VLM scenarios
- propagate uncertainty
- perform sensitivity analysis

Gate to proceed: scenario definitions are frozen before producing manuscript headline estimates.

### WP8. Publication outputs

Tasks:
- machine-readable processed data
- publication tables and figures generated from code
- methods/data dictionary
- reproducible environment
- versioned release/tag containing the analysis used for the manuscript

## Planned main figures

1. Study workflow and data architecture.
2. Measured energy per 1,000 benchmark units by task/model/GPU.
3. Empirical clinical workload distributions by modality.
4. National radiology AI energy/CO2eq across adoption scenarios.
5. Narrow multi-model workflow versus VLM versus optimized VLM.
6. Energy/performance Pareto frontier for prospective VLMs.
7. Mitigation waterfall or comparative reduction plot for model, GPU, quantization, and carbon-aware deployment.
8. Sensitivity analysis showing dominant drivers of national estimates.

## Planned main tables

1. Data sources, years, populations, variables, and role in the model.
2. Benchmark model/hardware/task characteristics.
3. Prospective VLM protocol and software/hardware configuration.
4. National scenario estimates with uncertainty intervals.
5. Mitigation effects and performance trade-offs.

## Analysis-freeze principle

Exploratory analyses may inform debugging and protocol feasibility. Once WP1-WP6 methods are validated, the primary scenario definitions, model panel, benchmark prompts/tasks, and outcome calculations will be frozen before generating final headline estimates. Any later change will be documented in `docs/decisions.md` with rationale.