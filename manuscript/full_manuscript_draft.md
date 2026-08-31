# Operational energy and carbon implications of radiology AI: a reproducible systems-modeling study

## Abstract

### Background
Artificial intelligence is increasingly incorporated into radiology workflows, but operational energy use may vary substantially across model families, hardware, clinical workload, and deployment strategy. A radiology sustainability analysis therefore requires measured inference energy, clinically meaningful workload units, deployment volume, and electricity carbon intensity rather than relying on model size or theoretical compute alone.

### Methods
We developed a reproducible systems-modeling framework linking radiology utilization or procedure volume, modality-specific AI workload, measured inference energy, adoption scenarios, and geographic electricity carbon intensity. Historical task-specific MONAI benchmarks were audited separately from prospective vision-language-model (VLM) measurements because the historical tracker energy unit remains incompletely resolved. For the prospective benchmark, MedGemma-4B, Qwen2.5-VL-7B-Instruct, and InternVL3-8B were evaluated on the same frozen 100-case public Open-I Indiana University single-image chest-radiograph cohort. Gross NVIDIA GPU board energy per completed case was the primary operational endpoint. Automated report fidelity was measured using F1-RadGraph RG_ER against the reference report. We also evaluated MedGemma quantization and generation-length restriction as mitigation strategies.

### Results
Mean gross GPU board energy was 0.0790 Wh/case for MedGemma, 0.1615 Wh/case for Qwen, and 0.2083 Wh/case for InternVL3. MedGemma used 51.1% less gross energy than Qwen and 62.1% less than InternVL3. Mean F1-RadGraph RG_ER was 0.2204, 0.1723, and 0.0995, respectively, leaving MedGemma non-dominated on the measured energy-fidelity plane. In a MedGemma quantization pilot, INT8 and INT4 reduced sampled VRAM but increased gross board energy relative to BF16 because inference time increased. Reducing the maximum generation length from 128 to 64 tokens produced no measurable energy change because actual output length was unchanged. A 32-token limit reduced gross energy by approximately 24.1%, but F1-RadGraph decreased by 0.0221 and 93% of outputs reached or nearly reached the token ceiling, consistent with truncation.

### Conclusions
Operational energy differed substantially across radiology VLMs even under a matched clinical workload. Lower memory use did not guarantee lower measured energy, and aggressive output-length restriction introduced a measurable energy-fidelity trade-off. These findings support direct energy measurement and joint energy-utility evaluation rather than inference from parameter count, precision, or memory footprint alone. National electricity and CO2-equivalent estimates remain pending completion of the clinical-volume, workload, and grid-emissions layers of the prespecified population model.

## Introduction

Radiology is one of the most active clinical domains for artificial intelligence. Deployment now spans image classification, segmentation, triage, quantification, report generation, and multimodal reasoning. The environmental implications of these systems are difficult to estimate because the computational unit of a benchmark is not necessarily equivalent to one clinical examination. Energy per forward pass must be connected to the number of images, series, volumes, patches, or generated tokens required by the real workflow.

Prior work in digital pathology and stroke management demonstrated that operational emissions can be estimated by combining measured compute energy with clinical workload and electricity carbon intensity. The same logic can be extended to radiology, but several additional sources of heterogeneity are important. Radiology workloads range from single-view radiographs to multi-series CT and MRI examinations. AI deployments may also shift from narrow task-specific models toward larger multimodal foundation or vision-language models. These systems differ not only in model scale but also in visual-token burden, prompt length, decoding length, precision, and runtime.

A central methodological problem is that commonly used engineering proxies do not directly identify the lowest-energy model. Parameter count, peak memory, and nominal precision can be informative, but operational energy is the time integral of hardware power during the complete inference workflow. A method that reduces memory but substantially increases runtime may increase total energy. Similarly, a lower output-token limit may save energy only when it actually shortens generation, and may degrade report fidelity if it truncates clinically relevant content.

We therefore designed a reproducible radiology sustainability framework with two linked levels of analysis. First, we measure and compare operational energy under controlled AI workloads and evaluate energy jointly with a common automated utility metric. Second, we scale measured energy to clinical examinations and national deployment scenarios using radiology volume, workload, adoption, and geographic electricity carbon intensity. The present manuscript draft contains the completed benchmark and mitigation findings. Population-scale electricity and CO2-equivalent estimates will be inserted only after the remaining workload, volume, and grid-emissions work packages pass their prespecified gates.

## Methods

### Study design

The study is a reproducible systems-modeling analysis of operational AI inference in radiology. The core population calculation is:

```text
examinations = radiology utilization x population or observed procedure volume
AI examinations = examinations x AI adoption
AI energy = AI examinations x clinical workload x measured energy per computational unit
CO2eq = AI energy x electricity carbon intensity
```

Disease-based and procedure-based volume models are maintained separately. Disease-based estimates use disease-specific imaging utilization linked to population burden. Procedure-based analyses use observed radiology service counts for validation and sensitivity analyses. These denominators will not be forced into a single estimate when their populations or definitions are not directly comparable.

The primary scope is operational AI inference. Model training, scanner acquisition energy, PACS storage, embodied hardware emissions, building energy, network transmission, and water use are excluded from the primary estimate.

### Historical task-specific benchmark audit

Historical MONAI classification and segmentation experiments are retained as the narrow-model reference. The benchmark scripts executed 5,000 forward-pass iterations. Normalization therefore uses the total number of processed samples:

```text
n_samples = 5000 x batch_size
energy_per_sample = total_run_energy / n_samples
```

Legacy values labeled as carbon per 1,000 batch items are not used as publication endpoints because the historical calculation omitted the 5,000-iteration denominator. Architecture-matched cross-GPU comparisons are retained descriptively, but the physical unit of the historical Experiment-Impact-Tracker `total_power` field has not yet been established sufficiently for direct combination with the prospective GPU-board Wh measurements. Historical and prospective energy series therefore remain separated until this unit issue is resolved.

### Prospective VLM benchmark cohort

We evaluated MedGemma-4B, Qwen2.5-VL-7B-Instruct, and InternVL3-8B on a frozen 100-case public, deidentified Open-I Indiana University chest-radiograph cohort. The cohort was restricted to reports linked to exactly one image. Sampling was deterministic and stratified by normal/non-normal metadata and reference-report length quartile. This cohort provides a controlled single-image workload and should not be interpreted as representative of all chest-radiograph studies, particularly multi-view examinations.

Each model generated a concise report-style description from the same image and frozen prompt under BF16 inference. Model loading was excluded from the measured boundary. Warm-up was performed before measurement.

### Operational energy measurement

Direct NVIDIA GPU board power was sampled during each measured block and integrated over elapsed time. Gross GPU board energy per completed case was the primary operational endpoint. Idle-adjusted net board energy was calculated as a secondary sensitivity endpoint. Ten matched 10-case blocks were available for each model.

Energy comparisons used paired block-level ratios with 20,000 bootstrap resamples. Exact two-sided sign tests were retained as a distribution-free directional check, and multiplicity adjustment was performed within the prespecified endpoint family. Gross board energy was emphasized because it is less dependent than net energy on the idle-baseline correction strategy.

### Automated report fidelity

Automated report fidelity was evaluated against the Open-I reference report for each case. The preferred utility endpoint was F1-RadGraph using the official Stanford-AIMI `radgraph` implementation, version 0.1.18, with `F1RadGraph(reward_level="all", model_type="radgraph-xl")`. The RG_ER component was used because it evaluates agreement in extracted radiology entities and relations.

Pairwise model comparisons were matched by case. We calculated mean paired differences and 95% bootstrap confidence intervals with 20,000 resamples. Secondary exploratory metrics included deterministic finding-state agreement, omission and hallucination proxies, unigram F1, and ROUGE-L F1.

These metrics compare generated text with a reference report. They are not image-grounded diagnostic truth and do not constitute radiologist-adjudicated clinical accuracy, safety, sensitivity, or specificity. Open-I reference reports may also be incomplete.

### Energy-utility analysis

Operational efficiency was summarized in a two-dimensional Pareto comparison using mean gross GPU board energy and mean F1-RadGraph RG_ER. A configuration was considered dominated when another configuration had both lower mean gross energy and higher mean automated fidelity. Energy per unit of RadGraph score was retained only as a descriptive quantity rather than a clinical cost-effectiveness metric.

### Quantization mitigation

A staged MedGemma pilot compared BF16, INT8, and INT4 on the first 10 cases of the frozen cohort, with three repeated 10-case blocks per precision. Gross board Wh/case remained the primary endpoint. Idle-adjusted energy, runtime, sampled VRAM, and lexical similarity were secondary screening measures. Quantized modes were not advanced to the full 100-case benchmark if they failed to reduce gross board energy.

### Generation-length mitigation

MedGemma BF16 was evaluated on the full 100-case cohort with maximum-new-token limits of 128, 64, and 32. Each condition was measured in ten matched 10-case blocks, with token-budget order randomized within block. We recorded gross and idle-adjusted energy, runtime, actual output-token count, near-cap frequency, and F1-RadGraph RG_ER.

Pairwise energy comparisons used block-level bootstrap ratios. Utility comparisons used case-level paired mean differences. Because bootstrap inference on the mean and the exact sign test target different properties of the paired distribution, the bootstrap confidence interval around the mean difference was used as the primary inference for average automated fidelity change. Near-cap frequency was used to identify likely truncation.

### Population scaling and grid emissions

The population model will combine measured energy with modality-specific clinical workload, disease-based or procedure-based radiology volume, explicit adoption scenarios of 10%, 25%, 50%, 75%, and 100%, and geographic electricity carbon intensity. EPA eGRID will serve as the primary annual electricity-emissions source. Carbon emissions will be calculated as:

```text
CO2eq_kg = energy_kWh x carbon_intensity_kgCO2e_per_kWh
```

Population estimates and uncertainty intervals are intentionally not reported in this draft because the imaging-workload, clinical-volume, and geographic grid-emissions gates are not yet complete.

### Uncertainty and sensitivity

The final population model will propagate uncertainty using Monte Carlo simulation. Candidate uncertain inputs include utilization, procedure volume, imaging workload, AI adoption, measured benchmark variation, workload multiplier, and electricity carbon intensity. Structural uncertainty will be represented by separate scenarios rather than absorbed into parametric confidence intervals. One-way and global sensitivity analyses will identify the inputs that dominate national estimates.

## Results

### Prospective VLM operational energy

Mean gross GPU board energy was 0.0790 Wh/case for MedGemma, 0.1615 Wh/case for Qwen, and 0.2083 Wh/case for InternVL3. MedGemma used approximately 51.1% less gross energy than Qwen in the matched-block comparison (ratio 0.4892, 95% CI 0.4682-0.5082) and 62.1% less than InternVL3 (ratio 0.3793, 95% CI 0.3570-0.4013). Qwen used approximately 22.5% less gross energy than InternVL3 (ratio 0.7753, 95% CI 0.7383-0.8060).

Idle-adjusted net energy showed the same ordering. Gross board energy remains the primary operational endpoint.

### Automated report fidelity

Mean F1-RadGraph RG_ER was 0.2204 for MedGemma, 0.1723 for Qwen, and 0.0995 for InternVL3. MedGemma exceeded Qwen by 0.0480 (95% CI 0.0270-0.0691) and InternVL3 by 0.1208 (95% CI 0.0971-0.1445). Qwen exceeded InternVL3 by 0.0728 (95% CI 0.0523-0.0932).

The deterministic finding-state proxy produced the same overall ranking for mean state agreement: 0.7891 for MedGemma, 0.7427 for Qwen, and 0.6664 for InternVL3. Estimated omission-proxy rates were 32%, 36%, and 37%, respectively. Hallucination-proxy rates were 21%, 14%, and 24%, respectively. These measures remain exploratory.

### Energy-utility relationship

MedGemma was not Pareto-dominated on the gross-energy/RadGraph plane. Qwen was dominated by MedGemma because it required more gross energy while achieving lower automated report fidelity. InternVL3 was dominated by both MedGemma and Qwen. Descriptive gross energy per unit of RadGraph score was 0.359 Wh for MedGemma, 0.937 Wh for Qwen, and 2.093 Wh for InternVL3.

### Quantization mitigation

Mean gross energy in the 10-case MedGemma pilot was approximately 0.0770 Wh/case for BF16, 0.2335 Wh/case for INT8, and 0.1051 Wh/case for INT4. Relative to BF16, INT8 increased gross energy by approximately 203% and INT4 increased gross energy by approximately 36.5%.

Peak sampled VRAM decreased from approximately 8,984 MiB in BF16 to 5,664 MiB with INT8 and 4,830 MiB with INT4. Median runtime increased from approximately 2.64 s/case for BF16 to 9.53 s/case for INT8 and 4.01 s/case for INT4. INT4 lowered the idle-adjusted net-energy estimate, but it did not lower the prespecified primary gross endpoint. Neither quantized condition was advanced to the full 100-case benchmark.

### Generation-length mitigation

The 128-token and 64-token settings were effectively equivalent. Mean gross energy was 0.08121 Wh/case at 128 tokens and 0.08128 Wh/case at 64 tokens, with a 64/128 ratio of 1.0008 (95% CI 0.9932-1.0078). Both conditions generated a mean of 41.29 output tokens and had identical mean F1-RadGraph RG_ER of 0.22036. Neither showed near-cap truncation.

The 32-token limit reduced mean gross energy to 0.06164 Wh/case. The 32/128 gross-energy ratio was 0.7589 (95% CI 0.7415-0.7767), corresponding to approximately 24.1% lower gross energy. Median runtime decreased from 2.465 s/case at 128 tokens to 1.884 s/case at 32 tokens.

The energy reduction was accompanied by lower automated report fidelity. Mean F1-RadGraph RG_ER decreased from 0.22036 to 0.19823. The paired mean difference was -0.02213, with a 95% bootstrap CI of -0.03408 to -0.01050. Ninety-three percent of 32-token outputs reached or nearly reached the configured ceiling. The 32-token setting was therefore energy-saving but not fidelity-preserving.

### Population electricity and CO2-equivalent estimates

[RESULTS PENDING: imaging-workload, clinical-volume, EPA eGRID, adoption-scenario, Monte Carlo, and sensitivity outputs will be inserted here after the corresponding prespecified gates are complete. No placeholder number should be interpreted as a result.]

## Discussion

In a matched single-image chest-radiograph workload, operational energy differed substantially among contemporary VLMs. MedGemma consumed about half the gross GPU board energy of Qwen and less than 40% of InternVL3 while also achieving the highest reference-based RadGraph score. This finding shows why environmental efficiency should not be inferred from model category alone. Direct measurement can reveal large operational differences even when models are evaluated under the same input, prompt, precision, and hardware framework.

The mitigation experiments reinforce the same point. Quantization substantially reduced sampled VRAM but did not reduce the primary gross-energy endpoint. INT8 and INT4 both increased runtime, and the longer execution offset the expected memory-side efficiency benefit. This result challenges the assumption that lower precision is automatically a lower-carbon deployment choice. Quantization can remain useful for fitting models into constrained memory, but that engineering advantage is not equivalent to lower operational energy.

Generation-length restriction produced a different pattern. A 64-token cap did not change actual generation behavior because the model already generated approximately 41 tokens on average. Consequently, energy and automated fidelity were unchanged. A 32-token cap reduced runtime and gross energy by about one quarter, but most outputs reached the ceiling and mean RadGraph fidelity fell. This is a genuine energy-utility trade-off rather than a free mitigation. It also illustrates why requested token limits are weak workload descriptors unless actual generated-token counts are recorded.

The prospective benchmark should not be interpreted as a universal ranking of radiology VLMs. It represents one frozen single-image chest-radiograph report-generation workload. The Open-I references are not a radiologist-adjudicated study endpoint, and the automated metrics evaluate candidate-reference agreement rather than image-grounded diagnostic truth. Multi-image radiographs, CT, MRI, ultrasound, PET, and other workflows may change both model ranking and absolute energy use.

The historical task-specific benchmark adds a complementary narrow-AI perspective, but its tracker energy unit remains unresolved. Until that unit is established, the historical `total_power` measurements should not be combined numerically with the prospective Wh measurements. This separation is preferable to converting an ambiguous legacy field into apparently precise population estimates.

The final contribution of the study will be to connect these measured compute results to real radiology workload and deployment volume. A single chest-radiograph request consumes little electricity in absolute terms, but national burden is a function of the number of eligible examinations, images or volumes per examination, model calls per workflow, adoption, and grid intensity. The population model will therefore report explicit adoption scenarios and separate disease-based and procedure-based denominators. It will also quantify structural uncertainty rather than presenting a single national estimate as if the deployment pathway were known.

### Implications for sustainable radiology AI deployment

Three practical principles emerge from the completed benchmark. First, measured energy should be preferred over model size, VRAM, or precision as the primary operational endpoint. Second, mitigation must be evaluated jointly with utility because a lower-energy configuration may simply be doing less clinically relevant work. Third, deployment decisions should distinguish urgent workflows, where model and hardware efficiency are the primary mitigation levers, from deferrable workflows that may additionally permit carbon-aware temporal or geographic processing.

### Limitations

This benchmark has several limitations. The prospective analysis used a single RTX 6000 Ada platform and therefore does not establish hardware-independent rankings. The cohort contained exactly-one-image Open-I cases and is not representative of all radiographic examinations. Automated report fidelity was reference-based and not radiologist-adjudicated. Open-I reports may be incomplete, which can penalize plausible generated findings not present in the reference. The quantization experiment was intentionally a 10-case screening pilot because neither quantized condition reduced the primary endpoint. The 32-token generation condition caused frequent truncation, limiting its clinical interpretability. Runtime comparisons across the three main models were not reported because the saved Qwen case-level export did not retain valid elapsed times. Finally, national electricity and CO2-equivalent estimates are not yet included because the remaining population-model inputs have not completed prespecified validation gates.

## Conclusions

Operational energy use varied markedly across radiology VLMs under the same single-image workload. MedGemma achieved the lowest gross GPU board energy and highest automated report fidelity among the three tested models. Quantization reduced memory use but increased gross energy, while aggressive generation-length restriction lowered energy at the cost of automated report fidelity. These results show that environmental efficiency cannot be inferred reliably from parameter count, precision, memory footprint, or requested token limit alone. Direct measurement and energy-utility analysis should be integrated into radiology AI evaluation. The next stage is to scale these measured endpoints through clinically grounded workload, volume, adoption, and electricity-carbon models to estimate national operational emissions.

## Tables and figures currently available

- Table 1: three-model operational energy and automated report fidelity.
- Table 2: paired operational-energy comparisons.
- Table 3: paired F1-RadGraph comparisons.
- Table 4: MedGemma mitigation summary.
- Figure 1: three-model energy-utility Pareto comparison.
- Figure 2: model-level energy and automated fidelity summary.
- Figure 3: MedGemma generation-length energy-fidelity trade-off.

## Results still required before manuscript freeze

1. Imaging-workload multipliers with provenance and uncertainty for the modeled modalities and AI tasks.
2. Disease-based TriNetX/GBD clinical-volume estimates.
3. Procedure-based CMS validation estimates.
4. EPA eGRID carbon-intensity import and geographic harmonization.
5. National adoption scenarios and annual electricity/CO2-equivalent estimates.
6. Monte Carlo uncertainty intervals and sensitivity analysis.
7. Final integration of narrow-model and VLM workflows only where measurement units and workload mapping are defensible.

## Provenance of completed prospective results

The primary three-model automated-fidelity analysis was completed by RunRelay job `K9R4M7Q2`. The generation-length experiment was measured in `R8M5Q2K7` and reconstructed without repeating model inference by `M7Q2K8R5` at exact project commit `0f427f7c6f5f15d477f4f36a98012934ac9077ca`. The manuscript-ready benchmark source remains `manuscript/wp3_radgraph_methods_results.md`.