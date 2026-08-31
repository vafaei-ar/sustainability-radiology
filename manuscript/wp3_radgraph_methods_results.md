# WP3 prospective VLM benchmark: manuscript-ready Methods and Results

## Methods

### Prospective model comparison

We evaluated MedGemma-4B, Qwen2.5-VL-7B-Instruct, and InternVL3-8B on the same frozen cohort of 100 public, deidentified Open-I Indiana University chest radiograph cases. The cohort was restricted to reports with exactly one linked image and was sampled deterministically with stratification by normal/non-normal metadata and reference-report length quartile. Each model generated one report-style description per case under the frozen BF16 protocol. Model loading was excluded from the measured inference boundary.

Operational energy was measured as direct NVIDIA GPU board energy. Gross board energy per completed case was the primary endpoint. Idle-adjusted net board energy was retained as a secondary sensitivity endpoint. Energy comparisons used ten matched 10-case blocks. We used paired bootstrap confidence intervals with 20,000 resamples and exact two-sided sign tests across matched blocks. Holm correction was applied across the operational pairwise tests.

Automated report fidelity was evaluated against the corresponding Open-I reference report for all 100 cases per model, yielding 300 model-reference pairs. The preferred utility endpoint was F1-RadGraph using the official Stanford-AIMI `radgraph` implementation, package version 0.1.18. We used `F1RadGraph(reward_level="all", model_type="radgraph-xl")` and report the RG_ER component, which scores agreement in extracted radiology entities and relations. Utility comparisons were paired at the case level. We calculated mean paired differences, 95% bootstrap confidence intervals from 20,000 resamples, exact two-sided sign tests, and Holm-adjusted P values within the RadGraph metric family.

As secondary exploratory utility measures, we retained deterministic finding-state agreement across 11 predefined chest-radiograph finding categories, omission and hallucination proxies, unigram F1, and ROUGE-L F1. These secondary metrics were not used as the primary energy-utility endpoint.

We assessed operational efficiency using a two-dimensional Pareto comparison of mean gross GPU board energy per case and mean F1-RadGraph RG_ER. A model was considered Pareto-dominated when another model had both lower mean gross energy and higher mean RadGraph utility. The ratio of gross Wh per unit of RadGraph score was reported descriptively and was not treated as a clinical utility-adjusted cost-effectiveness measure.

No radiologist adjudication was performed. Automated report-fidelity scores should not be interpreted as diagnostic accuracy, clinical acceptability, or patient-level safety. Open-I report text served as the reference standard, and deidentification placeholders, incomplete report descriptions, and restriction to exactly-one-image cases may affect apparent agreement. Runtime comparisons were omitted because the saved Qwen case-level export did not retain elapsed times and the available historical block medians used a known nonstandard even-sample median implementation. Energy endpoints were unaffected by this runtime issue.

### Mitigation experiments

We evaluated two operational mitigation strategies for MedGemma-4B. First, a staged 10-case quantization pilot compared BF16, INT8, and INT4 inference across three repeated 10-case blocks per precision. Gross GPU board Wh/case remained the prespecified primary endpoint, with idle-adjusted net energy, runtime, sampled VRAM, and lexical report-similarity metrics as secondary screening measures. Quantized conditions were not advanced to the full 100-case benchmark if gross board energy did not decrease.

Second, we evaluated generation-length restriction on the full frozen 100-case cohort. The same BF16 MedGemma checkpoint and prompt were run with maximum-new-token limits of 128, 64, and 32. Each token budget was evaluated in ten matched 10-case blocks. Budget order was randomized within block to reduce systematic order effects. We recorded gross and idle-adjusted GPU board energy, case runtime, actual generated-token count, and the fraction of cases terminating at or near the configured token ceiling. The primary utility endpoint remained F1-RadGraph RG_ER against the Open-I reference report. Pairwise energy ratios and paired RadGraph mean differences used 20,000 bootstrap resamples. Holm correction was applied separately within the energy and RadGraph endpoint families.

For generation-length mitigation, we interpreted a lower-energy operating point as fidelity-preserving only when automated report fidelity remained materially unchanged and truncation was not common. Because the exact sign test and the bootstrap mean-difference analysis quantify different properties of paired RadGraph differences, both were retained, but inference about average fidelity change was based primarily on the paired mean difference and its bootstrap confidence interval. Near-cap frequency was reported explicitly to identify likely truncation.

## Results

### GPU board energy

Mean gross GPU board energy was 0.0790 Wh/case for MedGemma, 0.1615 Wh/case for Qwen, and 0.2083 Wh/case for InternVL3. MedGemma used approximately 51.1% less gross board energy than Qwen in the matched-block comparison (ratio 0.4892, 95% CI 0.4682-0.5082) and 62.1% less than InternVL3 (ratio 0.3793, 95% CI 0.3570-0.4013). Qwen used approximately 22.5% less gross board energy than InternVL3 (ratio 0.7753, 95% CI 0.7383-0.8060). For each comparison, the numerator model had lower energy in all 10 matched blocks; the Holm-adjusted exact sign-test P value was 0.0117.

Idle-adjusted net-energy comparisons showed the same ordering. MedGemma/Qwen, MedGemma/InternVL3, and Qwen/InternVL3 net-energy ratios were 0.2943 (95% CI 0.2837-0.3051), 0.2444 (95% CI 0.2314-0.2583), and 0.8305 (95% CI 0.7975-0.8606), respectively. Gross board energy remains the primary operational endpoint because it is less dependent on the idle-baseline correction strategy.

### Automated radiology report fidelity

F1-RadGraph RG_ER was highest for MedGemma (mean 0.2204), followed by Qwen (0.1723) and InternVL3 (0.0995). In paired 100-case comparisons, MedGemma exceeded Qwen by 0.0480 (95% CI 0.0270-0.0691; Holm-adjusted P=6.90e-05) and exceeded InternVL3 by 0.1208 (95% CI 0.0971-0.1445; Holm-adjusted P=4.57e-15). Qwen exceeded InternVL3 by 0.0728 (95% CI 0.0523-0.0932; Holm-adjusted P=1.84e-06).

The deterministic finding-state proxy showed the same overall ranking for mean state agreement: 0.7891 for MedGemma, 0.7427 for Qwen, and 0.6664 for InternVL3. Estimated omission-proxy rates were 32%, 36%, and 37%, respectively. Hallucination-proxy rates were 21%, 14%, and 24%, respectively. These proxy endpoints are exploratory and do not replace the RadGraph endpoint or human review.

### Energy-utility relationship

Using mean gross GPU board energy and mean F1-RadGraph RG_ER, MedGemma was not Pareto-dominated. Qwen was dominated by MedGemma because it consumed more gross board energy while achieving a lower RadGraph score. InternVL3 was dominated by both MedGemma and Qwen. Descriptive gross energy per unit of RadGraph score was 0.359 Wh for MedGemma, 0.937 Wh for Qwen, and 2.093 Wh for InternVL3.

These results support a model-dependent trade-off between operational energy and automated report fidelity within this frozen single-image chest-radiograph benchmark. They do not establish clinical superiority because the utility endpoint is automated reference-based report fidelity rather than radiologist-adjudicated diagnostic performance.

### Quantization mitigation

In the 10-case MedGemma pilot, weight-only quantization reduced sampled memory but did not reduce the primary gross GPU board-energy endpoint. Mean gross energy was approximately 0.0770 Wh/case for BF16, 0.2335 Wh/case for INT8, and 0.1051 Wh/case for INT4. Relative to BF16, INT8 increased gross energy by approximately 203%, while INT4 increased gross energy by approximately 36.5%. Peak sampled VRAM decreased from approximately 8,984 MiB in BF16 to 5,664 MiB with INT8 and 4,830 MiB with INT4.

The energy penalty was accompanied by longer inference. Median runtime was approximately 2.64 s/case for BF16, 9.53 s/case for INT8, and 4.01 s/case for INT4. INT4 lowered idle-adjusted net energy to approximately 0.0246 Wh/case compared with 0.0289 Wh/case for BF16, but the prespecified primary gross endpoint increased. INT8 and INT4 therefore did not meet the advancement criterion for a full 100-case mitigation experiment. These results indicate that lower memory use alone does not guarantee lower operational energy when quantized kernels increase execution time.

### Generation-length mitigation

The 128-token and 64-token conditions were operationally and quantitatively equivalent. Mean gross GPU board energy was 0.08121 Wh/case at 128 tokens and 0.08128 Wh/case at 64 tokens. The paired gross-energy ratio for 64 versus 128 tokens was 1.0008 (95% CI 0.9932-1.0078). Both settings generated a mean of 41.29 output tokens, neither showed near-cap truncation, and both achieved mean F1-RadGraph RG_ER of 0.22036. The 64-token cap therefore provided no measurable mitigation benefit.

The 32-token condition reduced mean gross GPU board energy to 0.06164 Wh/case, corresponding to a 32/128 ratio of 0.7589 (95% CI 0.7415-0.7767; Holm-adjusted sign-test P=0.00586), or approximately 24.1% lower gross energy. Median runtime decreased from 2.465 s/case at 128 tokens to 1.884 s/case at 32 tokens. Idle-adjusted net energy fell from 0.02909 to 0.02203 Wh/case.

This energy reduction was accompanied by lower automated report fidelity. Mean F1-RadGraph RG_ER decreased from 0.22036 at 128 tokens to 0.19823 at 32 tokens. The paired mean difference was -0.02213, with a 95% bootstrap CI of -0.03408 to -0.01050. Although the exact paired sign test returned P=1.0, it tests the balance of positive and negative paired differences rather than the average magnitude of those differences. The bootstrap interval therefore provides the more relevant inference for the prespecified mean utility endpoint. In addition, 93% of 32-token outputs reached or nearly reached the token ceiling, strongly indicating truncation. We therefore classify the 32-token setting as an energy-saving but not fidelity-preserving operating point.

On point estimates, the automated Pareto procedure classified 64 tokens as dominated by 128 tokens and retained both 128 and 32 tokens as non-dominated. Because the 64- and 128-token energy estimates were statistically indistinguishable and their outputs and RadGraph scores were identical, we interpret these two conditions as operationally equivalent rather than emphasizing the formal dominance label.

## Suggested table titles and figure legends

**Table 1. Operational energy and automated report fidelity across three radiology vision-language models.** Mean gross NVIDIA GPU board energy per completed case and mean F1-RadGraph RG_ER are reported for the frozen 100-case Open-I single-image cohort. Pareto dominance was defined using lower gross energy and higher RadGraph fidelity. Deterministic finding-state, omission, and hallucination measures are exploratory reference-based proxies.

**Table 2. Paired operational energy comparisons.** Ratios compare matched 10-case block means. Values below 1 indicate lower energy for the numerator model. Confidence intervals were estimated with 20,000 paired bootstrap resamples. P values are Holm-adjusted exact two-sided sign tests across operational pairwise comparisons. Gross GPU board energy is the primary endpoint; idle-adjusted net energy is secondary.

**Table 3. Paired F1-RadGraph comparisons.** Differences are model A minus model B across the same 100 cases. Confidence intervals were estimated with 20,000 paired bootstrap resamples. P values are Holm-adjusted exact two-sided sign tests within the RadGraph metric family. RadGraph is an automated reference-based report-fidelity metric and does not represent radiologist-adjudicated diagnostic accuracy.

**Table 4. MedGemma operational mitigation experiments.** The quantization pilot reports BF16, INT8, and INT4 gross and idle-adjusted GPU board energy, runtime, and sampled VRAM. The generation-length experiment reports the full 100-case 128-, 64-, and 32-token conditions with gross and net energy, actual generated-token burden, near-cap frequency, and F1-RadGraph RG_ER.

**Figure 1. Energy-utility Pareto comparison.** Mean gross GPU board energy per completed case is plotted against mean F1-RadGraph RG_ER. The preferred direction is toward lower energy and higher automated report fidelity. MedGemma was not Pareto-dominated; Qwen was dominated by MedGemma, and InternVL3 was dominated by both MedGemma and Qwen.

**Figure 2. Model-level operational energy and automated report fidelity.** Panel A shows mean gross GPU board energy per completed case. Panel B shows mean F1-RadGraph RG_ER. The figure summarizes the same frozen 100-case Open-I single-image benchmark and should not be interpreted as a clinical-performance comparison in the absence of radiologist adjudication.

**Figure 3. MedGemma generation-length energy-fidelity trade-off.** Mean gross GPU board energy per completed case is plotted against mean F1-RadGraph RG_ER for maximum-new-token limits of 128, 64, and 32. The 64-token condition overlaps the 128-token condition, while the 32-token condition shifts toward lower energy and lower automated report fidelity. Ninety-three percent of 32-token outputs reached or nearly reached the configured token ceiling, supporting truncation as the mechanism of fidelity loss.

## Provenance and analysis freeze

Primary three-model source: RunRelay job `K9R4M7Q2`, task `wp3_three_model_100case_analysis`, project commit `fdbfb1943b275c7a0b965944c94ad075e8c3e394`. The completed run reported RadGraph status `ok`, six analysis stages completed, and six declared artifacts delivered to Google Drive. Runtime comparisons remain quarantined.

Quantization mitigation source: RunRelay job `Z6R3M8Q5`, MedGemma-4B BF16/INT8/INT4 staged 10-case pilot. Quantized conditions were not advanced because neither reduced the primary gross GPU board-energy endpoint.

Generation-length mitigation source: RunRelay measurement job `R8M5Q2K7` completed all 30 energy blocks, 300 case generations, and 300 F1-RadGraph RG_ER scores before a final CSV serialization error. RunRelay recovery job `M7Q2K8R5`, exact project commit `0f427f7c6f5f15d477f4f36a98012934ac9077ca`, validated the saved measurements and reconstructed the final pairwise statistics and summary without repeating model inference or RadGraph scoring.