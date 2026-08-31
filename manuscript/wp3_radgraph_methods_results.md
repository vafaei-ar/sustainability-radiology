# WP3 prospective VLM benchmark: manuscript-ready Methods and Results

## Methods

### Prospective model comparison

We evaluated MedGemma-4B, Qwen2.5-VL-7B-Instruct, and InternVL3-8B on the same frozen cohort of 100 public, deidentified Open-I Indiana University chest radiograph cases. The cohort was restricted to reports with exactly one linked image and was sampled deterministically with stratification by normal/non-normal metadata and reference-report length quartile. Each model generated one report-style description per case under the frozen BF16 protocol. Model loading was excluded from the measured inference boundary.

Operational energy was measured as direct NVIDIA GPU board energy. Gross board energy per completed case was the primary endpoint. Idle-adjusted net board energy was retained as a secondary sensitivity endpoint. Energy comparisons used ten matched 10-case blocks. We used paired bootstrap confidence intervals with 20,000 resamples and exact two-sided sign tests across matched blocks. Holm correction was applied across the operational pairwise tests.

Automated report fidelity was evaluated against the corresponding Open-I reference report for all 100 cases per model, yielding 300 model-reference pairs. The preferred utility endpoint was F1-RadGraph using the official Stanford-AIMI `radgraph` implementation, package version 0.1.18. We used `F1RadGraph(reward_level="all", model_type="radgraph-xl")` and report the RG_ER component, which scores agreement in extracted radiology entities and relations. Utility comparisons were paired at the case level. We calculated mean paired differences, 95% bootstrap confidence intervals from 20,000 resamples, exact two-sided sign tests, and Holm-adjusted P values within the RadGraph metric family.

As secondary exploratory utility measures, we retained deterministic finding-state agreement across 11 predefined chest-radiograph finding categories, omission and hallucination proxies, unigram F1, and ROUGE-L F1. These secondary metrics were not used as the primary energy-utility endpoint.

We assessed operational efficiency using a two-dimensional Pareto comparison of mean gross GPU board energy per case and mean F1-RadGraph RG_ER. A model was considered Pareto-dominated when another model had both lower mean gross energy and higher mean RadGraph utility. The ratio of gross Wh per unit of RadGraph score was reported descriptively and was not treated as a clinical utility-adjusted cost-effectiveness measure.

No radiologist adjudication was performed. Automated report-fidelity scores should not be interpreted as diagnostic accuracy, clinical acceptability, or patient-level safety. Open-I report text served as the reference standard, and deidentification placeholders, incomplete report descriptions, and restriction to exactly-one-image cases may affect apparent agreement. Runtime comparisons were omitted because the saved Qwen case-level export did not retain elapsed times and the available historical block medians used a known nonstandard even-sample median implementation. Energy endpoints were unaffected by this runtime issue.

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

## Provenance and analysis freeze

Primary source: RunRelay job `K9R4M7Q2`, task `wp3_three_model_100case_analysis`, project commit `fdbfb1943b275c7a0b965944c94ad075e8c3e394`. The completed run reported RadGraph status `ok`, six analysis stages completed, and six declared artifacts delivered to Google Drive. Runtime comparisons remain quarantined.