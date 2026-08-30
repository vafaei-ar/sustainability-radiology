# Methodological decisions

This file records analysis decisions that affect reproducibility and interpretation.

## 2026-08-29: benchmark normalization

Historical MONAI scripts executed 5,000 forward passes per measured run. Each forward pass processed `n_batch` samples. Therefore:

```text
n_samples = 5000 * n_batch
tracker_total_power_per_inference = Total_power / n_samples
```

The physical unit of the recovered historical `Total_power` field has not yet been established with publication-grade source evidence. Historical derived endpoints therefore remain unit-neutral and must not be labeled kWh until verified independently.

The historical `Kg_carbon(batch=1000)` field is retained only as a legacy audit field and is not used as the normalized inference outcome.

## 2026-08-29: hardware comparisons require architecture matching

The historical benchmark coverage differs across GPUs. In particular, A100 3D segmentation has only 4 runs from 2 architectures, whereas RTX5000 and RTX 6000 Ada contain substantially broader model coverage. Unmatched medians would confound hardware with model mix. Primary GPU comparisons will therefore use matched architectures or model-level paired comparisons.

## 2026-08-30: historical VLM denominator recovered

The historical VLM summary contains 45 runs across 9 model/configuration groups. Source reconstruction shows that `n_batch = 128` corresponds to 128 sequential completed generation/chat calls for InternVL2, Florence-2, PaliGemma checkpoint code, and the inventoried Moondream script. Historical VLM normalization is therefore:

```text
tracker_total_power_per_case = Total_power / 128
```

The historical VLM pilot remains descriptive because tasks, preprocessing, decoding, device placement, and measured pipeline boundaries were not standardized across model families.

## 2026-08-30: Florence-2 historical runs are CPU evidence

Recovered tracker logs for historical Florence-2 runs show zero NVIDIA attributable power and zero estimated GPU utilization while CPU utilization is substantial. These runs must not be ranked against GPU-executed VLMs as a model-efficiency comparison. They are retained as evidence that device placement can materially affect apparent operational burden.

InternVL2 tracker logs show substantial NVIDIA attributable power and high GPU utilization, supporting classification of those historical runs as GPU-executed measurements.

## 2026-08-30: prospective VLM benchmark is the publication-grade comparison

The primary VLM comparison will use a standardized prospective protocol under fixed hardware, task, image burden, prompts, decoding, and precision. The initial panel is MedGemma-4B, InternVL3-8B, Qwen2.5-VL-7B, and LLaVA-Med v1.5, with an open 3D radiology VLM added if technically feasible.

The baseline precision is BF16. INT8 and INT4 are mitigation analyses within the same model and case set. Primary normalization is energy per completed case request, with secondary endpoints per 1000 cases, per generated token, wall-clock time, throughput, peak VRAM, and GPU utilization. Energy results will be interpreted jointly with clinical-task performance using a performance-energy Pareto framework.

One-time model loading is excluded from the primary tracked interval. Warmup occurs before measurement. Preprocessing, inference, and generation are included in the tracked case-processing interval. Case order is randomized across repeated runs.

## 2026-08-29: carbon-intensity source

EPA eGRID will be the primary open annual US electricity CO2e source. Electricity Maps values used in prior work can be retained as a sensitivity/comparability analysis if licensing and reproducibility permit.

## 2026-08-29: CMS role

CMS Medicare procedure counts will be used as observed external validation and a Medicare-specific analysis, not as direct total-US imaging volume without population adjustment.
