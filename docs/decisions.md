# Methodological decisions

This file records analysis decisions that affect reproducibility and interpretation.

## 2026-08-29: benchmark normalization

Historical MONAI scripts executed 5,000 forward passes per measured run. Each forward pass processed `n_batch` samples. Therefore:

```text
n_samples = 5000 * n_batch
energy_per_inference_kWh = Total_power / n_samples
```

The historical `Kg_carbon(batch=1000)` field is retained only as a legacy audit field and is not used as the normalized inference outcome.

## 2026-08-29: hardware comparisons require architecture matching

The historical benchmark coverage differs across GPUs. In particular, A100 3D segmentation has only 4 runs from 2 architectures, whereas RTX5000 and RTX 6000 Ada contain substantially broader model coverage. Unmatched medians would confound hardware with model mix. Primary GPU comparisons will therefore use matched architectures or model-level paired comparisons.

## 2026-08-29: VLM benchmark denominator unresolved

The source package contains RTX 6000 Ada measurements for PaliGemma, InternVL2, Moondream2, and Florence-2. The benchmark execution script/iteration denominator is not included in the compact source package. These measurements must not be converted to per-inference energy until the original benchmark loop, number of processed samples, prompt/output behavior, and precision are recovered.

## 2026-08-29: carbon-intensity source

EPA eGRID will be the primary open annual US electricity CO2e source. Electricity Maps values used in prior work can be retained as a sensitivity/comparability analysis if licensing and reproducibility permit.

## 2026-08-29: CMS role

CMS Medicare procedure counts will be used as observed external validation and a Medicare-specific analysis, not as direct total-US imaging volume without population adjustment.
