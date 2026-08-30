"""Audit and normalize historical radiology AI benchmark results.

The legacy MONAI CSV files contain total measured run energy and a historical
`Kg_carbon(batch=1000)` column. The latter is not a valid per-1000-inference
normalization for the benchmark scripts because each measured run executed
5,000 forward-pass iterations. This module always derives normalized energy
from total run energy, batch size, and iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LEGACY_NARROW_REQUIRED = {
    "Task",
    "Architecture",
    "Dimension",
    "itry",
    "n_batch",
    "Total_power",
}


@dataclass(frozen=True)
class LegacyBenchmarkSpec:
    """Metadata needed to normalize a historical benchmark file."""

    gpu: str
    n_iterations: int = 5000
    measured: bool = True


def load_legacy_narrow_csv(
    path: str | Path,
    spec: LegacyBenchmarkSpec,
) -> pd.DataFrame:
    """Load and normalize a legacy MONAI classification/segmentation CSV.

    Returns one row per benchmark run with total samples processed, energy per
    inference, and energy per 1,000 inferences. `Total_power` is interpreted as
    total run electricity use in kWh, matching the Experiment Impact Tracker
    output used by the historical scripts.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = LEGACY_NARROW_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")
    if spec.n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("Unnamed:")])
    out["gpu"] = spec.gpu
    out["source_file"] = path.name
    out["measurement_status"] = "measured" if spec.measured else "simulated"
    out["n_iterations"] = spec.n_iterations

    out["n_batch"] = pd.to_numeric(out["n_batch"], errors="coerce")
    out["Total_power"] = pd.to_numeric(out["Total_power"], errors="coerce")
    out = out.dropna(subset=["n_batch", "Total_power"])
    out = out[(out["n_batch"] > 0) & (out["Total_power"] >= 0)].copy()

    out["n_samples"] = out["n_batch"] * out["n_iterations"]
    out["energy_per_inference_kwh"] = out["Total_power"] / out["n_samples"]
    out["energy_per_1000_inferences_kwh"] = (
        out["energy_per_inference_kwh"] * 1000.0
    )

    # Retain the legacy field only for discrepancy auditing. Never use it as
    # the normalized outcome in new analyses.
    if "Kg_carbon(batch=1000)" in out.columns:
        out = out.rename(
            columns={"Kg_carbon(batch=1000)": "legacy_kg_carbon_batch1000"}
        )

    return out


def summarize_narrow_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized benchmark energy by GPU, task, and dimension."""
    required = {
        "gpu",
        "Task",
        "Dimension",
        "energy_per_1000_inferences_kwh",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        df.groupby(["gpu", "Task", "Dimension"], dropna=False)[
            "energy_per_1000_inferences_kwh"
        ]
        .agg(n="count", median="median", mean="mean", sd="std", minimum="min", maximum="max")
        .reset_index()
        .sort_values(["Task", "Dimension", "gpu"])
        .reset_index(drop=True)
    )


def architecture_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Count unique architectures represented in each GPU/task/dimension cell.

    Hardware comparisons are only valid when architecture coverage is comparable
    or when models are explicitly matched across GPUs. This table is therefore a
    mandatory audit output.
    """
    required = {"gpu", "Task", "Dimension", "Architecture"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        df.groupby(["gpu", "Task", "Dimension"], dropna=False)
        .agg(
            n_runs=("Architecture", "size"),
            n_architectures=("Architecture", "nunique"),
        )
        .reset_index()
        .sort_values(["Task", "Dimension", "gpu"])
        .reset_index(drop=True)
    )


def matched_architecture_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return GPU medians using only architectures observed on all included GPUs.

    This prevents a misleading hardware comparison caused by different model
    mixes, such as the sparse historical A100 3D segmentation measurements.
    """
    required = {
        "gpu",
        "Task",
        "Dimension",
        "Architecture",
        "energy_per_1000_inferences_kwh",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_gpus = df["gpu"].nunique()
    coverage = (
        df[["gpu", "Task", "Dimension", "Architecture"]]
        .drop_duplicates()
        .groupby(["Task", "Dimension", "Architecture"])["gpu"]
        .nunique()
        .reset_index(name="n_gpus")
    )
    matched = coverage.loc[coverage["n_gpus"] == n_gpus, ["Task", "Dimension", "Architecture"]]
    tmp = df.merge(matched, on=["Task", "Dimension", "Architecture"], how="inner")

    return (
        tmp.groupby(["gpu", "Task", "Dimension"])["energy_per_1000_inferences_kwh"]
        .agg(n="count", n_architectures=lambda x: tmp.loc[x.index, "Architecture"].nunique(), median="median", mean="mean", sd="std")
        .reset_index()
        .sort_values(["Task", "Dimension", "gpu"])
        .reset_index(drop=True)
    )
