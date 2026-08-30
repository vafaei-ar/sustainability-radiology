#!/usr/bin/env python3
"""Run the publication-grade WP1 audit for historical narrow-model benchmarks.

This script intentionally uses only the Python standard library so it can run in
RunRelay's safe exact-commit environment without relying on an inherited Python
environment.

Expected inputs
---------------
data/legacy/narrow/A100.csv
data/legacy/narrow/RTX5000.csv
data/legacy/narrow/RTX_6000_Ada.csv

Outputs
-------
results/wp1/normalized_runs.csv
results/wp1/architecture_coverage.csv
results/wp1/matched_architecture_summary.csv
results/wp1/run_variability.csv
results/wp1/wp1_report.md
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

N_ITERATIONS = 5000
INPUTS = {
    "A100": Path("data/legacy/narrow/A100.csv"),
    "RTX5000": Path("data/legacy/narrow/RTX5000.csv"),
    "RTX_6000_Ada": Path("data/legacy/narrow/RTX_6000_Ada.csv"),
}
EXPECTED_SHA256 = {
    "A100": "30ac79634ffdf914c21fbcd7c37b42c877aea0957e1983227f92c8da58f40bb4",
    "RTX5000": "e2fcf2759ef3d0c940c13dea7435ec7bee0c094aeb7c88ef8e3c6861c73f5f56",
    "RTX_6000_Ada": "1ec0faa176e1a5b982971b674f0a6b891b78ffa2cf057ad070e9268f620809d1",
}
OUTDIR = Path("results/wp1")
REQUIRED = {"Task", "Architecture", "Dimension", "itry", "n_batch", "Total_power"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_runs(gpu: str, path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    observed = sha256(path)
    expected = EXPECTED_SHA256[gpu]
    if observed != expected:
        raise RuntimeError(
            f"Input checksum mismatch for {path}: expected {expected}, observed {observed}"
        )

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for source_row, row in enumerate(reader, start=2):
            batch = safe_float(row.get("n_batch", ""))
            total = safe_float(row.get("Total_power", ""))
            if batch is None or total is None or batch <= 0 or total < 0:
                continue
            n_samples = int(round(batch * N_ITERATIONS))
            energy_per_inference = total / n_samples
            rows.append(
                {
                    "gpu": gpu,
                    "source_file": path.name,
                    "source_row": source_row,
                    "Task": row["Task"].strip(),
                    "Architecture": row["Architecture"].strip(),
                    "Dimension": row["Dimension"].strip(),
                    "itry": row["itry"].strip(),
                    "n_batch": batch,
                    "n_iterations": N_ITERATIONS,
                    "n_samples": n_samples,
                    "total_energy_kwh": total,
                    "energy_per_inference_kwh": energy_per_inference,
                    "energy_per_1000_inferences_kwh": energy_per_inference * 1000.0,
                    "measurement_status": "measured",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def cv(values: list[float]) -> float:
    m = mean(values)
    return sd(values) / m if len(values) > 1 and m != 0 else math.nan


def median_abs_deviation(values: list[float]) -> float:
    if not values:
        return math.nan
    med = statistics.median(values)
    return statistics.median([abs(x - med) for x in values])


def main() -> None:
    all_runs: list[dict[str, object]] = []
    checksums = []
    for gpu, path in INPUTS.items():
        all_runs.extend(load_runs(gpu, path))
        checksums.append((gpu, path.as_posix(), sha256(path)))

    if not all_runs:
        raise RuntimeError("No valid benchmark rows loaded")

    normalized_fields = [
        "gpu", "source_file", "source_row", "Task", "Architecture", "Dimension",
        "itry", "n_batch", "n_iterations", "n_samples", "total_energy_kwh",
        "energy_per_inference_kwh", "energy_per_1000_inferences_kwh",
        "measurement_status",
    ]
    write_csv(OUTDIR / "normalized_runs.csv", all_runs, normalized_fields)

    coverage_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in all_runs:
        coverage_groups[(str(r["gpu"]), str(r["Task"]), str(r["Dimension"]))].append(r)

    coverage_rows = []
    for key in sorted(coverage_groups):
        rs = coverage_groups[key]
        coverage_rows.append({
            "gpu": key[0], "Task": key[1], "Dimension": key[2],
            "n_runs": len(rs),
            "n_architectures": len({str(r["Architecture"]) for r in rs}),
        })
    write_csv(
        OUTDIR / "architecture_coverage.csv", coverage_rows,
        ["gpu", "Task", "Dimension", "n_runs", "n_architectures"],
    )

    gpus = sorted(INPUTS)
    arch_gpu: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in all_runs:
        arch_gpu[(str(r["Task"]), str(r["Dimension"]), str(r["Architecture"]))].add(str(r["gpu"]))
    matched = {k for k, seen in arch_gpu.items() if seen == set(gpus)}

    matched_groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in all_runs:
        arch_key = (str(r["Task"]), str(r["Dimension"]), str(r["Architecture"]))
        if arch_key in matched:
            matched_groups[(str(r["gpu"]), str(r["Task"]), str(r["Dimension"]))].append(r)

    matched_rows = []
    for key in sorted(matched_groups):
        rs = matched_groups[key]
        vals = [float(r["energy_per_1000_inferences_kwh"]) for r in rs]
        matched_rows.append({
            "gpu": key[0], "Task": key[1], "Dimension": key[2],
            "n_runs": len(rs),
            "n_architectures": len({str(r["Architecture"]) for r in rs}),
            "median_kwh_per_1000": statistics.median(vals),
            "mean_kwh_per_1000": mean(vals),
            "sd_kwh_per_1000": sd(vals),
            "cv": cv(vals),
        })
    write_csv(
        OUTDIR / "matched_architecture_summary.csv", matched_rows,
        ["gpu", "Task", "Dimension", "n_runs", "n_architectures",
         "median_kwh_per_1000", "mean_kwh_per_1000", "sd_kwh_per_1000", "cv"],
    )

    run_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for r in all_runs:
        key = (str(r["gpu"]), str(r["Task"]), str(r["Dimension"]), str(r["Architecture"]))
        run_groups[key].append(float(r["energy_per_1000_inferences_kwh"]))

    variability_rows = []
    for key in sorted(run_groups):
        vals = run_groups[key]
        med = statistics.median(vals)
        mad = median_abs_deviation(vals)
        # Robust flag only when there are >=3 replicates and nonzero MAD.
        outliers = 0
        if len(vals) >= 3 and mad > 0:
            outliers = sum(abs(x - med) / mad > 3.5 for x in vals)
        variability_rows.append({
            "gpu": key[0], "Task": key[1], "Dimension": key[2], "Architecture": key[3],
            "n_runs": len(vals), "median_kwh_per_1000": med,
            "mean_kwh_per_1000": mean(vals), "sd_kwh_per_1000": sd(vals),
            "cv": cv(vals), "mad_kwh_per_1000": mad, "robust_outlier_count": outliers,
        })
    write_csv(
        OUTDIR / "run_variability.csv", variability_rows,
        ["gpu", "Task", "Dimension", "Architecture", "n_runs",
         "median_kwh_per_1000", "mean_kwh_per_1000", "sd_kwh_per_1000", "cv",
         "mad_kwh_per_1000", "robust_outlier_count"],
    )

    report = [
        "# WP1 benchmark audit report",
        "",
        f"Normalized measured runs: **{len(all_runs)}**",
        f"Benchmark iterations per run: **{N_ITERATIONS}**",
        "",
        "## Input integrity",
        "",
    ]
    for gpu, path, digest in checksums:
        report.append(f"- {gpu}: `{path}` SHA256 `{digest}`")
    report.extend([
        "",
        "## Normalization",
        "",
        "`energy_per_inference_kwh = Total_power / (n_batch * 5000)`",
        "",
        "The historical `Kg_carbon(batch=1000)` field is not used as the normalized endpoint.",
        "",
        "## Matched-architecture summary",
        "",
        "| GPU | Task | Dimension | Runs | Architectures | Median kWh/1000 | CV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in matched_rows:
        report.append(
            f"| {r['gpu']} | {r['Task']} | {r['Dimension']} | {r['n_runs']} | "
            f"{r['n_architectures']} | {float(r['median_kwh_per_1000']):.9f} | "
            f"{float(r['cv']):.3f} |"
        )
    report.extend([
        "",
        "## Audit notes",
        "",
        "- Hardware comparisons use architectures observed on all included GPUs within each task/dimension cell.",
        "- `run_variability.csv` summarizes replicate variability by GPU, task, dimension, and architecture.",
        "- Robust outlier flags are descriptive only and do not automatically exclude observations.",
        "- All rows in these three source CSVs are treated as measured. Simulated files are intentionally excluded from WP1.",
        "",
    ])
    (OUTDIR / "wp1_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"WP1_OK normalized_runs={len(all_runs)} matched_cells={len(matched_rows)}")
    for r in matched_rows:
        print(
            f"{r['gpu']} {r['Task']} {r['Dimension']} "
            f"median_kwh_per_1000={float(r['median_kwh_per_1000']):.9f} "
            f"n_arch={r['n_architectures']}"
        )


if __name__ == "__main__":
    main()
