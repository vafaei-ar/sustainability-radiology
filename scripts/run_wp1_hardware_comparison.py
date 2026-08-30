#!/usr/bin/env python3
"""Publication-oriented architecture-level paired GPU comparison for WP1.

The script first runs the verified legacy-source WP1 audit, then collapses replicate
runs to one median energy estimate per architecture/GPU/task/dimension and compares
GPUs only within architectures observed on all three GPUs.
"""
from __future__ import annotations

import csv
import math
import random
import runpy
import statistics
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("results/wp1")
GPUS = ["A100", "RTX5000", "RTX_6000_Ada"]
PAIRS = [
    ("A100", "RTX5000"),
    ("A100", "RTX_6000_Ada"),
    ("RTX_6000_Ada", "RTX5000"),
]
BOOTSTRAP_N = 10000
SEED = 20260830

# Materialize verified inputs and regenerate the base WP1 outputs.
runpy.run_path("scripts/run_wp1_from_legacy_source.py")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


runs = read_rows(OUTDIR / "normalized_runs.csv")
groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
for r in runs:
    key = (r["gpu"], r["Task"], r["Dimension"], r["Architecture"])
    groups[key].append(float(r["energy_per_1000_inferences_kwh"]))

arch_rows: list[dict[str, object]] = []
for key in sorted(groups):
    vals = groups[key]
    arch_rows.append({
        "gpu": key[0],
        "Task": key[1],
        "Dimension": key[2],
        "Architecture": key[3],
        "n_runs": len(vals),
        "median_kwh_per_1000": statistics.median(vals),
        "mean_kwh_per_1000": statistics.fmean(vals),
        "sd_kwh_per_1000": statistics.stdev(vals) if len(vals) > 1 else "",
    })
write_rows(
    OUTDIR / "architecture_level_summary.csv",
    arch_rows,
    ["gpu", "Task", "Dimension", "Architecture", "n_runs", "median_kwh_per_1000", "mean_kwh_per_1000", "sd_kwh_per_1000"],
)

lookup = {
    (str(r["gpu"]), str(r["Task"]), str(r["Dimension"]), str(r["Architecture"])): float(r["median_kwh_per_1000"])
    for r in arch_rows
}
cell_arches: dict[tuple[str, str], set[str]] = defaultdict(set)
for r in arch_rows:
    cell_arches[(str(r["Task"]), str(r["Dimension"]))].add(str(r["Architecture"]))

rng = random.Random(SEED)
ratio_rows: list[dict[str, object]] = []
for cell in sorted(cell_arches):
    task, dim = cell
    matched = []
    for arch in sorted(cell_arches[cell]):
        if all((gpu, task, dim, arch) in lookup for gpu in GPUS):
            matched.append(arch)
    for numerator, denominator in PAIRS:
        ratios = [lookup[(numerator, task, dim, arch)] / lookup[(denominator, task, dim, arch)] for arch in matched]
        if not ratios:
            continue
        log_ratios = [math.log(x) for x in ratios]
        geo = math.exp(statistics.fmean(log_ratios))
        med = statistics.median(ratios)
        ci_low: object = ""
        ci_high: object = ""
        if len(ratios) >= 4:
            boots = []
            n = len(log_ratios)
            for _ in range(BOOTSTRAP_N):
                sample = [log_ratios[rng.randrange(n)] for _ in range(n)]
                boots.append(math.exp(statistics.fmean(sample)))
            ci_low = quantile(boots, 0.025)
            ci_high = quantile(boots, 0.975)
        ratio_rows.append({
            "Task": task,
            "Dimension": dim,
            "numerator_gpu": numerator,
            "denominator_gpu": denominator,
            "n_architectures": len(ratios),
            "geometric_mean_ratio": geo,
            "median_ratio": med,
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
        })

write_rows(
    OUTDIR / "paired_gpu_ratios.csv",
    ratio_rows,
    ["Task", "Dimension", "numerator_gpu", "denominator_gpu", "n_architectures", "geometric_mean_ratio", "median_ratio", "min_ratio", "max_ratio", "bootstrap_ci95_low", "bootstrap_ci95_high"],
)

report = [
    "# WP1 architecture-level paired GPU comparison",
    "",
    "Replicate runs are first collapsed to the architecture-level median. Hardware ratios therefore weight each matched architecture once, avoiding unequal replicate weighting across GPUs.",
    "",
    "Ratio < 1 means the numerator GPU used less measured energy per 1000 benchmark samples than the denominator GPU.",
    "",
    "| Task | Dim | Numerator / denominator | Architectures | Geometric mean ratio | 95% bootstrap CI |",
    "|---|---:|---|---:|---:|---:|",
]
for r in ratio_rows:
    ci = "not estimated" if r["bootstrap_ci95_low"] == "" else f"{float(r['bootstrap_ci95_low']):.3f}-{float(r['bootstrap_ci95_high']):.3f}"
    report.append(
        f"| {r['Task']} | {r['Dimension']} | {r['numerator_gpu']} / {r['denominator_gpu']} | {r['n_architectures']} | {float(r['geometric_mean_ratio']):.3f} | {ci} |"
    )
report.extend([
    "",
    "Notes:",
    "- Bootstrap resamples architectures, not individual replicate runs.",
    "- Cells with fewer than four matched architectures do not receive a bootstrap CI and should be treated as descriptive only.",
    "- These comparisons remain conditional on verification that the historical `Total_power` field is measured energy in kWh.",
    "",
])
(OUTDIR / "wp1_hardware_comparison.md").write_text("\n".join(report), encoding="utf-8")

print(f"WP1_HARDWARE_OK architecture_rows={len(arch_rows)} ratio_rows={len(ratio_rows)}")
