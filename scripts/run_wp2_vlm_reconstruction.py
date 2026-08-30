#!/usr/bin/env python3
"""Reconstruct the historical VLM pilot from inventoried runner-side sources.

This task intentionally treats `total_power` as a tracker quantity rather than
asserting a physical unit. It verifies frozen source checksums, normalizes the
historical summary table by completed case calls, and exports source-grounded
protocol evidence for the scripts we can verify directly.
"""
from __future__ import annotations

import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path

CSV_SOURCE = Path("/home/asadr/works/repos/sustainability-stroke/external_data/models_gpus/RTX_6000_Ada_multimodel.csv")
CSV_SHA256 = "e8ace8c0f155c5d036b9b5326e0b12b187bfad01df779ffcb93d0a7a5213d751"
SCRIPT_SOURCES = {
    "InternVL2.py": (
        Path("/home/asadr/works/multimodal/InternVL2.py"),
        "dd2719120220ea3af64f4f4080382d8a2a94c27cf0721fa3e0530cb3d43b91cf",
    ),
    "florence-2.py": (
        Path("/home/asadr/works/multimodal/florence-2.py"),
        "87142a5eee78467f996a4c3f6aadec76264b14b9ccc2f74926b9f11ee1cd2dfe",
    ),
    "paligemma-checkpoint.py": (
        Path("/home/asadr/works/multimodal/.ipynb_checkpoints/paligemma-checkpoint.py"),
        "c9f8d78129e4c28c4adcdfb7e2e6a2da81fd8399805306d090795fede6617322",
    ),
    "moondream-next.py": (
        Path("/home/asadr/works/multimodal/moondream-next.py"),
        "14e14a6657699959a86cd939ba9bedd647811688e9cd078d44d7957b25c52c0a",
    ),
}
OUT = Path("results/wp2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing WP2 source: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, observed {observed}")


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return float("nan")
    return float(value)


def median(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return statistics.median(vals) if vals else float("nan")


def mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    return statistics.fmean(vals) if vals else float("nan")


def main() -> None:
    verify(CSV_SOURCE, CSV_SHA256)
    for _, (path, expected) in SCRIPT_SOURCES.items():
        verify(path, expected)

    with CSV_SOURCE.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {"Device", "Model", "Size", "trial", "n_batch", "total_power", "kg_carbon", "total_wall_clock_time"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise RuntimeError(f"Historical VLM CSV missing columns: {sorted(missing)}")

    OUT.mkdir(parents=True, exist_ok=True)
    normalized_path = OUT / "historical_vlm_normalized_runs.csv"
    normalized_fields = list(rows[0].keys()) + [
        "tracker_total_power_per_case",
        "tracker_total_power_per_1000_cases",
        "wall_clock_time_per_case",
    ]
    normalized: list[dict[str, object]] = []
    with normalized_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=normalized_fields)
        w.writeheader()
        for row in rows:
            n = fnum(row, "n_batch")
            if not (n > 0):
                raise RuntimeError(f"Invalid n_batch in row: {row}")
            total_power = fnum(row, "total_power")
            wall = fnum(row, "total_wall_clock_time")
            out = dict(row)
            out["tracker_total_power_per_case"] = total_power / n
            out["tracker_total_power_per_1000_cases"] = total_power / n * 1000.0
            out["wall_clock_time_per_case"] = wall / n
            normalized.append(out)
            w.writerow(out)

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        groups[(str(row["Device"]), str(row["Model"]), str(row["Size"]))].append(row)

    summary_path = OUT / "historical_vlm_model_summary.csv"
    summary_fields = [
        "Device", "Model", "Size", "n_runs", "n_batch_values",
        "median_tracker_total_power_per_case", "mean_tracker_total_power_per_case",
        "median_tracker_total_power_per_1000_cases", "median_wall_clock_time_per_case",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for key in sorted(groups):
            g = groups[key]
            per_case = [float(r["tracker_total_power_per_case"]) for r in g]
            per_1000 = [float(r["tracker_total_power_per_1000_cases"]) for r in g]
            wall = [float(r["wall_clock_time_per_case"]) for r in g]
            nvals = sorted({str(r["n_batch"]) for r in g})
            w.writerow({
                "Device": key[0], "Model": key[1], "Size": key[2],
                "n_runs": len(g), "n_batch_values": ";".join(nvals),
                "median_tracker_total_power_per_case": median(per_case),
                "mean_tracker_total_power_per_case": mean(per_case),
                "median_tracker_total_power_per_1000_cases": median(per_1000),
                "median_wall_clock_time_per_case": median(wall),
            })

    patterns = (
        "range(128)", "n_batch", "ImpactTracker", "tracker.launch_impact_monitor",
        "tracker.stop", ".cuda()", "bfloat16", "torch.bfloat16", "max_new_tokens",
        "num_beams", "model.chat", "generate(", "<image>", "Please describe the image shortly",
        "<OD>", "processor(", "post_process_generation",
    )
    report = [
        "# WP2 historical VLM reconstruction",
        "",
        f"Historical summary rows: **{len(rows)}**",
        f"CSV SHA256: `{CSV_SHA256}`",
        "",
        "## Denominator and endpoint",
        "",
        "The historical table is normalized as `total_power / n_batch`. The endpoint is named `tracker_total_power_per_case` because the physical unit of the historical tracker field remains unresolved. No kg-carbon field is used as the primary normalized endpoint.",
        "",
        "For scripts in which the source shows 128 sequential generation/chat calls inside the tracked interval, `n_batch = 128` is interpreted as 128 completed VLM case calls, not a tensor batch size.",
        "",
        "## Source-grounded protocol excerpts",
        "",
    ]
    for name, (path, expected) in SCRIPT_SOURCES.items():
        report.append(f"### {name}")
        report.append("")
        report.append(f"Source: `{path}`")
        report.append(f"SHA256: `{expected}`")
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = []
        for i, line in enumerate(text, start=1):
            if any(p.lower() in line.lower() for p in patterns):
                hits.append((i, line.strip()))
        if not hits:
            report.append("No selected protocol markers found.")
        else:
            for i, line in hits[:80]:
                report.append(f"- L{i}: `{line}`")
        report.append("")

    report.extend([
        "## Interpretation constraints",
        "",
        "- Cross-model comparisons are descriptive because scripts may differ in task, preprocessing, decoding, device placement, and measured pipeline boundaries.",
        "- InternVL2 and Florence-2 should not be treated as protocol-equivalent without accounting for those differences.",
        "- PaliGemma evidence is from a notebook checkpoint source, and the inventoried Moondream script is `moondream-next.py`; matching either script to a specific historical CSV label must therefore remain explicit rather than assumed.",
        "- Historical carbon values should not be used for cross-hardware carbon comparisons. Future carbon estimates should apply an independent grid-carbon intensity to the measured energy endpoint once the energy unit is definitively established.",
        "",
    ])
    (OUT / "historical_vlm_protocol_audit.md").write_text("\n".join(report), encoding="utf-8")

    print(f"WP2_VLM_RECONSTRUCTION_OK rows={len(rows)} groups={len(groups)}")


if __name__ == "__main__":
    main()
