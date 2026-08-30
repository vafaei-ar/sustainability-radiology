from __future__ import annotations

import csv
import importlib.util
import json
import os
import pathlib
import shutil
import statistics
import time

import torch
from huggingface_hub import HfApi, snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "wp3" / "two_model_reproducibility"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERNVL_ID = "OpenGVLab/InternVL3-8B"
INTERNVL_REVISION = "dab7194eaadae9ff191fef49b961847a18b4c822"
REPEATS = 3


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total,
        "phase": phase,
        "unit": "stages",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def run_without_subprogress(fn) -> None:
    prior = os.environ.pop("RUNRELAY_PROGRESS_FILE", None)
    try:
        fn()
    finally:
        if prior is not None:
            os.environ["RUNRELAY_PROGRESS_FILE"] = prior


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    api = HfApi()
    qwen_revision = api.model_info(QWEN_ID).sha
    internvl_revision = api.model_info(INTERNVL_ID, revision=INTERNVL_REVISION).sha
    if internvl_revision != INTERNVL_REVISION:
        raise RuntimeError(f"InternVL revision mismatch: {internvl_revision}")

    qwen_snapshot = pathlib.Path(snapshot_download(
        repo_id=QWEN_ID,
        revision=qwen_revision,
        cache_dir=str(ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"),
    )).resolve()
    internvl_snapshot = pathlib.Path(snapshot_download(
        repo_id=INTERNVL_ID,
        revision=internvl_revision,
        cache_dir=str(ROOT / ".wp3-models" / "InternVL3-8B-pinned"),
    )).resolve()
    progress(1, 9, "Pinned model revisions resolved")

    qwen = load_module("wp3_qwen_energy", ROOT / "scripts" / "run_wp3_openi_qwen_pilot.py")
    qwen.MODEL_ID = str(qwen_snapshot)
    qwen.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"

    internvl = load_module("wp3_internvl_energy", ROOT / "scripts" / "run_wp3_internvl3_openi_pilot.py")
    internvl.MODEL_ID = str(internvl_snapshot)
    internvl.MODEL_CACHE = ROOT / ".wp3-models" / "InternVL3-8B-pinned"

    repeat_rows: list[dict[str, object]] = []
    qwen_reports = []
    internvl_reports = []

    for rep in range(1, REPEATS + 1):
        rep_dir = OUT_DIR / f"qwen_repeat_{rep}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        qwen.OUT_DIR = rep_dir
        run_without_subprogress(qwen.main)
        report = json.loads((rep_dir / "report.json").read_text(encoding="utf-8"))
        qwen_reports.append(report)
        repeat_rows.append({
            "model": "Qwen2.5-VL-7B-Instruct",
            "revision": qwen_revision,
            "repeat": rep,
            "gross_wh_per_case": report["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": report["net_gpu_energy_wh_per_case"],
            "median_case_seconds": report["median_case_elapsed_seconds"],
            "idle_mean_power_w": report["idle_mean_power_w"],
            "peak_vram_mib": report["peak_vram_mib_torch"],
        })
        progress(1 + rep, 9, f"Qwen repeat {rep}/{REPEATS}")

    qval = load_module("wp3_qwen_validation", ROOT / "scripts" / "validate_wp3_openi_qwen_outputs.py")
    qval.MODEL_ID = str(qwen_snapshot)
    qval.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"
    qval.OUT_DIR = OUT_DIR / "qwen_validation"
    run_without_subprogress(qval.main)
    qwen_validation = json.loads((qval.OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    progress(5, 9, "Pinned Qwen output screening")

    for rep in range(1, REPEATS + 1):
        rep_dir = OUT_DIR / f"internvl_repeat_{rep}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        internvl.OUT_DIR = rep_dir
        run_without_subprogress(internvl.main)
        report = json.loads((rep_dir / "report.json").read_text(encoding="utf-8"))
        internvl_reports.append(report)
        repeat_rows.append({
            "model": "InternVL3-8B",
            "revision": internvl_revision,
            "repeat": rep,
            "gross_wh_per_case": report["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": report["net_gpu_energy_wh_per_case"],
            "median_case_seconds": report["median_case_elapsed_seconds"],
            "idle_mean_power_w": report["idle_mean_power_w"],
            "peak_vram_mib": report["peak_vram_mib_torch"],
        })
        progress(5 + rep, 9, f"InternVL repeat {rep}/{REPEATS}")

    fields = list(repeat_rows[0].keys())
    with (OUT_DIR / "repeats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repeat_rows)

    shutil.copy2(qval.OUT_DIR / "case_review.csv", OUT_DIR / "qwen_case_review.csv")
    shutil.copy2(OUT_DIR / "internvl_repeat_1" / "case_results.csv", OUT_DIR / "internvl_case_results.csv")

    q_net = [float(x["net_gpu_energy_wh_per_case"]) for x in qwen_reports]
    i_net = [float(x["net_gpu_energy_wh_per_case"]) for x in internvl_reports]
    q_gross = [float(x["gross_gpu_energy_wh_per_case"]) for x in qwen_reports]
    i_gross = [float(x["gross_gpu_energy_wh_per_case"]) for x in internvl_reports]
    q_time = [float(x["median_case_elapsed_seconds"]) for x in qwen_reports]
    i_time = [float(x["median_case_elapsed_seconds"]) for x in internvl_reports]

    summary = {
        "status": "WP3_TWO_MODEL_REPRODUCIBILITY_OK",
        "dataset": "Open-I frozen 10-case pilot",
        "precision": "BF16",
        "repeats_per_model": REPEATS,
        "qwen": {
            "model_id": QWEN_ID,
            "revision": qwen_revision,
            "median_net_wh_per_case": statistics.median(q_net),
            "net_energy_cv": cv(q_net),
            "median_gross_wh_per_case": statistics.median(q_gross),
            "gross_energy_cv": cv(q_gross),
            "median_case_seconds_across_repeats": statistics.median(q_time),
            "mean_unigram_f1": qwen_validation["mean_unigram_f1"],
            "mean_rouge_l_f1": qwen_validation["mean_rouge_l_f1"],
        },
        "internvl3": {
            "model_id": INTERNVL_ID,
            "revision": internvl_revision,
            "median_net_wh_per_case": statistics.median(i_net),
            "net_energy_cv": cv(i_net),
            "median_gross_wh_per_case": statistics.median(i_gross),
            "gross_energy_cv": cv(i_gross),
            "median_case_seconds_across_repeats": statistics.median(i_time),
            "mean_unigram_f1": statistics.mean(float(x["mean_unigram_f1"]) for x in internvl_reports),
            "mean_rouge_l_f1": statistics.mean(float(x["mean_rouge_l_f1"]) for x in internvl_reports),
        },
        "internvl_to_qwen": {
            "net_energy_ratio": statistics.median(i_net) / statistics.median(q_net),
            "gross_energy_ratio": statistics.median(i_gross) / statistics.median(q_gross),
            "runtime_ratio": statistics.median(i_time) / statistics.median(q_time),
        },
        "measurement_scope": "direct NVIDIA GPU board operational energy; model loading and warmup excluded; idle-adjusted energy secondary to gross board energy",
        "utility_note": "Unigram F1 and ROUGE-L are screening metrics only and do not establish clinical adequacy.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(9, 9, "Two-model comparison complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
