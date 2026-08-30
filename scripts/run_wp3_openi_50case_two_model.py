from __future__ import annotations

import csv
import importlib.util
import json
import os
import pathlib
import shutil
import statistics
import time

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "wp3" / "openi_50case_two_model"
PREP_OUT = ROOT / "results" / "wp3" / "openi_cxr_50"
DATA_DIR = ROOT / ".wp3-data" / "openi_cxr_pilot"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
INTERNVL_ID = "OpenGVLab/InternVL3-8B"
INTERNVL_REVISION = "dab7194eaadae9ff191fef49b961847a18b4c822"
N_CASES = 50
BATCH_SIZE = 10


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


def prepare_50_cases() -> pathlib.Path:
    prep = load_module("wp3_openi_prep_50", ROOT / "scripts" / "prepare_wp3_openi_cxr_pilot.py")
    prep.N_CASES = N_CASES
    prep.DATA_DIR = DATA_DIR
    prep.OUT_DIR = PREP_OUT
    run_without_subprogress(prep.main)
    manifest = PREP_OUT / "case_manifest.csv"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    if len(rows) != N_CASES:
        raise RuntimeError(f"Expected {N_CASES} prepared cases, found {len(rows)}")
    return manifest


def split_batches(manifest: pathlib.Path) -> list[pathlib.Path]:
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    batch_dir = OUT_DIR / "batch_manifests"
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        if len(batch) != BATCH_SIZE:
            raise RuntimeError("50-case manifest did not split evenly into 10-case blocks")
        path = batch_dir / f"batch_{start // BATCH_SIZE + 1}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(batch[0].keys()))
            w.writeheader(); w.writerows(batch)
        paths.append(path)
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_stages = 14
    manifest = prepare_50_cases()
    progress(1, total_stages, "Prepared frozen 50-case Open-I manifest")
    batches = split_batches(manifest)

    qwen_snapshot = pathlib.Path(snapshot_download(
        repo_id=QWEN_ID,
        revision=QWEN_REVISION,
        cache_dir=str(ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"),
    )).resolve()
    internvl_snapshot = pathlib.Path(snapshot_download(
        repo_id=INTERNVL_ID,
        revision=INTERNVL_REVISION,
        cache_dir=str(ROOT / ".wp3-models" / "InternVL3-8B-pinned"),
    )).resolve()
    progress(2, total_stages, "Pinned model revisions ready")

    qwen = load_module("wp3_qwen_energy_50", ROOT / "scripts" / "run_wp3_openi_qwen_pilot.py")
    qwen.MODEL_ID = str(qwen_snapshot)
    qwen.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"
    qval = load_module("wp3_qwen_validation_50", ROOT / "scripts" / "validate_wp3_openi_qwen_outputs.py")
    qval.MODEL_ID = str(qwen_snapshot)
    qval.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"
    internvl = load_module("wp3_internvl_energy_50", ROOT / "scripts" / "run_wp3_internvl3_openi_pilot.py")
    internvl.MODEL_ID = str(internvl_snapshot)
    internvl.MODEL_CACHE = ROOT / ".wp3-models" / "InternVL3-8B-pinned"

    block_rows: list[dict[str, object]] = []
    qwen_review_rows: list[dict[str, str]] = []
    internvl_case_rows: list[dict[str, str]] = []
    qwen_f1: list[float] = []
    qwen_rouge: list[float] = []
    internvl_f1: list[float] = []
    internvl_rouge: list[float] = []

    for i, batch in enumerate(batches, start=1):
        out = OUT_DIR / f"qwen_batch_{i}"
        if out.exists(): shutil.rmtree(out)
        qwen.MANIFEST = batch
        qwen.OUT_DIR = out
        run_without_subprogress(qwen.main)
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block_rows.append({
            "model": "Qwen2.5-VL-7B-Instruct", "batch": i,
            "gross_wh_per_case": rep["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": rep["net_gpu_energy_wh_per_case"],
            "median_case_seconds": rep["median_case_elapsed_seconds"],
            "idle_mean_power_w": rep["idle_mean_power_w"],
            "peak_vram_mib": rep["peak_vram_mib_torch"],
        })
        progress(2 + i, total_stages, f"Qwen energy block {i}/5")

    for i, batch in enumerate(batches, start=1):
        out = OUT_DIR / f"qwen_validation_batch_{i}"
        if out.exists(): shutil.rmtree(out)
        qval.MANIFEST = batch
        qval.OUT_DIR = out
        run_without_subprogress(qval.main)
        with (out / "case_review.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        qwen_review_rows.extend(rows)
        qwen_f1.extend(float(r["unigram_f1"]) for r in rows)
        qwen_rouge.extend(float(r["rouge_l_f1"]) for r in rows)
    progress(8, total_stages, "Qwen 50-case output screening complete")

    for i, batch in enumerate(batches, start=1):
        out = OUT_DIR / f"internvl_batch_{i}"
        if out.exists(): shutil.rmtree(out)
        internvl.MANIFEST = batch
        internvl.OUT_DIR = out
        run_without_subprogress(internvl.main)
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block_rows.append({
            "model": "InternVL3-8B", "batch": i,
            "gross_wh_per_case": rep["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": rep["net_gpu_energy_wh_per_case"],
            "median_case_seconds": rep["median_case_elapsed_seconds"],
            "idle_mean_power_w": rep["idle_mean_power_w"],
            "peak_vram_mib": rep["peak_vram_mib_torch"],
        })
        with (out / "case_results.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        internvl_case_rows.extend(rows)
        internvl_f1.extend(float(r["unigram_f1"]) for r in rows)
        internvl_rouge.extend(float(r["rouge_l_f1"]) for r in rows)
        progress(8 + i, total_stages, f"InternVL energy/output block {i}/5")

    with (OUT_DIR / "block_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        w.writeheader(); w.writerows(block_rows)
    with (OUT_DIR / "qwen_case_review_50.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qwen_review_rows[0].keys()))
        w.writeheader(); w.writerows(qwen_review_rows)
    with (OUT_DIR / "internvl_case_results_50.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(internvl_case_rows[0].keys()))
        w.writeheader(); w.writerows(internvl_case_rows)

    def model_blocks(name: str, key: str) -> list[float]:
        return [float(r[key]) for r in block_rows if r["model"] == name]

    q_net = model_blocks("Qwen2.5-VL-7B-Instruct", "net_wh_per_case")
    q_gross = model_blocks("Qwen2.5-VL-7B-Instruct", "gross_wh_per_case")
    q_time = model_blocks("Qwen2.5-VL-7B-Instruct", "median_case_seconds")
    i_net = model_blocks("InternVL3-8B", "net_wh_per_case")
    i_gross = model_blocks("InternVL3-8B", "gross_wh_per_case")
    i_time = model_blocks("InternVL3-8B", "median_case_seconds")

    summary = {
        "status": "WP3_OPENI_50CASE_TWO_MODEL_OK",
        "dataset": "Open-I frozen 50-case single-image expansion",
        "cases": N_CASES,
        "energy_blocks_per_model": len(batches),
        "cases_per_energy_block": BATCH_SIZE,
        "precision": "BF16",
        "qwen": {
            "model_id": QWEN_ID, "revision": QWEN_REVISION,
            "median_net_wh_per_case_across_blocks": statistics.median(q_net),
            "net_block_cv": cv(q_net),
            "median_gross_wh_per_case_across_blocks": statistics.median(q_gross),
            "gross_block_cv": cv(q_gross),
            "median_case_seconds_across_blocks": statistics.median(q_time),
            "mean_unigram_f1_50_cases": statistics.mean(qwen_f1),
            "mean_rouge_l_f1_50_cases": statistics.mean(qwen_rouge),
        },
        "internvl3": {
            "model_id": INTERNVL_ID, "revision": INTERNVL_REVISION,
            "median_net_wh_per_case_across_blocks": statistics.median(i_net),
            "net_block_cv": cv(i_net),
            "median_gross_wh_per_case_across_blocks": statistics.median(i_gross),
            "gross_block_cv": cv(i_gross),
            "median_case_seconds_across_blocks": statistics.median(i_time),
            "mean_unigram_f1_50_cases": statistics.mean(internvl_f1),
            "mean_rouge_l_f1_50_cases": statistics.mean(internvl_rouge),
        },
        "internvl_to_qwen": {
            "net_energy_ratio": statistics.median(i_net) / statistics.median(q_net),
            "gross_energy_ratio": statistics.median(i_gross) / statistics.median(q_gross),
            "runtime_ratio": statistics.median(i_time) / statistics.median(q_time),
        },
        "measurement_scope": "Direct NVIDIA GPU board operational energy. Model loading and warmup excluded. Gross board energy is primary; idle-adjusted energy is secondary.",
        "utility_note": "Unigram F1 and ROUGE-L remain screening metrics only and do not establish clinical adequacy.",
        "selection_rule": "First 50 lexicographically sorted eligible Open-I XML reports with at least one parentImage and nonempty FINDINGS or IMPRESSION; first parentImage used.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(manifest, OUT_DIR / "case_manifest_50.csv")
    progress(14, total_stages, "50-case two-model comparison complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
