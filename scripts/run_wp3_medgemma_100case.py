from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import pathlib
import statistics
import threading
import time

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model" / "case_manifest_100.csv"
OUT_DIR = ROOT / "results" / "wp3" / "medgemma_100case"
HELPER_PATH = ROOT / "scripts" / "run_wp3_remaining_bf16_panel_pilot.py"
MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."
MAX_NEW_TOKENS = 128
BLOCK_SIZE = 10
IDLE_SECONDS = 10.0


def load_helper():
    spec = importlib.util.spec_from_file_location("wp3_medgemma_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load WP3 helper module")
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
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "10-case blocks",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def mean_or_none(values):
    vals = [float(x) for x in values]
    return statistics.mean(vals) if vals else None


def cv(values):
    vals = [float(x) for x in values]
    if len(vals) < 2:
        return None
    m = statistics.mean(vals)
    return statistics.stdev(vals) / m if m else None


def main() -> None:
    helper = load_helper()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file():
        raise RuntimeError("Frozen 100-case Open-I manifest missing")
    cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(cases) != 100:
        raise RuntimeError(f"Expected 100 frozen cases, found {len(cases)}")
    if [int(r["case_index"]) for r in cases] != list(range(1, 101)):
        raise RuntimeError("Frozen case_index is not contiguous 1..100")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    refs = helper.extract_report_texts()
    snapshot = pathlib.Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(ROOT / ".wp3-models" / "MedGemma-4B-pinned"),
    )).resolve()
    pipe = pipeline(
        "image-text-to-text",
        model=str(snapshot),
        device=0,
        dtype=torch.bfloat16,
    )

    warm_img = Image.open(ROOT / cases[0]["local_image_path"]).convert("RGB")
    warm_out = pipe(helper.make_messages(warm_img), max_new_tokens=16, do_sample=False)
    if not helper.extract_generated_text(warm_out):
        raise RuntimeError("Empty MedGemma warmup output")
    del warm_img, warm_out
    torch.cuda.synchronize()

    block_rows = []
    case_rows = []
    progress(0, 10, "MedGemma warmup complete")

    for block_idx in range(10):
        block_cases = cases[block_idx * BLOCK_SIZE:(block_idx + 1) * BLOCK_SIZE]

        idle_rows = []
        idle_stop = threading.Event()
        idle_thread = threading.Thread(target=helper.sample_trace, args=(idle_stop, idle_rows), daemon=True)
        idle_thread.start()
        time.sleep(IDLE_SECONDS)
        idle_stop.set()
        idle_thread.join()
        if not idle_rows:
            raise RuntimeError(f"No idle power samples for block {block_idx + 1}")
        idle_mean_power = statistics.mean(r[1] for r in idle_rows)

        trace = []
        stop = threading.Event()
        sampler = threading.Thread(target=helper.sample_trace, args=(stop, trace), daemon=True)
        sampler.start()
        block_start = time.perf_counter()
        block_case_times = []

        for case in block_cases:
            findings, impression = refs.get(case["source_report_id"], ("", ""))
            reference = " ".join(x for x in (findings, impression) if x).strip()
            img = Image.open(ROOT / case["local_image_path"]).convert("RGB")
            t0 = time.perf_counter()
            out = pipe(helper.make_messages(img), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            pred = helper.extract_generated_text(out)
            if not pred:
                raise RuntimeError(f"Empty output for case {case['case_index']}")
            block_case_times.append(elapsed)
            case_rows.append({
                "model": "MedGemma-4B",
                "repo_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "block": block_idx + 1,
                "case_index": int(case["case_index"]),
                "source_report_id": case["source_report_id"],
                "source_image_id": case["source_image_id"],
                "normal_metadata_stratum": case.get("normal_metadata_stratum", ""),
                "reference_length_quartile": case.get("reference_length_quartile", ""),
                "elapsed_seconds": elapsed,
                "model_word_count": len(helper.normalize_tokens(pred)),
                "unigram_f1": helper.unigram_f1(pred, reference),
                "rouge_l_f1": helper.rouge_l_f1(pred, reference),
                "model_output": pred,
                "reference_findings": findings,
                "reference_impression": impression,
            })
            del img, out

        block_elapsed = time.perf_counter() - block_start
        stop.set()
        sampler.join()
        gross_wh = helper.integrate_wh(trace)
        net_wh = max(0.0, gross_wh - idle_mean_power * block_elapsed / 3600.0)
        block_rows.append({
            "block": block_idx + 1,
            "cases": len(block_cases),
            "gross_gpu_energy_wh_block": gross_wh,
            "gross_gpu_energy_wh_per_case": gross_wh / len(block_cases),
            "net_gpu_energy_wh_block": net_wh,
            "net_gpu_energy_wh_per_case": net_wh / len(block_cases),
            "idle_mean_power_w": idle_mean_power,
            "block_elapsed_seconds": block_elapsed,
            "median_case_elapsed_seconds": statistics.median(block_case_times),
            "mean_gpu_utilization_pct": statistics.mean(r[2] for r in trace) if trace else None,
            "peak_sampled_memory_mib": max(r[3] for r in trace) if trace else None,
        })
        progress(block_idx + 1, 10, f"Completed MedGemma block {block_idx + 1}/10")

    with (OUT_DIR / "block_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        w.writeheader(); w.writerows(block_rows)
    with (OUT_DIR / "case_results_100.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        w.writeheader(); w.writerows(case_rows)

    gross = [r["gross_gpu_energy_wh_per_case"] for r in block_rows]
    net = [r["net_gpu_energy_wh_per_case"] for r in block_rows]
    med_times = [r["median_case_elapsed_seconds"] for r in block_rows]
    report = {
        "status": "WP3_MEDGEMMA_100CASE_OK",
        "model": "MedGemma-4B",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "cases": 100,
        "blocks": 10,
        "block_size": 10,
        "prompt": PROMPT,
        "gross_gpu_energy_wh_per_case_mean_across_blocks": statistics.mean(gross),
        "gross_gpu_energy_wh_per_case_median_across_blocks": statistics.median(gross),
        "gross_gpu_energy_block_cv": cv(gross),
        "net_gpu_energy_wh_per_case_mean_across_blocks": statistics.mean(net),
        "net_gpu_energy_wh_per_case_median_across_blocks": statistics.median(net),
        "net_gpu_energy_block_cv": cv(net),
        "median_case_elapsed_seconds_across_blocks": statistics.median(med_times),
        "mean_unigram_f1_100cases": mean_or_none(r["unigram_f1"] for r in case_rows),
        "mean_rouge_l_f1_100cases": mean_or_none(r["rouge_l_f1"] for r in case_rows),
        "measurement_scope": "Direct NVIDIA GPU board operational energy; model loading and warmup excluded; gross primary and idle-adjusted net secondary.",
        "utility_limit": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WP3_MEDGEMMA_100CASE_OK")
    print(json.dumps(report, sort_keys=True))

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
