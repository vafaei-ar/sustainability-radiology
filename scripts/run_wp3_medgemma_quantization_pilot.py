from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import torch
from PIL import Image
from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model" / "case_manifest_100.csv"
OUT_DIR = ROOT / "results" / "wp3" / "medgemma_quantization_pilot"
HELPER_PATH = ROOT / "scripts" / "run_wp3_remaining_bf16_panel_pilot.py"
MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."
MAX_NEW_TOKENS = 128
PILOT_CASES = 10
REPEATS = 3
IDLE_SECONDS = 8.0
BNB_VERSION = "0.47.0"
PRECISIONS = ("bf16", "int8", "int4")


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
        "unit": "precision-repeat blocks",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def ensure_bitsandbytes() -> dict:
    """Install bitsandbytes into the project-local WP3 virtual environment.

    Transformers checks package metadata through importlib.metadata. Installing
    bitsandbytes into a detached --target directory can make the module
    importable while leaving that metadata invisible to the Transformers
    quantizer check. This task already runs under .venv-wp3, so install the
    pinned package into that project-local environment instead.
    """
    try:
        import bitsandbytes as bnb  # type: ignore
        return {"status": "available", "version": getattr(bnb, "__version__", "unknown"), "installed_now": False}
    except Exception:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input", "--upgrade",
            f"bitsandbytes=={BNB_VERSION}",
        ]
        run = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        if run.returncode != 0:
            return {"status": "failed", "version": None, "installed_now": False, "error": (run.stdout or "")[-4000:]}
        import importlib
        importlib.invalidate_caches()
        import bitsandbytes as bnb  # type: ignore
        return {"status": "available", "version": getattr(bnb, "__version__", "unknown"), "installed_now": True}


def make_pipeline(snapshot: pathlib.Path, precision: str):
    from transformers import BitsAndBytesConfig, pipeline

    kwargs = {
        "task": "image-text-to-text",
        "model": str(snapshot),
        "dtype": torch.bfloat16,
    }
    if precision == "bf16":
        kwargs["device"] = 0
    elif precision == "int8":
        kwargs["device_map"] = "auto"
        kwargs["model_kwargs"] = {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }
    elif precision == "int4":
        kwargs["device_map"] = "auto"
        kwargs["model_kwargs"] = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        }
    else:
        raise ValueError(precision)
    return pipeline(**kwargs)


def summarize(values):
    vals = [float(x) for x in values]
    return {
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "cv": (statistics.stdev(vals) / statistics.mean(vals)) if len(vals) > 1 and statistics.mean(vals) else None,
    }


def main() -> None:
    helper = load_helper()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file():
        raise RuntimeError("Frozen 100-case Open-I manifest missing")
    all_cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(all_cases) != 100:
        raise RuntimeError(f"Expected 100 frozen cases, found {len(all_cases)}")
    cases = all_cases[:PILOT_CASES]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    bnb = ensure_bitsandbytes()
    refs = helper.extract_report_texts()
    snapshot = pathlib.Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(ROOT / ".wp3-models" / "MedGemma-4B-pinned"),
    )).resolve()

    compatibility = []
    blocks = []
    case_rows = []
    total = len(PRECISIONS) * REPEATS
    done = 0

    for precision in PRECISIONS:
        if precision != "bf16" and bnb.get("status") != "available":
            compatibility.append({"precision": precision, "status": "skipped", "error": "bitsandbytes unavailable"})
            done += REPEATS
            progress(done, total, f"Skipped {precision}: bitsandbytes unavailable")
            continue

        pipe = None
        try:
            t_load = time.perf_counter()
            pipe = make_pipeline(snapshot, precision)
            load_seconds = time.perf_counter() - t_load
            warm_img = Image.open(ROOT / cases[0]["local_image_path"]).convert("RGB")
            warm_out = pipe(helper.make_messages(warm_img), max_new_tokens=16, do_sample=False)
            warm_text = helper.extract_generated_text(warm_out)
            if not warm_text:
                raise RuntimeError("Empty warmup output")
            torch.cuda.synchronize()
            compatibility.append({"precision": precision, "status": "ok", "model_load_seconds": load_seconds, "error": ""})
            del warm_img, warm_out

            for repeat in range(1, REPEATS + 1):
                idle_rows = []
                idle_stop = threading.Event()
                idle_thread = threading.Thread(target=helper.sample_trace, args=(idle_stop, idle_rows), daemon=True)
                idle_thread.start()
                time.sleep(IDLE_SECONDS)
                idle_stop.set(); idle_thread.join()
                if not idle_rows:
                    raise RuntimeError("No idle power samples")
                idle_mean = statistics.mean(r[1] for r in idle_rows)

                trace = []
                stop = threading.Event()
                sampler = threading.Thread(target=helper.sample_trace, args=(stop, trace), daemon=True)
                sampler.start()
                block_start = time.perf_counter()
                elapsed_cases = []
                for case in cases:
                    findings, impression = refs.get(case["source_report_id"], ("", ""))
                    reference = " ".join(x for x in (findings, impression) if x).strip()
                    img = Image.open(ROOT / case["local_image_path"]).convert("RGB")
                    t0 = time.perf_counter()
                    out = pipe(helper.make_messages(img), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    pred = helper.extract_generated_text(out)
                    if not pred:
                        raise RuntimeError(f"Empty output for {precision} case {case['case_index']}")
                    elapsed_cases.append(elapsed)
                    case_rows.append({
                        "precision": precision,
                        "repeat": repeat,
                        "case_index": int(case["case_index"]),
                        "source_report_id": case["source_report_id"],
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
                stop.set(); sampler.join()
                gross_wh = helper.integrate_wh(trace)
                net_wh = max(0.0, gross_wh - idle_mean * block_elapsed / 3600.0)
                blocks.append({
                    "precision": precision,
                    "repeat": repeat,
                    "cases": len(cases),
                    "gross_gpu_energy_wh_block": gross_wh,
                    "gross_gpu_energy_wh_per_case": gross_wh / len(cases),
                    "net_gpu_energy_wh_block": net_wh,
                    "net_gpu_energy_wh_per_case": net_wh / len(cases),
                    "idle_mean_power_w": idle_mean,
                    "block_elapsed_seconds": block_elapsed,
                    "median_case_elapsed_seconds": statistics.median(elapsed_cases),
                    "mean_gpu_utilization_pct": statistics.mean(r[2] for r in trace) if trace else None,
                    "peak_sampled_memory_mib": max(r[3] for r in trace) if trace else None,
                })
                done += 1
                progress(done, total, f"Completed {precision} repeat {repeat}/{REPEATS}")
        except Exception as exc:
            compatibility.append({"precision": precision, "status": "failed", "model_load_seconds": None, "error": f"{type(exc).__name__}: {exc}"})
            already = sum(1 for r in blocks if r["precision"] == precision)
            done += max(0, REPEATS - already)
            progress(done, total, f"{precision} failed: {type(exc).__name__}")
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    with (OUT_DIR / "compatibility.json").open("w", encoding="utf-8") as f:
        json.dump({"bitsandbytes": bnb, "precisions": compatibility}, f, indent=2)
        f.write("\n")

    if blocks:
        with (OUT_DIR / "block_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(blocks[0].keys())); w.writeheader(); w.writerows(blocks)
    else:
        (OUT_DIR / "block_summary.csv").write_text("precision,repeat,cases\n", encoding="utf-8")

    if case_rows:
        with (OUT_DIR / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys())); w.writeheader(); w.writerows(case_rows)
    else:
        (OUT_DIR / "case_results.csv").write_text("precision,repeat,case_index\n", encoding="utf-8")

    summaries = {}
    for precision in PRECISIONS:
        bs = [r for r in blocks if r["precision"] == precision]
        cs = [r for r in case_rows if r["precision"] == precision]
        if not bs:
            continue
        summaries[precision] = {
            "successful_repeats": len(bs),
            "gross_wh_per_case": summarize(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "net_wh_per_case": summarize(r["net_gpu_energy_wh_per_case"] for r in bs),
            "median_case_seconds": summarize(r["median_case_elapsed_seconds"] for r in bs),
            "mean_unigram_f1": statistics.mean(float(r["unigram_f1"]) for r in cs) if cs else None,
            "mean_rouge_l_f1": statistics.mean(float(r["rouge_l_f1"]) for r in cs) if cs else None,
            "peak_sampled_memory_mib_max": max(float(r["peak_sampled_memory_mib"]) for r in bs if r["peak_sampled_memory_mib"] not in (None, "")),
        }

    bf16 = summaries.get("bf16", {}).get("gross_wh_per_case", {}).get("mean")
    for precision in ("int8", "int4"):
        q = summaries.get(precision, {}).get("gross_wh_per_case", {}).get("mean")
        if bf16 and q is not None:
            summaries[precision]["gross_energy_reduction_vs_bf16"] = 1.0 - (q / bf16)

    report = {
        "status": "WP3_MEDGEMMA_QUANTIZATION_PILOT_COMPLETE",
        "model": "MedGemma-4B",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "pilot_case_indices": [int(r["case_index"]) for r in cases],
        "cases_per_repeat": PILOT_CASES,
        "repeats_per_precision": REPEATS,
        "precisions_requested": list(PRECISIONS),
        "bitsandbytes": bnb,
        "compatibility": compatibility,
        "summary": summaries,
        "measurement_scope": "Direct NVIDIA GPU board operational energy; model loading and warmup excluded; gross primary and idle-adjusted net secondary.",
        "interpretation_limit": "This is a 10-case mitigation compatibility/repeatability pilot. Utility is lexical screening only; no clinical or RadGraph non-inferiority claim should be made from this pilot.",
        "next_step_rule": "Advance a quantized precision to the frozen 100-case evaluation only if all 3 repeats complete and energy decreases without obvious output failure; final utility must be evaluated with F1-RadGraph on the full cohort.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WP3_MEDGEMMA_QUANTIZATION_PILOT_COMPLETE")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
