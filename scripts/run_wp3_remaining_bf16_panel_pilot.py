from __future__ import annotations

import csv
import gc
import json
import os
import pathlib
import re
import statistics
import subprocess
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter

import torch
from PIL import Image
from huggingface_hub import HfApi, snapshot_download
from transformers import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model" / "case_manifest_100.csv"
REPORTS_TGZ = ROOT / ".wp3-data" / "openi_cxr_pilot" / "NLMCXR_reports.tgz"
OUT_DIR = ROOT / "results" / "wp3" / "remaining_bf16_panel_pilot"
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."
MAX_NEW_TOKENS = 128
PILOT_CASES = 10
IDLE_SECONDS = 10.0
SAMPLE_INTERVAL = 0.2
MODELS = [
    {"short_name": "MedGemma-4B", "repo_id": "google/medgemma-4b-it"},
    {"short_name": "LLaVA-Med-v1.5", "repo_id": "microsoft/llava-med-v1.5-mistral-7b"},
]


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
        "unit": "model stages",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def unigram_f1(pred: str, ref: str) -> float:
    p, r = normalize_tokens(pred), normalize_tokens(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    p, r = normalize_tokens(pred), normalize_tokens(ref)
    if not p or not r:
        return 0.0
    lcs = lcs_len(p, r)
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_report_texts() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with tarfile.open(REPORTS_TGZ, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                root = ET.fromstring(f.read())
            except ET.ParseError:
                continue
            findings, impression = [], []
            for elem in root.iter():
                if not elem.tag.endswith("AbstractText") or not elem.text:
                    continue
                label = (elem.attrib.get("Label") or "").upper()
                if label == "FINDINGS":
                    findings.append(elem.text.strip())
                elif label == "IMPRESSION":
                    impression.append(elem.text.strip())
            out[pathlib.PurePosixPath(member.name).stem] = (" ".join(findings), " ".join(impression))
    return out


def query_power() -> tuple[float, float, float]:
    raw = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits", "--id=0",
        ],
        text=True,
    ).strip().splitlines()[0]
    p, u, m = [float(x.strip()) for x in raw.split(",")]
    return p, u, m


def sample_trace(stop: threading.Event, rows: list[tuple[float, float, float, float]]) -> None:
    while not stop.is_set():
        ts = time.perf_counter()
        try:
            p, u, m = query_power()
            rows.append((ts, p, u, m))
        except Exception:
            pass
        stop.wait(SAMPLE_INTERVAL)


def integrate_wh(rows: list[tuple[float, float, float, float]]) -> float:
    if len(rows) < 2:
        raise RuntimeError("Insufficient GPU power samples")
    total_ws = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        dt = b[0] - a[0]
        total_ws += 0.5 * (a[1] + b[1]) * dt
    return total_ws / 3600.0


def extract_generated_text(output) -> str:
    obj = output
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if isinstance(obj, dict):
        obj = obj.get("generated_text", obj.get("text", obj))
    if isinstance(obj, list):
        # Chat-style output is often a list of role/content messages.
        for item in reversed(obj):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for x in content:
                        if isinstance(x, dict) and isinstance(x.get("text"), str):
                            parts.append(x["text"])
                    if parts:
                        return " ".join(parts).strip()
        return str(obj)
    return str(obj).strip()


def make_messages(image: Image.Image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def run_model(model_spec: dict[str, str], cases: list[dict[str, str]], refs: dict[str, tuple[str, str]]) -> tuple[dict, list[dict]]:
    short = model_spec["short_name"]
    repo_id = model_spec["repo_id"]
    result = {
        "short_name": short,
        "repo_id": repo_id,
        "status": "not_started",
        "revision": None,
        "access_or_runtime_error": None,
    }
    case_rows: list[dict] = []
    api = HfApi()
    try:
        revision = api.model_info(repo_id).sha
        result["revision"] = revision
        snapshot = pathlib.Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(ROOT / ".wp3-models" / f"{short}-pinned"),
        )).resolve()
    except Exception as exc:
        result["status"] = "access_failed"
        result["access_or_runtime_error"] = f"{type(exc).__name__}: {exc}"[:1200]
        return result, case_rows

    try:
        pipe = pipeline(
            "image-text-to-text",
            model=str(snapshot),
            device=0,
            dtype=torch.bfloat16,
        )
        # Warmup first frozen case. Excluded from measured block.
        warm_img = Image.open(ROOT / cases[0]["local_image_path"]).convert("RGB")
        warm_out = pipe(make_messages(warm_img), max_new_tokens=16, do_sample=False)
        if not extract_generated_text(warm_out):
            raise RuntimeError("Empty warmup output")
        del warm_img, warm_out
        torch.cuda.synchronize()

        idle_rows: list[tuple[float, float, float, float]] = []
        idle_stop = threading.Event()
        idle_thread = threading.Thread(target=sample_trace, args=(idle_stop, idle_rows), daemon=True)
        idle_thread.start()
        time.sleep(IDLE_SECONDS)
        idle_stop.set(); idle_thread.join()
        if not idle_rows:
            raise RuntimeError("No idle GPU power samples")
        idle_mean_power = statistics.mean(r[1] for r in idle_rows)

        trace: list[tuple[float, float, float, float]] = []
        stop = threading.Event()
        sampler = threading.Thread(target=sample_trace, args=(stop, trace), daemon=True)
        sampler.start()
        block_start = time.perf_counter()
        for case in cases:
            findings, impression = refs.get(case["source_report_id"], ("", ""))
            reference = " ".join(x for x in (findings, impression) if x).strip()
            img = Image.open(ROOT / case["local_image_path"]).convert("RGB")
            t0 = time.perf_counter()
            out = pipe(make_messages(img), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            pred = extract_generated_text(out)
            if not pred:
                raise RuntimeError(f"Empty output for case {case['case_index']}")
            case_rows.append({
                "model": short,
                "repo_id": repo_id,
                "revision": revision,
                "case_index": int(case["case_index"]),
                "source_report_id": case["source_report_id"],
                "source_image_id": case["source_image_id"],
                "elapsed_seconds": elapsed,
                "model_word_count": len(normalize_tokens(pred)),
                "unigram_f1": unigram_f1(pred, reference),
                "rouge_l_f1": rouge_l_f1(pred, reference),
                "model_output": pred,
                "reference_findings": findings,
                "reference_impression": impression,
            })
            del img, out
        block_elapsed = time.perf_counter() - block_start
        stop.set(); sampler.join()
        gross_wh = integrate_wh(trace)
        net_wh = max(0.0, gross_wh - idle_mean_power * block_elapsed / 3600.0)
        result.update({
            "status": "pilot_ok",
            "cases": len(cases),
            "gross_gpu_energy_wh_block": gross_wh,
            "gross_gpu_energy_wh_per_case": gross_wh / len(cases),
            "net_gpu_energy_wh_block": net_wh,
            "net_gpu_energy_wh_per_case": net_wh / len(cases),
            "idle_mean_power_w": idle_mean_power,
            "block_elapsed_seconds": block_elapsed,
            "median_case_elapsed_seconds": statistics.median(r["elapsed_seconds"] for r in case_rows),
            "mean_unigram_f1": statistics.mean(r["unigram_f1"] for r in case_rows),
            "mean_rouge_l_f1": statistics.mean(r["rouge_l_f1"] for r in case_rows),
            "peak_sampled_memory_mib": max(r[3] for r in trace) if trace else None,
            "mean_gpu_utilization_pct": statistics.mean(r[2] for r in trace) if trace else None,
            "measurement_scope": "Direct NVIDIA GPU board operational energy; model loading and warmup excluded; gross primary and idle-adjusted net secondary.",
        })
    except Exception as exc:
        result["status"] = "runtime_failed"
        result["access_or_runtime_error"] = f"{type(exc).__name__}: {exc}"[:1200]
    finally:
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return result, case_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file() or not REPORTS_TGZ.is_file():
        raise RuntimeError("Frozen 100-case manifest or Open-I report archive missing")
    all_cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(all_cases) != 100:
        raise RuntimeError(f"Expected frozen 100-case manifest, found {len(all_cases)} rows")
    cases = all_cases[:PILOT_CASES]
    refs = extract_report_texts()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    summaries = []
    all_case_rows: list[dict] = []
    progress(0, len(MODELS) * 2, "Starting remaining BF16 panel")
    for idx, spec in enumerate(MODELS):
        progress(idx * 2, len(MODELS) * 2, f"Resolving {spec['short_name']}")
        summary, rows = run_model(spec, cases, refs)
        summaries.append(summary)
        all_case_rows.extend(rows)
        progress(idx * 2 + 2, len(MODELS) * 2, f"Finished {spec['short_name']}")

    with (OUT_DIR / "model_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in summaries for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summaries)

    if all_case_rows:
        with (OUT_DIR / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_case_rows[0].keys()))
            w.writeheader(); w.writerows(all_case_rows)
    else:
        (OUT_DIR / "case_results.csv").write_text("model,case_index\n", encoding="utf-8")

    report = {
        "status": "WP3_REMAINING_BF16_PANEL_PILOT_COMPLETE",
        "prompt": PROMPT,
        "pilot_cases": PILOT_CASES,
        "models": summaries,
        "interpretation": "Compatibility and 10-case technical screening for the remaining prospective BF16 panel. Lexical metrics are screening only. A model that fails access or runtime is reported without aborting the other model.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
