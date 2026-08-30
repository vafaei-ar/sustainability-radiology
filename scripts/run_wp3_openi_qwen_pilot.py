from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_cxr_pilot" / "case_manifest.csv"
OUT_DIR = ROOT / "results" / "wp3" / "openi_qwen_pilot"
MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SAMPLE_INTERVAL_S = 0.2
IDLE_SECONDS = 10.0
MAX_NEW_TOKENS = 128
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report_progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "cases",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def physical_gpu_id() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "0"
    return visible.split(",")[0].strip()


def query_gpu(gpu_id: str) -> tuple[float, float, float]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    raw = subprocess.check_output(cmd, text=True).strip().split(",")
    return float(raw[0].strip()), float(raw[1].strip()), float(raw[2].strip())


def integrate_energy_wh(samples: list[dict[str, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    joules = 0.0
    for left, right in zip(samples[:-1], samples[1:]):
        dt = right["t_s"] - left["t_s"]
        mean_w = (left["power_w"] + right["power_w"]) / 2.0
        joules += mean_w * dt
    return joules / 3600.0


def sample_for_duration(gpu_id: str, seconds: float) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        power_w, util_pct, mem_mib = query_gpu(gpu_id)
        samples.append({"t_s": now - t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
        if now - t0 >= seconds:
            break
        time.sleep(SAMPLE_INTERVAL_S)
    return samples


def load_cases() -> list[dict[str, str]]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"Missing prepared case manifest: {MANIFEST}")
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 frozen Open-I pilot cases, found {len(rows)}")
    for row in rows:
        image_path = ROOT / row["local_image_path"]
        if not image_path.is_file():
            raise RuntimeError(f"Missing project-local pilot image for case {row['case_index']}")
        if sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch for case {row['case_index']}")
    return rows


def prepare_inputs(processor, image_path: pathlib.Path):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs.to("cuda:0")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the WP3 project runtime")

    gpu_id = physical_gpu_id()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=MODEL_DIR,
    ).eval()

    warm_inputs = prepare_inputs(processor, ROOT / cases[0]["local_image_path"])
    with torch.inference_mode():
        _ = model.generate(**warm_inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    idle_samples = sample_for_duration(gpu_id, IDLE_SECONDS)
    idle_mean_power = sum(s["power_w"] for s in idle_samples) / len(idle_samples)

    block_samples: list[dict[str, float]] = []
    stop = threading.Event()
    block_t0 = time.perf_counter()

    def sampler() -> None:
        while not stop.is_set():
            now = time.perf_counter()
            power_w, util_pct, mem_mib = query_gpu(gpu_id)
            block_samples.append({"t_s": now - block_t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
            stop.wait(SAMPLE_INTERVAL_S)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    case_rows: list[dict[str, object]] = []
    try:
        for idx, case in enumerate(cases, start=1):
            image_path = ROOT / case["local_image_path"]
            inputs = prepare_inputs(processor, image_path)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            new_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            case_rows.append({
                "case_index": int(case["case_index"]),
                "source_report_id": case["source_report_id"],
                "source_image_id": case["source_image_id"],
                "image_sha256": case["image_sha256"],
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": new_tokens,
                "tokens_per_second": new_tokens / elapsed if elapsed else None,
                "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            report_progress(idx, len(cases), "Open-I Qwen pilot")
    finally:
        stop.set()
        thread.join(timeout=2.0)
    power_w, util_pct, mem_mib = query_gpu(gpu_id)
    block_samples.append({"t_s": time.perf_counter() - block_t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})

    gross_wh = integrate_energy_wh(block_samples)
    block_seconds = block_samples[-1]["t_s"] if block_samples else 0.0
    net_wh = max(0.0, gross_wh - idle_mean_power * block_seconds / 3600.0)

    with (OUT_DIR / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    with (OUT_DIR / "gpu_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(block_samples)
    with (OUT_DIR / "idle_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(idle_samples)

    report = {
        "status": "WP3_OPENI_QWEN_PILOT_OK",
        "dataset": "Open-I Indiana University Chest X-ray Collection",
        "model_id": MODEL_ID,
        "precision": "BF16",
        "task": "image_description",
        "cases": len(case_rows),
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "processor_use_fast": False,
        "do_sample": False,
        "num_beams": 1,
        "model_load_excluded": True,
        "warmup_excluded": True,
        "idle_seconds": IDLE_SECONDS,
        "idle_mean_power_w": idle_mean_power,
        "sampling_interval_seconds": SAMPLE_INTERVAL_S,
        "gross_gpu_energy_wh_block": gross_wh,
        "net_gpu_energy_wh_block": net_wh,
        "gross_gpu_energy_wh_per_case": gross_wh / len(case_rows),
        "net_gpu_energy_wh_per_case": net_wh / len(case_rows),
        "block_elapsed_seconds": block_seconds,
        "median_case_elapsed_seconds": sorted(float(r["elapsed_seconds"]) for r in case_rows)[len(case_rows) // 2],
        "peak_vram_mib_torch": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
        "mean_gpu_utilization_pct": sum(s["utilization_pct"] for s in block_samples) / len(block_samples),
        "physical_gpu_query_id": gpu_id,
        "device_name": torch.cuda.get_device_name(0),
        "interpretation": "First standardized public-radiology measurement pilot. Energy is estimated at the 10-case block level and normalized per completed case; generated text is retained only as hashes in exported artifacts.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
