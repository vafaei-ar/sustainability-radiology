from __future__ import annotations

import csv
import json
import pathlib
import statistics
import subprocess
import threading
import time

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "wp3" / "qwen_measurement_calibration"
MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
GPU_INDEX = 0
SAMPLE_INTERVAL_S = 0.2
IDLE_SECONDS = 10.0
REPLICATES = 5
CALLS_PER_REPLICATE = 10
MAX_NEW_TOKENS = 64


def make_synthetic_image(path: pathlib.Path) -> None:
    image = Image.new("L", (512, 512), color=28)
    draw = ImageDraw.Draw(image)
    draw.ellipse((110, 90, 402, 430), outline=170, width=4)
    draw.ellipse((175, 145, 337, 360), outline=105, width=3)
    draw.line((256, 80, 256, 440), fill=80, width=2)
    image.convert("RGB").save(path)


def query_gpu() -> tuple[float, float, float]:
    cmd = [
        "nvidia-smi",
        f"--id={GPU_INDEX}",
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


def prepare_inputs(processor: AutoProcessor, image_path: pathlib.Path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Describe this synthetic grayscale test image briefly. Do not infer a diagnosis."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to("cuda:0")


def sample_for_duration(seconds: float) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now - t0 > seconds:
            break
        power_w, util_pct, mem_mib = query_gpu()
        samples.append({"t_s": now - t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
        time.sleep(SAMPLE_INTERVAL_S)
    power_w, util_pct, mem_mib = query_gpu()
    samples.append({"t_s": time.perf_counter() - t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
    return samples


def measure_block(model, inputs) -> tuple[list[dict[str, float]], float, list[int]]:
    samples: list[dict[str, float]] = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def sampler() -> None:
        while not stop.is_set():
            now = time.perf_counter()
            power_w, util_pct, mem_mib = query_gpu()
            samples.append({"t_s": now - t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
            stop.wait(SAMPLE_INTERVAL_S)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    generated_lengths: list[int] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(CALLS_PER_REPLICATE):
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
            generated_lengths.append(int(generated.shape[-1] - inputs["input_ids"].shape[-1]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join(timeout=2.0)
    power_w, util_pct, mem_mib = query_gpu()
    samples.append({"t_s": time.perf_counter() - t0, "power_w": power_w, "utilization_pct": util_pct, "memory_used_mib": mem_mib})
    return samples, elapsed, generated_lengths


def write_trace(path: pathlib.Path, samples: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(samples)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUT_DIR / "synthetic_test.png"
    make_synthetic_image(image_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the WP3 project runtime")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": GPU_INDEX},
        cache_dir=MODEL_DIR,
    ).eval()
    inputs = prepare_inputs(processor, image_path)

    with torch.inference_mode():
        for _ in range(5):
            _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    torch.cuda.synchronize()

    idle_samples = sample_for_duration(IDLE_SECONDS)
    write_trace(OUT_DIR / "idle_trace.csv", idle_samples)
    idle_mean_power_w = statistics.mean(s["power_w"] for s in idle_samples)

    rows = []
    all_trace_rows = []
    for replicate in range(1, REPLICATES + 1):
        torch.cuda.reset_peak_memory_stats(GPU_INDEX)
        samples, elapsed, generated_lengths = measure_block(model, inputs)
        energy_wh = integrate_energy_wh(samples)
        baseline_wh = idle_mean_power_w * elapsed / 3600.0
        net_wh = energy_wh - baseline_wh
        mean_power = statistics.mean(s["power_w"] for s in samples)
        mean_util = statistics.mean(s["utilization_pct"] for s in samples)
        row = {
            "replicate": replicate,
            "calls": CALLS_PER_REPLICATE,
            "elapsed_seconds": elapsed,
            "gross_gpu_energy_wh": energy_wh,
            "idle_baseline_wh": baseline_wh,
            "net_gpu_energy_wh": net_wh,
            "gross_gpu_energy_wh_per_call": energy_wh / CALLS_PER_REPLICATE,
            "net_gpu_energy_wh_per_call": net_wh / CALLS_PER_REPLICATE,
            "mean_power_w": mean_power,
            "mean_gpu_utilization_pct": mean_util,
            "peak_vram_mib_torch": torch.cuda.max_memory_allocated(GPU_INDEX) / (1024 * 1024),
            "generated_tokens_total": sum(generated_lengths),
            "generated_tokens_per_call": statistics.mean(generated_lengths),
            "samples": len(samples),
        }
        rows.append(row)
        for sample in samples:
            all_trace_rows.append({"replicate": replicate, **sample})

    with (OUT_DIR / "replicates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "gpu_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["replicate", "t_s", "power_w", "utilization_pct", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(all_trace_rows)

    net_values = [r["net_gpu_energy_wh_per_call"] for r in rows]
    gross_values = [r["gross_gpu_energy_wh_per_call"] for r in rows]
    report = {
        "status": "WP3_QWEN_MEASUREMENT_CALIBRATION_OK",
        "model_id": MODEL_ID,
        "precision": "BF16",
        "device_name": torch.cuda.get_device_name(GPU_INDEX),
        "synthetic_nonclinical_input": True,
        "model_load_excluded": True,
        "warmup_excluded": True,
        "processor_use_fast": False,
        "sampling_interval_seconds": SAMPLE_INTERVAL_S,
        "idle_seconds": IDLE_SECONDS,
        "idle_mean_power_w": idle_mean_power_w,
        "replicates": REPLICATES,
        "calls_per_replicate": CALLS_PER_REPLICATE,
        "gross_energy_wh_per_call_median": statistics.median(gross_values),
        "net_energy_wh_per_call_median": statistics.median(net_values),
        "net_energy_wh_per_call_mean": statistics.mean(net_values),
        "net_energy_wh_per_call_stdev": statistics.stdev(net_values),
        "net_energy_wh_per_call_cv": statistics.stdev(net_values) / statistics.mean(net_values),
        "interpretation": "Measurement-method calibration only. Synthetic input is not a clinical case and these values must not be used for clinical performance or population extrapolation.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
