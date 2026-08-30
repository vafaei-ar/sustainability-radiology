from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import subprocess
import threading
import time

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "wp3" / "qwen_smoke_pilot"
MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
GPU_INDEX = 0
SAMPLE_INTERVAL_S = 0.2
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


def integrate_energy_kwh(samples: list[dict[str, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    joules = 0.0
    for left, right in zip(samples[:-1], samples[1:]):
        dt = right["t_s"] - left["t_s"]
        mean_w = (left["power_w"] + right["power_w"]) / 2.0
        joules += mean_w * dt
    return joules / 3_600_000.0


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUT_DIR / "synthetic_test.png"
    make_synthetic_image(image_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the WP3 project runtime")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": GPU_INDEX},
        cache_dir=MODEL_DIR,
    ).eval()

    inputs = prepare_inputs(processor, image_path)

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=16, do_sample=False, num_beams=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(GPU_INDEX)

    samples: list[dict[str, float]] = []
    stop = threading.Event()
    t0 = time.perf_counter()

    def sampler() -> None:
        while not stop.is_set():
            now = time.perf_counter()
            power_w, util_pct, mem_mib = query_gpu()
            samples.append(
                {
                    "t_s": now - t0,
                    "power_w": power_w,
                    "utilization_pct": util_pct,
                    "memory_used_mib": mem_mib,
                }
            )
            stop.wait(SAMPLE_INTERVAL_S)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join(timeout=2.0)
    power_w, util_pct, mem_mib = query_gpu()
    samples.append(
        {
            "t_s": time.perf_counter() - t0,
            "power_w": power_w,
            "utilization_pct": util_pct,
            "memory_used_mib": mem_mib,
        }
    )

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    new_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    trimmed = generated[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    energy_kwh = integrate_energy_kwh(samples)

    trace_path = OUT_DIR / "gpu_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"])
        writer.writeheader()
        writer.writerows(samples)

    report = {
        "status": "WP3_QWEN_SMOKE_PILOT_OK",
        "model_id": MODEL_ID,
        "precision": "BF16",
        "device_name": torch.cuda.get_device_name(GPU_INDEX),
        "gpu_index": GPU_INDEX,
        "synthetic_nonclinical_input": True,
        "model_load_excluded": True,
        "warmup_excluded": True,
        "sampling_interval_seconds": SAMPLE_INTERVAL_S,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": new_tokens,
        "tokens_per_second": new_tokens / elapsed if elapsed else None,
        "peak_vram_mib_torch": torch.cuda.max_memory_allocated(GPU_INDEX) / (1024 * 1024),
        "gpu_energy_kwh_nvidia_smi_integrated": energy_kwh,
        "gpu_energy_wh_nvidia_smi_integrated": energy_kwh * 1000.0,
        "mean_power_w": sum(s["power_w"] for s in samples) / len(samples) if samples else None,
        "mean_gpu_utilization_pct": sum(s["utilization_pct"] for s in samples) / len(samples) if samples else None,
        "generated_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "generated_text": text,
        "interpretation": "Runtime and measurement smoke test only. Synthetic input is not a clinical case and must not be included in scientific model-performance comparisons.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
