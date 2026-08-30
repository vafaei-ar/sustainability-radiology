from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import subprocess
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_cxr_pilot" / "case_manifest.csv"
REPORTS_TGZ = ROOT / ".wp3-data" / "openi_cxr_pilot" / "NLMCXR_reports.tgz"
OUT_DIR = ROOT / "results" / "wp3" / "internvl3_openi_pilot"
MODEL_ID = "OpenGVLab/InternVL3-8B"
MODEL_CACHE = ROOT / ".wp3-models" / "InternVL3-8B"
PROMPT = "<image>\nDescribe the chest radiograph findings concisely. Do not infer patient identity."
MAX_NEW_TOKENS = 128
MAX_TILES = 6
SAMPLE_INTERVAL_S = 0.2
IDLE_SECONDS = 10.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    return visible.split(",")[0].strip() if visible else "0"


def query_gpu(gpu_id: str) -> tuple[float, float, float]:
    raw = subprocess.check_output([
        "nvidia-smi", f"--id={gpu_id}",
        "--query-gpu=power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    return float(raw[0].strip()), float(raw[1].strip()), float(raw[2].strip())


def integrate_energy_wh(samples: list[dict[str, float]]) -> float:
    joules = 0.0
    for left, right in zip(samples[:-1], samples[1:]):
        dt = right["t_s"] - left["t_s"]
        joules += ((left["power_w"] + right["power_w"]) / 2.0) * dt
    return joules / 3600.0


def sample_for_duration(gpu_id: str, seconds: float) -> list[dict[str, float]]:
    samples = []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        p, u, m = query_gpu(gpu_id)
        samples.append({"t_s": now - t0, "power_w": p, "utilization_pct": u, "memory_used_mib": m})
        if now - t0 >= seconds:
            break
        time.sleep(SAMPLE_INTERVAL_S)
    return samples


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        images.append(resized.crop(box))
    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


def load_image(path: pathlib.Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = build_transform(448)
    tiles = dynamic_preprocess(image, max_num=MAX_TILES, image_size=448, use_thumbnail=True)
    return torch.stack([transform(tile) for tile in tiles])


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def unigram_f1(pred: str, ref: str) -> float:
    p, r = normalize_tokens(pred), normalize_tokens(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
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
    if not lcs:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_report_texts() -> dict[str, tuple[str, str]]:
    out = {}
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
                if label == "FINDINGS": findings.append(elem.text.strip())
                if label == "IMPRESSION": impression.append(elem.text.strip())
            out[pathlib.PurePosixPath(member.name).stem] = (" ".join(findings), " ".join(impression))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(cases) != 10:
        raise RuntimeError(f"Expected 10 frozen Open-I cases, found {len(cases)}")
    refs = extract_report_texts()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False, cache_dir=MODEL_CACHE)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        cache_dir=MODEL_CACHE,
    ).eval().cuda()
    generation_config = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "num_beams": 1}

    # Phase 0 compatibility preflight on one frozen case. This also serves as warmup and is excluded from energy measurement.
    warm_pixels = load_image(ROOT / cases[0]["local_image_path"]).to(torch.bfloat16).cuda()
    with torch.inference_mode():
        warm_response = model.chat(tokenizer, warm_pixels, PROMPT, generation_config)
    torch.cuda.synchronize()
    if not isinstance(warm_response, str) or not warm_response.strip():
        raise RuntimeError("InternVL3 preflight returned an empty/non-string response")
    torch.cuda.reset_peak_memory_stats(0)
    report_progress(0, 10, "InternVL3 preflight passed")

    gpu_id = physical_gpu_id()
    idle_samples = sample_for_duration(gpu_id, IDLE_SECONDS)
    idle_mean_power = sum(x["power_w"] for x in idle_samples) / len(idle_samples)

    samples = []
    stop = threading.Event()
    t0 = time.perf_counter()
    def sampler():
        while not stop.is_set():
            now = time.perf_counter()
            p, u, m = query_gpu(gpu_id)
            samples.append({"t_s": now - t0, "power_w": p, "utilization_pct": u, "memory_used_mib": m})
            stop.wait(SAMPLE_INTERVAL_S)
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()

    rows = []
    try:
        for i, case in enumerate(cases, start=1):
            findings, impression = refs.get(case["source_report_id"], ("", ""))
            reference = " ".join(x for x in [findings, impression] if x).strip()
            if not reference:
                raise RuntimeError(f"Missing reference text for {case['source_report_id']}")
            pixels = load_image(ROOT / case["local_image_path"]).to(torch.bfloat16).cuda()
            tile_count = int(pixels.shape[0])
            start = time.perf_counter()
            with torch.inference_mode():
                response = model.chat(tokenizer, pixels, PROMPT, generation_config)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            output_tokens = len(tokenizer.encode(response, add_special_tokens=False))
            rows.append({
                "case_index": int(case["case_index"]),
                "source_report_id": case["source_report_id"],
                "source_image_id": case["source_image_id"],
                "tile_count": tile_count,
                "elapsed_seconds": elapsed,
                "generated_tokens_approx": output_tokens,
                "model_word_count": len(normalize_tokens(response)),
                "unigram_f1": unigram_f1(response, reference),
                "rouge_l_f1": rouge_l_f1(response, reference),
                "model_output": response,
                "reference_findings": findings,
                "reference_impression": impression,
            })
            report_progress(i, 10, "InternVL3 Open-I pilot")
    finally:
        stop.set(); thread.join(timeout=2.0)
    p, u, m = query_gpu(gpu_id)
    samples.append({"t_s": time.perf_counter() - t0, "power_w": p, "utilization_pct": u, "memory_used_mib": m})

    gross_wh = integrate_energy_wh(samples)
    block_seconds = samples[-1]["t_s"] if samples else 0.0
    net_wh = max(0.0, gross_wh - idle_mean_power * block_seconds / 3600.0)

    with (OUT_DIR / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT_DIR / "gpu_trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"]); w.writeheader(); w.writerows(samples)
    with (OUT_DIR / "idle_trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "power_w", "utilization_pct", "memory_used_mib"]); w.writeheader(); w.writerows(idle_samples)

    report = {
        "status": "WP3_INTERNVL3_OPENI_PILOT_OK",
        "model_id": MODEL_ID,
        "precision": "BF16",
        "task": "image_description",
        "cases": len(rows),
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_tiles": MAX_TILES,
        "model_load_excluded": True,
        "preflight_warmup_excluded": True,
        "sampling_interval_seconds": SAMPLE_INTERVAL_S,
        "idle_seconds": IDLE_SECONDS,
        "idle_mean_power_w": idle_mean_power,
        "gross_gpu_energy_wh_block": gross_wh,
        "net_gpu_energy_wh_block": net_wh,
        "gross_gpu_energy_wh_per_case": gross_wh / len(rows),
        "net_gpu_energy_wh_per_case": net_wh / len(rows),
        "block_elapsed_seconds": block_seconds,
        "median_case_elapsed_seconds": sorted(float(r["elapsed_seconds"]) for r in rows)[len(rows)//2],
        "mean_unigram_f1": sum(float(r["unigram_f1"]) for r in rows) / len(rows),
        "mean_rouge_l_f1": sum(float(r["rouge_l_f1"]) for r in rows) / len(rows),
        "median_model_word_count": sorted(int(r["model_word_count"]) for r in rows)[len(rows)//2],
        "median_tile_count": sorted(int(r["tile_count"]) for r in rows)[len(rows)//2],
        "peak_vram_mib_torch": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
        "mean_gpu_utilization_pct": sum(s["utilization_pct"] for s in samples) / len(samples),
        "device_name": torch.cuda.get_device_name(0),
        "physical_gpu_query_id": gpu_id,
        "interpretation": "Combined compatibility preflight and 10-case prospective Open-I pilot. Energy is direct NVIDIA GPU board-energy integration with an idle baseline. Lexical agreement is screening only and does not establish clinical adequacy.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
