from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_cxr_pilot" / "case_manifest.csv"
REPORTS_TGZ = ROOT / ".wp3-data" / "openi_cxr_pilot" / "NLMCXR_reports.tgz"
OUT_DIR = ROOT / "results" / "wp3" / "openi_qwen_prompt_sensitivity"
MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 128

PROMPTS = {
    "generic": "Describe the chest radiograph findings concisely. Do not infer patient identity.",
    "report_structured": (
        "Write a concise chest radiograph report with exactly two sections: FINDINGS and IMPRESSION. "
        "Describe only visible findings and clinically relevant negatives. Do not infer patient demographics, "
        "history, or identity. Keep the total response under 80 words."
    ),
    "report_terse": (
        "Report this chest radiograph in no more than 50 words. State the key positive findings and major "
        "pertinent negatives, then give a one-sentence impression. Do not speculate beyond the image or infer identity."
    ),
}


def report_progress(current: int, total: int) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": "Open-I prompt sensitivity",
        "unit": "case-prompt generations",
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
    pc, rc = Counter(p), Counter(r)
    overlap = sum((pc & rc).values())
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


def prepare_inputs(processor, image_path: pathlib.Path, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda:0")


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(cases) != 10:
        raise RuntimeError(f"Expected 10 frozen cases, found {len(cases)}")
    refs = extract_report_texts()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=MODEL_DIR,
    ).eval()
    # Avoid irrelevant sampling warnings in deterministic generation.
    model.generation_config.temperature = None

    rows: list[dict[str, object]] = []
    total = len(cases) * len(PROMPTS)
    completed = 0
    for prompt_id, prompt in PROMPTS.items():
        for case in cases:
            findings, impression = refs.get(case["source_report_id"], ("", ""))
            reference = " ".join(x for x in [findings, impression] if x).strip()
            if not reference:
                raise RuntimeError(f"Missing reference text for {case['source_report_id']}")
            inputs = prepare_inputs(processor, ROOT / case["local_image_path"], prompt)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
            torch.cuda.synchronize()
            new_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
            trimmed = generated[:, inputs["input_ids"].shape[-1]:]
            pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            rows.append({
                "prompt_id": prompt_id,
                "case_index": int(case["case_index"]),
                "source_report_id": case["source_report_id"],
                "source_image_id": case["source_image_id"],
                "reference_findings": findings,
                "reference_impression": impression,
                "model_output": pred,
                "reference_word_count": len(normalize_tokens(reference)),
                "model_word_count": len(normalize_tokens(pred)),
                "generated_tokens": new_tokens,
                "hit_max_new_tokens": new_tokens >= MAX_NEW_TOKENS,
                "unigram_f1": unigram_f1(pred, reference),
                "rouge_l_f1": rouge_l_f1(pred, reference),
            })
            completed += 1
            report_progress(completed, total)

    case_path = OUT_DIR / "case_prompt_results.csv"
    with case_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for prompt_id, prompt in PROMPTS.items():
        subset = [r for r in rows if r["prompt_id"] == prompt_id]
        uf1 = [float(r["unigram_f1"]) for r in subset]
        rf1 = [float(r["rouge_l_f1"]) for r in subset]
        words = [float(r["model_word_count"]) for r in subset]
        toks = [float(r["generated_tokens"]) for r in subset]
        summary_rows.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "cases": len(subset),
            "mean_unigram_f1": sum(uf1) / len(uf1),
            "median_unigram_f1": median(uf1),
            "mean_rouge_l_f1": sum(rf1) / len(rf1),
            "median_rouge_l_f1": median(rf1),
            "median_model_word_count": median(words),
            "median_generated_tokens": median(toks),
            "cases_hitting_max_new_tokens": sum(bool(r["hit_max_new_tokens"]) for r in subset),
        })

    summary_path = OUT_DIR / "prompt_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    ranked = sorted(summary_rows, key=lambda r: (float(r["mean_rouge_l_f1"]), float(r["mean_unigram_f1"])), reverse=True)
    report = {
        "status": "WP3_OPENI_QWEN_PROMPT_SENSITIVITY_OK",
        "model_id": MODEL_ID,
        "precision": "BF16",
        "cases": len(cases),
        "prompts": len(PROMPTS),
        "generations": len(rows),
        "max_new_tokens": MAX_NEW_TOKENS,
        "lexical_ranked_prompt_ids": [r["prompt_id"] for r in ranked],
        "interpretation": (
            "Prompt sensitivity screen on the fixed 10-case Open-I pilot. Lexical overlap is not a clinical-quality endpoint. "
            "Use these results to avoid freezing a clearly poorly aligned prompt, then confirm the selected prompt with structured clinical review before scaling."
        ),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])
    print(json.dumps({"ranked_prompt_ids": report["lexical_ranked_prompt_ids"], "summary": summary_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
