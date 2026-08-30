from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import tarfile
import time
import xml.etree.ElementTree as ET

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_cxr_pilot" / "case_manifest.csv"
REPORTS_TGZ = ROOT / ".wp3-data" / "openi_cxr_pilot" / "NLMCXR_reports.tgz"
OUT_DIR = ROOT / "results" / "wp3" / "openi_qwen_output_validation"
MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."
MAX_NEW_TOKENS = 128


def report_progress(current: int, total: int) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    payload = {"schema_version": 1, "current": current, "total": total, "fraction": current / total, "phase": "Open-I output validation", "unit": "cases", "updated_at_epoch": time.time()}
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
    from collections import Counter
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
                if label == "FINDINGS": findings.append(elem.text.strip())
                if label == "IMPRESSION": impression.append(elem.text.strip())
            out[pathlib.PurePosixPath(member.name).stem] = (" ".join(findings), " ".join(impression))
    return out


def prepare_inputs(processor, image_path: pathlib.Path):
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda:0")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if not cases:
        raise RuntimeError("Frozen manifest contains no cases")
    case_indices = [int(case["case_index"]) for case in cases]
    if case_indices != list(range(1, len(cases) + 1)):
        raise RuntimeError("Frozen manifest case_index values must be contiguous starting at 1")
    refs = extract_report_texts()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0}, cache_dir=MODEL_DIR).eval()

    rows = []
    for i, case in enumerate(cases, start=1):
        findings, impression = refs.get(case["source_report_id"], ("", ""))
        reference = " ".join(x for x in [findings, impression] if x).strip()
        if not reference:
            raise RuntimeError(f"Missing reference text for {case['source_report_id']}")
        inputs = prepare_inputs(processor, ROOT / case["local_image_path"])
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        torch.cuda.synchronize()
        new_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        pred = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        rows.append({
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
            "manual_adequacy": "",
            "manual_major_error": "",
            "manual_omission": "",
            "manual_comments": "",
        })
        report_progress(i, len(cases))

    fields = list(rows[0].keys())
    with (OUT_DIR / "case_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "status": "WP3_OPENI_QWEN_OUTPUT_VALIDATION_OK",
        "cases": len(rows),
        "model_id": MODEL_ID,
        "precision": "BF16",
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "mean_unigram_f1": sum(float(r["unigram_f1"]) for r in rows) / len(rows),
        "mean_rouge_l_f1": sum(float(r["rouge_l_f1"]) for r in rows) / len(rows),
        "median_model_word_count": sorted(int(r["model_word_count"]) for r in rows)[len(rows)//2],
        "cases_hitting_max_new_tokens": sum(bool(r["hit_max_new_tokens"]) for r in rows),
        "interpretation": "Automated lexical agreement is a screening measure only and does not establish clinical adequacy. case_review.csv includes public Open-I reference text, generated output, and blank structured reviewer fields for clinical review before scaling.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
