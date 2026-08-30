from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import shutil
import statistics
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter

from huggingface_hub import snapshot_download

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".wp3-data" / "openi_cxr_pilot"
REPORTS_TGZ = DATA_DIR / "NLMCXR_reports.tgz"
IMAGES_TGZ = DATA_DIR / "NLMCXR_png.tgz"
IMAGE_DIR = DATA_DIR / "images_100_single"
OUT_DIR = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
MANIFEST = OUT_DIR / "case_manifest_100.csv"
QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
INTERNVL_ID = "OpenGVLab/InternVL3-8B"
INTERNVL_REVISION = "dab7194eaadae9ff191fef49b961847a18b4c822"
N_CASES = 100
BLOCK_SIZE = 10
SEED = 20260830
TOP_NONNORMAL_LABELS = 10


def progress(current: int, total: int, phase: str, unit: str = "stages") -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": unit,
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


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def abstract_text(root: ET.Element, label: str) -> str:
    parts: list[str] = []
    target = label.upper()
    for elem in root.iter():
        if elem.tag.endswith("AbstractText") and (elem.attrib.get("Label") or "").upper() == target and elem.text:
            parts.append(elem.text.strip())
    return " ".join(x for x in parts if x)


def image_ids(root: ET.Element) -> list[str]:
    vals: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("parentImage"):
            value = (elem.attrib.get("id") or "").strip()
            if value:
                vals.append(value)
    return vals


def mesh_labels(root: ET.Element) -> list[str]:
    vals: list[str] = []
    for elem in root.iter():
        local = elem.tag.rsplit("}", 1)[-1].lower()
        if local in {"major", "minor"} and elem.text:
            value = " ".join(elem.text.strip().split())
            if value:
                vals.append(value)
    return sorted(set(vals))


def is_normal(labels: list[str]) -> bool:
    normalized = {x.strip().lower().rstrip(".") for x in labels}
    return "normal" in normalized or "normal chest" in normalized


def stable_key(report_id: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}|{salt}|{report_id}".encode("utf-8")).hexdigest()


def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    pos = p * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def quartile(word_count: int, q1: float, q2: float, q3: float) -> int:
    if word_count <= q1:
        return 1
    if word_count <= q2:
        return 2
    if word_count <= q3:
        return 3
    return 4


def collect_single_image_candidates() -> list[dict[str, object]]:
    if not REPORTS_TGZ.is_file() or not IMAGES_TGZ.is_file():
        raise RuntimeError("Open-I report/image archives are missing from the project-local WP3 cache")
    candidates: list[dict[str, object]] = []
    with tarfile.open(REPORTS_TGZ, "r:gz") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".xml")),
            key=lambda m: m.name,
        )
        for member in members:
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                root = ET.fromstring(f.read())
            except ET.ParseError:
                continue
            ids = image_ids(root)
            findings = abstract_text(root, "FINDINGS")
            impression = abstract_text(root, "IMPRESSION")
            if len(ids) != 1 or not (findings or impression):
                continue
            labels = mesh_labels(root)
            reference = (findings + " " + impression).strip()
            candidates.append(
                {
                    "report_id": pathlib.PurePosixPath(member.name).stem,
                    "image_id": ids[0],
                    "findings": findings,
                    "impression": impression,
                    "labels": labels,
                    "normal": is_normal(labels),
                    "reference_word_count": len(reference.split()),
                }
            )
    if len(candidates) < N_CASES:
        raise RuntimeError(f"Only {len(candidates)} eligible single-image reports; need {N_CASES}")
    return candidates


def proportional_targets(candidates: list[dict[str, object]], target_total: int) -> dict[tuple[bool, int], int]:
    counts = Counter((bool(c["normal"]), int(c["quartile"])) for c in candidates)
    raw = {key: target_total * count / len(candidates) for key, count in counts.items()}
    floors = {key: int(value) for key, value in raw.items()}
    remaining = target_total - sum(floors.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - floors[key]), str(key)))
    for key in order[:remaining]:
        floors[key] += 1
    return floors


def select_cases(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    lengths = sorted(int(c["reference_word_count"]) for c in candidates)
    q1 = percentile(lengths, 0.25)
    q2 = percentile(lengths, 0.50)
    q3 = percentile(lengths, 0.75)
    for c in candidates:
        c["quartile"] = quartile(int(c["reference_word_count"]), q1, q2, q3)

    targets = proportional_targets(candidates, N_CASES)
    nonnormal_labels = Counter()
    for c in candidates:
        if not bool(c["normal"]):
            for label in c["labels"]:
                normalized = label.strip().lower().rstrip(".")
                if normalized not in {"normal", "normal chest", "no indexing"}:
                    nonnormal_labels[label] += 1
    top_labels = [label for label, _ in nonnormal_labels.most_common(TOP_NONNORMAL_LABELS)]

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_counts: Counter[tuple[bool, int]] = Counter()

    # Guarantee deterministic representation of common non-normal metadata labels when possible.
    for label in top_labels:
        pool = [
            c for c in candidates
            if not bool(c["normal"]) and label in c["labels"] and str(c["report_id"]) not in selected_ids
        ]
        pool.sort(key=lambda c: stable_key(str(c["report_id"]), f"label:{label}"))
        chosen = None
        for c in pool:
            key = (bool(c["normal"]), int(c["quartile"]))
            if selected_counts[key] < targets.get(key, 0):
                chosen = c
                break
        if chosen is not None:
            selected.append(chosen)
            selected_ids.add(str(chosen["report_id"]))
            selected_counts[(bool(chosen["normal"]), int(chosen["quartile"]))] += 1

    # Fill each normal-status x reference-length-quartile stratum to its proportional target.
    for key in sorted(targets, key=str):
        need = targets[key] - selected_counts[key]
        if need <= 0:
            continue
        pool = [
            c for c in candidates
            if (bool(c["normal"]), int(c["quartile"])) == key and str(c["report_id"]) not in selected_ids
        ]
        pool.sort(key=lambda c: stable_key(str(c["report_id"]), f"stratum:{key}"))
        if len(pool) < need:
            raise RuntimeError(f"Insufficient candidates for stratum {key}: need {need}, found {len(pool)}")
        for c in pool[:need]:
            selected.append(c)
            selected_ids.add(str(c["report_id"]))
            selected_counts[key] += 1

    if len(selected) != N_CASES:
        raise RuntimeError(f"Expected {N_CASES} selected cases, found {len(selected)}")

    # Stable pseudo-random order mixes strata across 10-case energy blocks.
    selected.sort(key=lambda c: stable_key(str(c["report_id"]), "final-order"))
    label_coverage = {
        label: sum(1 for c in selected if label in c["labels"])
        for label in top_labels
    }
    report = {
        "status": "WP3_OPENI_100CASE_SINGLE_IMAGE_SELECTION_OK",
        "seed": SEED,
        "eligible_single_image_pool": len(candidates),
        "pool_normal_count": sum(bool(c["normal"]) for c in candidates),
        "pool_normal_fraction": sum(bool(c["normal"]) for c in candidates) / len(candidates),
        "selected_cases": len(selected),
        "selected_normal_count": sum(bool(c["normal"]) for c in selected),
        "selected_normal_fraction": sum(bool(c["normal"]) for c in selected) / len(selected),
        "reference_word_count_quartile_thresholds": {"q1": q1, "q2": q2, "q3": q3},
        "stratum_targets": {f"normal={k[0]},quartile={k[1]}": v for k, v in sorted(targets.items(), key=lambda x: str(x[0]))},
        "stratum_selected": {f"normal={k[0]},quartile={k[1]}": selected_counts[k] for k in sorted(targets, key=str)},
        "top_nonnormal_mesh_label_coverage": label_coverage,
        "design": "Single-image reports only to preserve image-reference correspondence for the single_2d benchmark arm. Deterministic proportional sampling by MeSH normal/non-normal metadata status and reference-text length quartile, with common non-normal MeSH-label coverage and stable hash ordering across 10-case energy blocks.",
        "normal_label_guardrail": "MeSH normal/non-normal is a metadata sampling stratum, not an independent clinical adjudication.",
    }
    return selected, report


def extract_images(selected: list[dict[str, object]]) -> None:
    wanted = {str(c["image_id"]) + ".png" for c in selected}
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(IMAGES_TGZ, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            basename = pathlib.PurePosixPath(member.name).name
            if basename not in wanted:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target = IMAGE_DIR / basename
            with target.open("wb") as out:
                out.write(src.read())
            found.add(basename)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - found)
    if missing:
        raise RuntimeError("Missing selected Open-I images: " + ", ".join(missing))


def write_manifest(selected: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_index", "dataset", "source_report_id", "source_image_id", "image_count_in_report",
        "local_image_path", "image_sha256", "findings_sha256", "impression_sha256",
        "normal_metadata_stratum", "reference_word_count", "reference_length_quartile", "mesh_labels_sha256",
    ]
    rows: list[dict[str, object]] = []
    for idx, c in enumerate(selected, start=1):
        image_path = IMAGE_DIR / f"{c['image_id']}.png"
        labels_serialized = "\n".join(str(x) for x in c["labels"])
        rows.append(
            {
                "case_index": idx,
                "dataset": "Open-I Indiana University Chest X-ray Collection",
                "source_report_id": c["report_id"],
                "source_image_id": c["image_id"],
                "image_count_in_report": 1,
                "local_image_path": image_path.relative_to(ROOT).as_posix(),
                "image_sha256": sha256_file(image_path),
                "findings_sha256": sha256_text(str(c["findings"])),
                "impression_sha256": sha256_text(str(c["impression"])),
                "normal_metadata_stratum": "normal" if bool(c["normal"]) else "non_normal",
                "reference_word_count": c["reference_word_count"],
                "reference_length_quartile": c["quartile"],
                "mesh_labels_sha256": sha256_text(labels_serialized),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_blocks() -> list[pathlib.Path]:
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(rows) != N_CASES:
        raise RuntimeError(f"Manifest has {len(rows)} rows, expected {N_CASES}")
    block_dir = OUT_DIR / "block_manifests"
    block_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    for start in range(0, N_CASES, BLOCK_SIZE):
        block = rows[start:start + BLOCK_SIZE]
        path = block_dir / f"block_{start // BLOCK_SIZE + 1}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(block[0].keys()))
            writer.writeheader()
            writer.writerows(block)
        paths.append(path)
    return paths


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    return statistics.stdev(values) / mean if mean else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_stages = 25

    candidates = collect_single_image_candidates()
    selected, sampling_report = select_cases(candidates)
    extract_images(selected)
    write_manifest(selected)
    (OUT_DIR / "sampling_report.json").write_text(json.dumps(sampling_report, indent=2) + "\n", encoding="utf-8")
    blocks = split_blocks()
    progress(1, total_stages, "Frozen stratified 100-case single-image manifest")

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

    qwen = load_module("wp3_qwen_energy_100single", ROOT / "scripts" / "run_wp3_openi_qwen_pilot.py")
    qwen.MODEL_ID = str(qwen_snapshot)
    qwen.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"
    qval = load_module("wp3_qwen_validation_100single", ROOT / "scripts" / "validate_wp3_openi_qwen_outputs.py")
    qval.MODEL_ID = str(qwen_snapshot)
    qval.MODEL_DIR = ROOT / ".wp3-models" / "Qwen2.5-VL-7B-Instruct-pinned"
    internvl = load_module("wp3_internvl_energy_100single", ROOT / "scripts" / "run_wp3_internvl3_openi_pilot.py")
    internvl.MODEL_ID = str(internvl_snapshot)
    internvl.MODEL_CACHE = ROOT / ".wp3-models" / "InternVL3-8B-pinned"

    block_rows: list[dict[str, object]] = []

    for i, block in enumerate(blocks, start=1):
        out = OUT_DIR / f"qwen_block_{i}"
        if out.exists():
            shutil.rmtree(out)
        qwen.MANIFEST = block
        qwen.OUT_DIR = out
        run_without_subprogress(qwen.main)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block_rows.append({
            "model": "Qwen2.5-VL-7B-Instruct", "block": i,
            "gross_wh_per_case": report["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": report["net_gpu_energy_wh_per_case"],
            "median_case_seconds": report["median_case_elapsed_seconds"],
            "idle_mean_power_w": report["idle_mean_power_w"],
            "peak_vram_mib": report["peak_vram_mib_torch"],
        })
        progress(2 + i, total_stages, f"Qwen energy block {i}/10")

    qval.OUT_DIR = OUT_DIR / "qwen_validation_100"
    qval.MANIFEST = MANIFEST
    run_without_subprogress(qval.main)
    qwen_validation = json.loads((qval.OUT_DIR / "summary.json").read_text(encoding="utf-8"))
    shutil.copy2(qval.OUT_DIR / "case_review.csv", OUT_DIR / "qwen_case_review_100.csv")
    progress(13, total_stages, "Qwen 100-case output screening complete")

    internvl_case_rows: list[dict[str, str]] = []
    internvl_f1: list[float] = []
    internvl_rouge: list[float] = []
    for i, block in enumerate(blocks, start=1):
        out = OUT_DIR / f"internvl_block_{i}"
        if out.exists():
            shutil.rmtree(out)
        internvl.MANIFEST = block
        internvl.OUT_DIR = out
        run_without_subprogress(internvl.main)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        block_rows.append({
            "model": "InternVL3-8B", "block": i,
            "gross_wh_per_case": report["gross_gpu_energy_wh_per_case"],
            "net_wh_per_case": report["net_gpu_energy_wh_per_case"],
            "median_case_seconds": report["median_case_elapsed_seconds"],
            "idle_mean_power_w": report["idle_mean_power_w"],
            "peak_vram_mib": report["peak_vram_mib_torch"],
        })
        with (out / "case_results.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        internvl_case_rows.extend(rows)
        internvl_f1.extend(float(r["unigram_f1"]) for r in rows)
        internvl_rouge.extend(float(r["rouge_l_f1"]) for r in rows)
        progress(13 + i, total_stages, f"InternVL energy/output block {i}/10")

    with (OUT_DIR / "internvl_case_results_100.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(internvl_case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(internvl_case_rows)

    with (OUT_DIR / "block_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()))
        writer.writeheader()
        writer.writerows(block_rows)

    def metric(model: str, key: str) -> list[float]:
        return [float(r[key]) for r in block_rows if r["model"] == model]

    q_gross = metric("Qwen2.5-VL-7B-Instruct", "gross_wh_per_case")
    q_net = metric("Qwen2.5-VL-7B-Instruct", "net_wh_per_case")
    q_time = metric("Qwen2.5-VL-7B-Instruct", "median_case_seconds")
    i_gross = metric("InternVL3-8B", "gross_wh_per_case")
    i_net = metric("InternVL3-8B", "net_wh_per_case")
    i_time = metric("InternVL3-8B", "median_case_seconds")

    summary = {
        "status": "WP3_OPENI_100CASE_SINGLE_IMAGE_TWO_MODEL_OK",
        "dataset": "Open-I stratified 100-case single-image benchmark",
        "cases": N_CASES,
        "energy_blocks_per_model": len(blocks),
        "cases_per_energy_block": BLOCK_SIZE,
        "precision": "BF16",
        "sampling_report": sampling_report,
        "qwen": {
            "model_id": QWEN_ID, "revision": QWEN_REVISION,
            "median_gross_wh_per_case_across_blocks": statistics.median(q_gross),
            "gross_block_cv": cv(q_gross),
            "median_net_wh_per_case_across_blocks": statistics.median(q_net),
            "net_block_cv": cv(q_net),
            "median_case_seconds_across_blocks": statistics.median(q_time),
            "mean_unigram_f1_100_cases": qwen_validation["mean_unigram_f1"],
            "mean_rouge_l_f1_100_cases": qwen_validation["mean_rouge_l_f1"],
        },
        "internvl3": {
            "model_id": INTERNVL_ID, "revision": INTERNVL_REVISION,
            "median_gross_wh_per_case_across_blocks": statistics.median(i_gross),
            "gross_block_cv": cv(i_gross),
            "median_net_wh_per_case_across_blocks": statistics.median(i_net),
            "net_block_cv": cv(i_net),
            "median_case_seconds_across_blocks": statistics.median(i_time),
            "mean_unigram_f1_100_cases": statistics.mean(internvl_f1),
            "mean_rouge_l_f1_100_cases": statistics.mean(internvl_rouge),
        },
        "internvl_to_qwen": {
            "gross_energy_ratio": statistics.median(i_gross) / statistics.median(q_gross),
            "net_energy_ratio": statistics.median(i_net) / statistics.median(q_net),
            "runtime_ratio": statistics.median(i_time) / statistics.median(q_time),
        },
        "measurement_scope": "Direct NVIDIA GPU board operational energy. Model loading and warmup excluded. Gross board energy is primary; idle-adjusted energy is secondary.",
        "benchmark_scope": "single_2d arm only; only reports with exactly one associated Open-I image were included to preserve image-reference correspondence.",
        "utility_note": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(25, total_stages, "100-case single-image two-model benchmark complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
