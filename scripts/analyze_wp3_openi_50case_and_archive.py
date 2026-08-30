from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import statistics
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "wp3" / "openi_50case_two_model"
REPORTS_TGZ = ROOT / ".wp3-data" / "openi_cxr_pilot" / "NLMCXR_reports.tgz"
OUT_DIR = ROOT / "results" / "wp3" / "openi_50case_analysis"
BOOTSTRAP_N = 20000
SEED = 20260830


def report_progress(current: int, total: int, phase: str, unit: str = "stages") -> None:
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


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = p * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


def paired_bootstrap(a: list[float], b: list[float], *, ratio: bool, rng: random.Random) -> dict[str, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired vectors must have equal nonzero length")
    point = statistics.mean(a) / statistics.mean(b) if ratio else statistics.mean(a[i] - b[i] for i in range(len(a)))
    vals: list[float] = []
    n = len(a)
    for _ in range(BOOTSTRAP_N):
        idx = [rng.randrange(n) for _ in range(n)]
        aa = [a[i] for i in idx]
        bb = [b[i] for i in idx]
        if ratio:
            den = statistics.mean(bb)
            vals.append(statistics.mean(aa) / den if den else math.nan)
        else:
            vals.append(statistics.mean(aa[i] - bb[i] for i in range(n)))
    vals = sorted(v for v in vals if math.isfinite(v))
    return {
        "point_estimate": point,
        "bootstrap_ci95_low": percentile(vals, 0.025),
        "bootstrap_ci95_high": percentile(vals, 0.975),
        "bootstrap_resamples": BOOTSTRAP_N,
    }


def exact_two_sided_sign_test(differences: list[float]) -> float | None:
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return None
    positives = sum(d > 0 for d in nonzero)
    k = min(positives, n - positives)
    tail = sum(math.comb(n, j) for j in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_block_pairs() -> dict[str, dict[str, list[float]]]:
    path = INPUT_DIR / "block_summary.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    by_model: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_model[row["model"]][int(row["batch"])] = row
    q_name = "Qwen2.5-VL-7B-Instruct"
    i_name = "InternVL3-8B"
    batches = sorted(set(by_model[q_name]) & set(by_model[i_name]))
    if batches != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected paired batches 1..5, found {batches}")
    metrics = {
        "gross_wh_per_case": {},
        "net_wh_per_case": {},
        "median_case_seconds": {},
    }
    for metric in metrics:
        metrics[metric] = {
            "qwen": [float(by_model[q_name][b][metric]) for b in batches],
            "internvl3": [float(by_model[i_name][b][metric]) for b in batches],
        }
    return metrics


def load_utility_pairs() -> dict[str, dict[str, list[float]]]:
    q_path = INPUT_DIR / "qwen_case_review_50.csv"
    i_path = INPUT_DIR / "internvl_case_results_50.csv"
    q_rows = {row["source_report_id"]: row for row in csv.DictReader(q_path.open("r", encoding="utf-8", newline=""))}
    i_rows = {row["source_report_id"]: row for row in csv.DictReader(i_path.open("r", encoding="utf-8", newline=""))}
    ids = sorted(set(q_rows) & set(i_rows))
    if len(ids) != 50:
        raise RuntimeError(f"Expected 50 paired utility cases, found {len(ids)}")
    out = {}
    for metric in ["unigram_f1", "rouge_l_f1"]:
        out[metric] = {
            "qwen": [float(q_rows[rid][metric]) for rid in ids],
            "internvl3": [float(i_rows[rid][metric]) for rid in ids],
        }
    return out


def words(text: str) -> int:
    return len(text.split())


def abstract_text(root: ET.Element, label: str) -> str:
    parts = []
    target = label.upper()
    for elem in root.iter():
        if elem.tag.endswith("AbstractText") and (elem.attrib.get("Label") or "").upper() == target and elem.text:
            parts.append(elem.text.strip())
    return " ".join(x for x in parts if x)


def image_ids(root: ET.Element) -> list[str]:
    vals = []
    for elem in root.iter():
        if elem.tag.endswith("parentImage"):
            value = (elem.attrib.get("id") or "").strip()
            if value:
                vals.append(value)
    return vals


def mesh_labels(root: ET.Element) -> list[str]:
    labels = []
    for elem in root.iter():
        local = elem.tag.rsplit("}", 1)[-1].lower()
        if local in {"major", "minor"} and elem.text:
            value = " ".join(elem.text.strip().split())
            if value:
                labels.append(value)
    return labels


def characterize_archive() -> tuple[dict[str, object], list[tuple[str, int]]]:
    if not REPORTS_TGZ.is_file():
        raise RuntimeError(f"Missing project-local Open-I reports archive: {REPORTS_TGZ}")
    total_xml = 0
    eligible = 0
    single = 0
    multi = 0
    findings_present = 0
    impression_present = 0
    both_present = 0
    mesh_present = 0
    mesh_normal = 0
    lengths: list[int] = []
    image_counts: Counter[int] = Counter()
    labels: Counter[str] = Counter()
    report_label_sets: list[set[str]] = []

    with tarfile.open(REPORTS_TGZ, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".xml")]
        total_xml = len(members)
        for idx, member in enumerate(members, start=1):
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
            if not ids or not (findings or impression):
                continue
            eligible += 1
            n_img = len(ids)
            image_counts[n_img] += 1
            if n_img == 1:
                single += 1
            else:
                multi += 1
            if findings:
                findings_present += 1
            if impression:
                impression_present += 1
            if findings and impression:
                both_present += 1
            lengths.append(words((findings + " " + impression).strip()))
            labs = {x for x in mesh_labels(root)}
            if labs:
                mesh_present += 1
                report_label_sets.append(labs)
                for label in labs:
                    labels[label] += 1
                normalized = {x.strip().lower().rstrip(".") for x in labs}
                if "normal" in normalized or "normal chest" in normalized:
                    mesh_normal += 1
            if idx % 500 == 0:
                report_progress(idx, max(total_xml, 1), "Characterizing Open-I XML reports", unit="reports")

    lengths_sorted = sorted(lengths)
    top_labels = labels.most_common(30)
    summary: dict[str, object] = {
        "total_xml_reports": total_xml,
        "eligible_reports_with_image_and_reference_text": eligible,
        "single_image_reports": single,
        "multi_image_reports": multi,
        "single_image_fraction": single / eligible if eligible else None,
        "findings_present": findings_present,
        "impression_present": impression_present,
        "both_findings_and_impression_present": both_present,
        "reports_with_mesh_labels": mesh_present,
        "mesh_normal_reports": mesh_normal,
        "mesh_normal_fraction_among_eligible": mesh_normal / eligible if eligible else None,
        "reference_word_count_median": statistics.median(lengths) if lengths else None,
        "reference_word_count_q1": percentile(lengths_sorted, 0.25) if lengths else None,
        "reference_word_count_q3": percentile(lengths_sorted, 0.75) if lengths else None,
        "max_images_per_report": max(image_counts) if image_counts else None,
        "image_count_distribution": {str(k): image_counts[k] for k in sorted(image_counts)},
        "mesh_normal_definition": "Report has a MeSH major/minor label exactly normalized to 'normal' or 'normal chest'. This is a text-metadata stratum, not an independent clinical adjudication.",
    }
    return summary, top_labels


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for required in [
        INPUT_DIR / "block_summary.csv",
        INPUT_DIR / "qwen_case_review_50.csv",
        INPUT_DIR / "internvl_case_results_50.csv",
    ]:
        if not required.is_file():
            raise RuntimeError(f"Missing prior 50-case artifact: {required}")

    rng = random.Random(SEED)
    report_progress(1, 4, "Loaded 50-case paired inputs")

    block_metrics = load_block_pairs()
    block_analysis: dict[str, object] = {}
    for metric, values in block_metrics.items():
        q = values["qwen"]
        i = values["internvl3"]
        ratios = [i[k] / q[k] for k in range(len(q))]
        diffs = [i[k] - q[k] for k in range(len(q))]
        block_analysis[metric] = {
            "qwen_values": q,
            "internvl3_values": i,
            "paired_internvl_to_qwen_ratios": ratios,
            "paired_internvl_minus_qwen_differences": diffs,
            "ratio_of_paired_means": paired_bootstrap(i, q, ratio=True, rng=rng),
            "mean_paired_difference": paired_bootstrap(i, q, ratio=False, rng=rng),
            "exact_two_sided_sign_test_p_for_internvl_minus_qwen": exact_two_sided_sign_test(diffs),
        }
    report_progress(2, 4, "Completed paired block energy/runtime bootstrap")

    utility_metrics = load_utility_pairs()
    utility_analysis: dict[str, object] = {}
    for metric, values in utility_metrics.items():
        q = values["qwen"]
        i = values["internvl3"]
        diffs = [q[k] - i[k] for k in range(len(q))]
        utility_analysis[metric] = {
            "qwen_mean": statistics.mean(q),
            "internvl3_mean": statistics.mean(i),
            "qwen_minus_internvl3": paired_bootstrap(q, i, ratio=False, rng=rng),
            "exact_two_sided_sign_test_p_for_qwen_minus_internvl3": exact_two_sided_sign_test(diffs),
            "paired_cases": len(q),
        }
    report_progress(3, 4, "Completed paired 50-case utility bootstrap")

    archive_summary, top_labels = characterize_archive()

    statistical = {
        "status": "WP3_OPENI_50CASE_PAIRED_ANALYSIS_OK",
        "bootstrap_seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_N,
        "energy_runtime_analysis_unit": "paired 10-case block; five matched blocks per model",
        "utility_analysis_unit": "paired Open-I report/image case; 50 matched cases",
        "block_analysis": block_analysis,
        "utility_analysis": utility_analysis,
        "interpretation_guardrails": [
            "Gross NVIDIA GPU board energy is the primary operational-energy endpoint; idle-adjusted energy is secondary.",
            "Five energy blocks provide limited inferential resolution. Bootstrap intervals are descriptive and exact sign tests have coarse p-value resolution.",
            "Unigram F1 and ROUGE-L are lexical screening metrics and do not establish clinical adequacy.",
            "The 50-case cohort is deterministic but lexicographically selected and is not representative of the full Open-I archive.",
        ],
    }
    (OUT_DIR / "paired_statistical_analysis.json").write_text(json.dumps(statistical, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "archive_characterization.json").write_text(json.dumps(archive_summary, indent=2) + "\n", encoding="utf-8")

    with (OUT_DIR / "mesh_label_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mesh_label", "eligible_report_count"])
        writer.writerows(top_labels)

    plan = f"""# Open-I 100-case stratification plan\n\n## Evidence from the full archive\n\n- XML reports inspected: {archive_summary['total_xml_reports']}\n- Eligible reports with at least one image and nonempty FINDINGS or IMPRESSION: {archive_summary['eligible_reports_with_image_and_reference_text']}\n- Single-image eligible reports: {archive_summary['single_image_reports']} ({archive_summary['single_image_fraction']:.1%})\n- Multi-image eligible reports: {archive_summary['multi_image_reports']}\n- Reports with MeSH labels: {archive_summary['reports_with_mesh_labels']}\n- Reports carrying a MeSH label normalized to normal/normal chest: {archive_summary['mesh_normal_reports']} ({archive_summary['mesh_normal_fraction_among_eligible']:.1%} of eligible reports)\n- Reference-text word-count median [Q1, Q3]: {archive_summary['reference_word_count_median']} [{archive_summary['reference_word_count_q1']:.1f}, {archive_summary['reference_word_count_q3']:.1f}]\n\n## Recommended sampling design\n\nDo not extend the lexicographic first-50 rule. Freeze the next cohort by deterministic stratified sampling from all eligible reports. Use MeSH normal versus non-normal status as the primary metadata stratum where labels are available, image-count class (single versus multi-image report) as a secondary stratum, and reference-text length quartile as a workload-diversity check. Within the non-normal stratum, require coverage across common MeSH labels rather than allowing one frequent label to dominate.\n\nThe final 100-case manifest should store only source report/image identifiers, image hashes, strata, and reference hashes. Raw radiographs should remain project-local. View position should not be called frontal/lateral unless it is verified from a reliable source field.\n\nThis characterization is based on Open-I report metadata and does not independently adjudicate clinical diagnoses. The normal stratum is therefore a sampling label, not a gold-standard clinical classification.\n"""
    (OUT_DIR / "stratification_plan.md").write_text(plan, encoding="utf-8")

    report_progress(4, 4, "Paired analysis and archive characterization complete")
    print(statistical["status"])
    print(json.dumps({"archive": archive_summary, "top_mesh_labels": top_labels[:10]}, sort_keys=True))


if __name__ == "__main__":
    main()
