from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import statistics
import time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
OUT_DIR = ROOT / "results" / "wp3" / "openi_100case_analysis"
BOOTSTRAP_N = 20000
SEED = 20260830


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
        "unit": "stages",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def percentile(values: list[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def paired_bootstrap(a: list[float], b: list[float], *, ratio: bool, rng: random.Random) -> dict[str, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired vectors must have equal nonzero length")
    point = statistics.mean(a) / statistics.mean(b) if ratio else statistics.mean(x - y for x, y in zip(a, b))
    vals = []
    n = len(a)
    for _ in range(BOOTSTRAP_N):
        idx = [rng.randrange(n) for _ in range(n)]
        aa = [a[i] for i in idx]
        bb = [b[i] for i in idx]
        if ratio:
            den = statistics.mean(bb)
            vals.append(statistics.mean(aa) / den if den else math.nan)
        else:
            vals.append(statistics.mean(x - y for x, y in zip(aa, bb)))
    vals = [v for v in vals if math.isfinite(v)]
    return {
        "point_estimate": point,
        "bootstrap_ci95_low": percentile(vals, 0.025),
        "bootstrap_ci95_high": percentile(vals, 0.975),
        "bootstrap_resamples": BOOTSTRAP_N,
    }


def exact_sign_test(diffs: list[float]) -> float | None:
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if not n:
        return None
    positives = sum(d > 0 for d in nonzero)
    k = min(positives, n - positives)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def analyze_blocks(rng: random.Random) -> dict[str, object]:
    rows = load_csv(INPUT_DIR / "block_summary.csv")
    by_model: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_model[row["model"]][int(row["block"])] = row
    qname = "Qwen2.5-VL-7B-Instruct"
    iname = "InternVL3-8B"
    blocks = sorted(set(by_model[qname]) & set(by_model[iname]))
    if blocks != list(range(1, 11)):
        raise RuntimeError(f"Expected paired blocks 1..10, found {blocks}")
    out: dict[str, object] = {}
    for metric in ["gross_wh_per_case", "net_wh_per_case", "median_case_seconds"]:
        q = [float(by_model[qname][b][metric]) for b in blocks]
        i = [float(by_model[iname][b][metric]) for b in blocks]
        diffs = [x - y for x, y in zip(i, q)]
        out[metric] = {
            "qwen_values": q,
            "internvl3_values": i,
            "internvl3_to_qwen_ratio": paired_bootstrap(i, q, ratio=True, rng=rng),
            "internvl3_minus_qwen": paired_bootstrap(i, q, ratio=False, rng=rng),
            "exact_two_sided_sign_test_p": exact_sign_test(diffs),
        }
    return out


def analyze_cases(rng: random.Random) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = {r["source_report_id"]: r for r in load_csv(INPUT_DIR / "case_manifest_100.csv")}
    q = {r["source_report_id"]: r for r in load_csv(INPUT_DIR / "qwen_case_review_100.csv")}
    i = {r["source_report_id"]: r for r in load_csv(INPUT_DIR / "internvl_case_results_100.csv")}
    ids = sorted(set(manifest) & set(q) & set(i))
    if len(ids) != 100:
        raise RuntimeError(f"Expected 100 paired cases, found {len(ids)}")

    overall: dict[str, object] = {}
    subgroup_rows: list[dict[str, object]] = []
    for metric in ["unigram_f1", "rouge_l_f1"]:
        qv = [float(q[r][metric]) for r in ids]
        iv = [float(i[r][metric]) for r in ids]
        diffs = [a - b for a, b in zip(qv, iv)]
        overall[metric] = {
            "paired_cases": len(ids),
            "qwen_mean": statistics.mean(qv),
            "internvl3_mean": statistics.mean(iv),
            "qwen_minus_internvl3": paired_bootstrap(qv, iv, ratio=False, rng=rng),
            "exact_two_sided_sign_test_p": exact_sign_test(diffs),
        }

        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rid in ids:
            m = manifest[rid]
            groups[(m["normal_metadata_stratum"], m["reference_length_quartile"])].append(rid)
        groups[("all", "all")] = ids
        groups[("normal", "all")] = [r for r in ids if manifest[r]["normal_metadata_stratum"] == "normal"]
        groups[("non_normal", "all")] = [r for r in ids if manifest[r]["normal_metadata_stratum"] == "non_normal"]

        for (normal_status, quartile), members in sorted(groups.items()):
            if not members:
                continue
            qa = [float(q[r][metric]) for r in members]
            ia = [float(i[r][metric]) for r in members]
            ci = paired_bootstrap(qa, ia, ratio=False, rng=rng)
            subgroup_rows.append({
                "metric": metric,
                "normal_metadata_stratum": normal_status,
                "reference_length_quartile": quartile,
                "n": len(members),
                "qwen_mean": statistics.mean(qa),
                "internvl3_mean": statistics.mean(ia),
                "qwen_minus_internvl3": ci["point_estimate"],
                "ci95_low": ci["bootstrap_ci95_low"],
                "ci95_high": ci["bootstrap_ci95_high"],
            })
    return overall, subgroup_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        INPUT_DIR / "block_summary.csv",
        INPUT_DIR / "case_manifest_100.csv",
        INPUT_DIR / "qwen_case_review_100.csv",
        INPUT_DIR / "internvl_case_results_100.csv",
        INPUT_DIR / "summary.json",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing required input: {path}")

    rng = random.Random(SEED)
    progress(1, 4, "Loaded 100-case benchmark artifacts")
    block_analysis = analyze_blocks(rng)
    progress(2, 4, "Completed paired 10-block energy/runtime analysis")
    case_analysis, subgroup_rows = analyze_cases(rng)
    progress(3, 4, "Completed paired 100-case utility and subgroup analysis")

    result = {
        "status": "WP3_OPENI_100CASE_PAIRED_ANALYSIS_OK",
        "bootstrap_seed": SEED,
        "bootstrap_resamples": BOOTSTRAP_N,
        "energy_runtime_analysis_unit": "paired 10-case block; 10 matched blocks per model",
        "utility_analysis_unit": "paired Open-I single-image report case; 100 matched cases",
        "block_analysis": block_analysis,
        "utility_analysis": case_analysis,
        "guardrails": [
            "Gross NVIDIA GPU board energy is the primary operational-energy endpoint; idle-adjusted energy is secondary.",
            "Energy inference is block-level because direct board energy was integrated over 10-case blocks, not individually metered cases.",
            "Unigram F1 and ROUGE-L are lexical screening metrics and do not establish clinical adequacy.",
            "MeSH normal/non-normal is a metadata sampling stratum, not independent clinical adjudication.",
        ],
    }
    (OUT_DIR / "paired_statistical_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (OUT_DIR / "utility_subgroup_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subgroup_rows)

    gross = block_analysis["gross_wh_per_case"]["internvl3_to_qwen_ratio"]
    net = block_analysis["net_wh_per_case"]["internvl3_to_qwen_ratio"]
    runtime = block_analysis["median_case_seconds"]["internvl3_to_qwen_ratio"]
    uni = case_analysis["unigram_f1"]["qwen_minus_internvl3"]
    rouge = case_analysis["rouge_l_f1"]["qwen_minus_internvl3"]
    report = f"""# WP3 Open-I 100-case paired analysis\n\nThe benchmark contains 100 deterministically stratified single-image Open-I reports, measured in 10 matched 10-case blocks per model.\n\n## Primary operational result\n\nInternVL3/Qwen gross GPU board-energy ratio: {gross['point_estimate']:.3f} (bootstrap 95% CI {gross['bootstrap_ci95_low']:.3f} to {gross['bootstrap_ci95_high']:.3f}).\nIdle-adjusted energy ratio: {net['point_estimate']:.3f} (95% CI {net['bootstrap_ci95_low']:.3f} to {net['bootstrap_ci95_high']:.3f}).\nRuntime ratio: {runtime['point_estimate']:.3f} (95% CI {runtime['bootstrap_ci95_low']:.3f} to {runtime['bootstrap_ci95_high']:.3f}).\n\n## Lexical screening\n\nQwen minus InternVL3 unigram-F1 difference: {uni['point_estimate']:.3f} (95% CI {uni['bootstrap_ci95_low']:.3f} to {uni['bootstrap_ci95_high']:.3f}).\nQwen minus InternVL3 ROUGE-L difference: {rouge['point_estimate']:.3f} (95% CI {rouge['bootstrap_ci95_low']:.3f} to {rouge['bootstrap_ci95_high']:.3f}).\n\n## Interpretation guardrails\n\nGross GPU board energy is the primary endpoint. Idle-adjusted energy is secondary. Energy inference is at the 10-case block level. Lexical metrics are screening measures only and cannot establish clinical adequacy. The MeSH normal/non-normal variable is a sampling metadata stratum rather than independent clinical adjudication.\n"""
    (OUT_DIR / "publication_summary.md").write_text(report, encoding="utf-8")
    progress(4, 4, "100-case paired analysis complete")
    print(result["status"])
    print(json.dumps({"gross_ratio": gross, "net_ratio": net, "runtime_ratio": runtime, "unigram_difference": uni, "rouge_difference": rouge}, sort_keys=True))


if __name__ == "__main__":
    main()
