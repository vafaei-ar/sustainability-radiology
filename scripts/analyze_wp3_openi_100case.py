from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
OUT = ROOT / "results" / "wp3" / "openi_100case_analysis"
BOOT = 20000
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
        "unit": "bootstrap batches",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required input: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("empty values")
    pos = p * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def exact_sign_p(diffs: list[float]) -> float:
    nz = [x for x in diffs if x != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = min(sum(x > 0 for x in nz), sum(x < 0 for x in nz))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap_difference(a: list[float], b: list[float], rng: random.Random, n_boot: int) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired vectors must be nonempty and equal length")
    observed = statistics.mean(x - y for x, y in zip(a, b))
    n = len(a)
    reps = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(statistics.mean(a[i] - b[i] for i in idx))
    return observed, quantile(reps, 0.025), quantile(reps, 0.975)


def paired_bootstrap_ratio(num: list[float], den: list[float], rng: random.Random, n_boot: int) -> tuple[float, float, float]:
    if len(num) != len(den) or not num:
        raise ValueError("paired vectors must be nonempty and equal length")
    observed = statistics.mean(num) / statistics.mean(den)
    n = len(num)
    reps = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        nmean = statistics.mean(num[i] for i in idx)
        dmean = statistics.mean(den[i] for i in idx)
        reps.append(nmean / dmean)
    return observed, quantile(reps, 0.025), quantile(reps, 0.975)


def paired_case_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest = read_csv(SRC / "case_manifest_100.csv")
    qwen = read_csv(SRC / "qwen_case_review_100.csv")
    intern = read_csv(SRC / "internvl_case_results_100.csv")
    if not (len(manifest) == len(qwen) == len(intern) == 100):
        raise RuntimeError(f"Expected 100 rows in all case-level inputs, got manifest={len(manifest)}, qwen={len(qwen)}, internvl={len(intern)}")
    def key(rows):
        return {int(r["case_index"]): r for r in rows}
    mk, qk, ik = key(manifest), key(qwen), key(intern)
    if set(mk) != set(qk) or set(mk) != set(ik):
        raise RuntimeError("Case-index sets do not match across manifest and model outputs")
    return manifest, qwen, intern


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    block = read_csv(SRC / "block_summary.csv")
    if len(block) != 20:
        raise RuntimeError(f"Expected 20 block rows, found {len(block)}")
    qblocks = {int(r["block"]): r for r in block if r["model"] == "Qwen2.5-VL-7B-Instruct"}
    iblocks = {int(r["block"]): r for r in block if r["model"] == "InternVL3-8B"}
    if set(qblocks) != set(range(1, 11)) or set(iblocks) != set(range(1, 11)):
        raise RuntimeError("Expected matched blocks 1..10 for both models")

    block_metrics = [
        ("gross_wh_per_case", "ratio", "InternVL/Qwen"),
        ("net_wh_per_case", "ratio", "InternVL/Qwen"),
        ("median_case_seconds", "ratio", "InternVL/Qwen"),
    ]
    energy_results = {}
    for metric, _, label in block_metrics:
        q = [float(qblocks[i][metric]) for i in range(1, 11)]
        iv = [float(iblocks[i][metric]) for i in range(1, 11)]
        est, lo, hi = paired_bootstrap_ratio(iv, q, rng, BOOT)
        diffs = [x - y for x, y in zip(iv, q)]
        energy_results[metric] = {
            "comparison": label,
            "ratio_of_paired_block_means": est,
            "bootstrap_95_ci": [lo, hi],
            "exact_sign_test_p_two_sided": exact_sign_p(diffs),
            "internvl_higher_blocks": sum(x > 0 for x in diffs),
            "paired_blocks": 10,
        }
    progress(1, 4, "Paired block bootstrap complete")

    manifest, qwen, intern = paired_case_rows()
    qk = {int(r["case_index"]): r for r in qwen}
    ik = {int(r["case_index"]): r for r in intern}
    mk = {int(r["case_index"]): r for r in manifest}

    utility_results = {}
    for metric in ["unigram_f1", "rouge_l_f1"]:
        q = [float(qk[i][metric]) for i in range(1, 101)]
        iv = [float(ik[i][metric]) for i in range(1, 101)]
        est, lo, hi = paired_bootstrap_difference(q, iv, rng, BOOT)
        diffs = [x - y for x, y in zip(q, iv)]
        utility_results[metric] = {
            "comparison": "Qwen-InternVL",
            "mean_paired_difference": est,
            "bootstrap_95_ci": [lo, hi],
            "exact_sign_test_p_two_sided": exact_sign_p(diffs),
            "qwen_higher_cases": sum(x > 0 for x in diffs),
            "internvl_higher_cases": sum(x < 0 for x in diffs),
            "ties": sum(x == 0 for x in diffs),
            "paired_cases": 100,
        }
    progress(2, 4, "Paired case bootstrap complete")

    subgroup_rows = []
    subgroup_specs = [("normal_metadata_stratum", ["normal", "non_normal"]), ("reference_length_quartile", ["1", "2", "3", "4"])]
    for field, levels in subgroup_specs:
        for level in levels:
            ids = [i for i in range(1, 101) if mk[i][field] == level]
            if not ids:
                continue
            for metric in ["unigram_f1", "rouge_l_f1"]:
                q = [float(qk[i][metric]) for i in ids]
                iv = [float(ik[i][metric]) for i in ids]
                est, lo, hi = paired_bootstrap_difference(q, iv, rng, BOOT)
                subgroup_rows.append({
                    "stratifier": field,
                    "level": level,
                    "metric": metric,
                    "n_cases": len(ids),
                    "qwen_mean": statistics.mean(q),
                    "internvl_mean": statistics.mean(iv),
                    "qwen_minus_internvl": est,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                })
    progress(3, 4, "Subgroup bootstrap complete")

    with (OUT / "subgroup_utility.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(subgroup_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subgroup_rows)

    summary = {
        "status": "WP3_OPENI_100CASE_ANALYSIS_OK",
        "seed": SEED,
        "bootstrap_iterations": BOOT,
        "energy_and_runtime": energy_results,
        "utility_screening": utility_results,
        "interpretation": {
            "energy_unit": "Direct NVIDIA GPU board operational energy; gross primary and idle-adjusted net secondary.",
            "energy_inference_unit": "Ten matched 10-case blocks. Block bootstrap preserves paired case-mix at the block level.",
            "utility_inference_unit": "One hundred paired cases.",
            "utility_limit": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.",
            "subgroups": "Normal/non-normal is a MeSH metadata sampling stratum, not independent clinical adjudication. Subgroup intervals are exploratory and unadjusted for multiplicity.",
        },
    }
    (OUT / "paired_analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WP3 Open-I 100-case paired analysis",
        "",
        "This analysis compares pinned BF16 Qwen2.5-VL-7B-Instruct and InternVL3-8B on the frozen stratified 100-case single-image Open-I benchmark.",
        "",
        "Energy and runtime inference uses ten matched 10-case blocks. Lexical utility screening uses 100 paired cases. Gross GPU board energy is primary; idle-adjusted net energy is secondary.",
        "",
    ]
    for metric, res in energy_results.items():
        lines.append(f"- {metric}: InternVL/Qwen ratio {res['ratio_of_paired_block_means']:.3f}, 95% bootstrap CI {res['bootstrap_95_ci'][0]:.3f}-{res['bootstrap_95_ci'][1]:.3f}; InternVL higher in {res['internvl_higher_blocks']}/10 matched blocks; exact sign-test p={res['exact_sign_test_p_two_sided']:.4f}.")
    for metric, res in utility_results.items():
        lines.append(f"- {metric}: Qwen-InternVL mean paired difference {res['mean_paired_difference']:.4f}, 95% bootstrap CI {res['bootstrap_95_ci'][0]:.4f}-{res['bootstrap_95_ci'][1]:.4f}; exact sign-test p={res['exact_sign_test_p_two_sided']:.4g}.")
    lines += ["", "Lexical agreement remains a screening endpoint and must not be interpreted as clinical adequacy."]
    (OUT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(4, 4, "100-case paired analysis complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
