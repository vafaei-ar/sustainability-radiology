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
TWO = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
MED = ROOT / "results" / "wp3" / "medgemma_100case"
OUT = ROOT / "results" / "wp3" / "three_model_100case_analysis"
BOOT = 20000
SEED = 20260831
REVIEW_N = 50


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
        "unit": "analysis stages",
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


def paired_bootstrap_ratio(num: list[float], den: list[float], rng: random.Random) -> tuple[float, float, float]:
    if len(num) != len(den) or not num:
        raise ValueError("paired vectors must be nonempty and equal length")
    observed = statistics.mean(num) / statistics.mean(den)
    n = len(num)
    reps = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(statistics.mean(num[i] for i in idx) / statistics.mean(den[i] for i in idx))
    return observed, quantile(reps, 0.025), quantile(reps, 0.975)


def paired_bootstrap_difference(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired vectors must be nonempty and equal length")
    observed = statistics.mean(x - y for x, y in zip(a, b))
    n = len(a)
    reps = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        reps.append(statistics.mean(a[i] - b[i] for i in idx))
    return observed, quantile(reps, 0.025), quantile(reps, 0.975)


def holm_adjust(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda x: x[1])
    m = len(ordered)
    out: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = min(1.0, running)
    return out


def index_rows(rows: list[dict[str, str]], expected: int, label: str) -> dict[int, dict[str, str]]:
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows for {label}, found {len(rows)}")
    out = {int(r["case_index"]): r for r in rows}
    if set(out) != set(range(1, expected + 1)):
        raise RuntimeError(f"Non-contiguous case indices for {label}")
    return out


def load_blocks() -> dict[str, dict[int, dict[str, str]]]:
    two = read_csv(TWO / "block_summary.csv")
    med = read_csv(MED / "block_summary.csv")
    q = {int(r["block"]): r for r in two if r["model"] == "Qwen2.5-VL-7B-Instruct"}
    i = {int(r["block"]): r for r in two if r["model"] == "InternVL3-8B"}
    m = {int(r["block"]): r for r in med}
    expected = set(range(1, 11))
    if set(q) != expected or set(i) != expected or set(m) != expected:
        raise RuntimeError("Expected matched blocks 1..10 for Qwen, InternVL3, and MedGemma")
    return {"Qwen": q, "InternVL3": i, "MedGemma": m}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    blocks = load_blocks()

    pair_specs = [("MedGemma", "Qwen"), ("MedGemma", "InternVL3"), ("Qwen", "InternVL3")]
    metrics = ["gross_wh_per_case", "net_wh_per_case", "median_case_seconds"]
    operational_rows = []
    raw_sign_ps: list[tuple[str, float]] = []
    for metric in metrics:
        for a, b in pair_specs:
            av = [float(blocks[a][j][metric]) for j in range(1, 11)]
            bv = [float(blocks[b][j][metric]) for j in range(1, 11)]
            est, lo, hi = paired_bootstrap_ratio(av, bv, rng)
            diffs = [x - y for x, y in zip(av, bv)]
            p = exact_sign_p(diffs)
            key = f"{metric}:{a}/{b}"
            raw_sign_ps.append((key, p))
            operational_rows.append({
                "metric": metric,
                "numerator_model": a,
                "denominator_model": b,
                "ratio_of_paired_block_means": est,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "exact_sign_p_two_sided": p,
                "numerator_higher_blocks": sum(x > 0 for x in diffs),
                "numerator_lower_blocks": sum(x < 0 for x in diffs),
                "ties": sum(x == 0 for x in diffs),
                "paired_blocks": 10,
                "holm_p_across_all_operational_pairwise_tests": None,
            })
    adj = holm_adjust(raw_sign_ps)
    for row in operational_rows:
        key = f"{row['metric']}:{row['numerator_model']}/{row['denominator_model']}"
        row["holm_p_across_all_operational_pairwise_tests"] = adj[key]
    progress(1, 4, "Operational pairwise bootstrap complete")

    manifest = index_rows(read_csv(TWO / "case_manifest_100.csv"), 100, "manifest")
    q = index_rows(read_csv(TWO / "qwen_case_review_100.csv"), 100, "Qwen")
    i = index_rows(read_csv(TWO / "internvl_case_results_100.csv"), 100, "InternVL3")
    m = index_rows(read_csv(MED / "case_results_100.csv"), 100, "MedGemma")
    cases = {"Qwen": q, "InternVL3": i, "MedGemma": m}

    utility_rows = []
    utility_raw_ps: list[tuple[str, float]] = []
    for metric in ["unigram_f1", "rouge_l_f1"]:
        for a, b in pair_specs:
            av = [float(cases[a][j][metric]) for j in range(1, 101)]
            bv = [float(cases[b][j][metric]) for j in range(1, 101)]
            est, lo, hi = paired_bootstrap_difference(av, bv, rng)
            diffs = [x - y for x, y in zip(av, bv)]
            p = exact_sign_p(diffs)
            key = f"{metric}:{a}-{b}"
            utility_raw_ps.append((key, p))
            utility_rows.append({
                "metric": metric,
                "model_a": a,
                "model_b": b,
                "mean_a": statistics.mean(av),
                "mean_b": statistics.mean(bv),
                "mean_paired_difference_a_minus_b": est,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "exact_sign_p_two_sided": p,
                "a_higher_cases": sum(x > 0 for x in diffs),
                "b_higher_cases": sum(x < 0 for x in diffs),
                "ties": sum(x == 0 for x in diffs),
                "paired_cases": 100,
                "holm_p_across_all_utility_pairwise_tests": None,
            })
    uadj = holm_adjust(utility_raw_ps)
    for row in utility_rows:
        key = f"{row['metric']}:{row['model_a']}-{row['model_b']}"
        row["holm_p_across_all_utility_pairwise_tests"] = uadj[key]
    progress(2, 4, "Utility pairwise bootstrap complete")

    # Deterministic balanced clinical-review packet. This prepares review but does not perform adjudication.
    strata: dict[tuple[str, str], list[int]] = {}
    for idx, r in manifest.items():
        key = (r["normal_metadata_stratum"], r["reference_length_quartile"])
        strata.setdefault(key, []).append(idx)
    review_ids: list[int] = []
    target_per_stratum = REVIEW_N // len(strata)
    remainder = REVIEW_N - target_per_stratum * len(strata)
    for sidx, key in enumerate(sorted(strata)):
        ids = list(strata[key])
        local_rng = random.Random(f"{SEED}:{key}")
        local_rng.shuffle(ids)
        take = target_per_stratum + (1 if sidx < remainder else 0)
        review_ids.extend(ids[:take])
    review_ids = sorted(review_ids)
    if len(review_ids) != REVIEW_N:
        raise RuntimeError(f"Expected {REVIEW_N} clinical-review cases, found {len(review_ids)}")

    blind_rng = random.Random(SEED + 17)
    review_rows = []
    for idx in review_ids:
        mapping = ["Qwen", "InternVL3", "MedGemma"]
        blind_rng.shuffle(mapping)
        reference_findings = q[idx].get("reference_findings", "") or i[idx].get("reference_findings", "") or m[idx].get("reference_findings", "")
        reference_impression = q[idx].get("reference_impression", "") or i[idx].get("reference_impression", "") or m[idx].get("reference_impression", "")
        row = {
            "review_case_id": f"R{idx:03d}",
            "case_index": idx,
            "normal_metadata_stratum": manifest[idx]["normal_metadata_stratum"],
            "reference_length_quartile": manifest[idx]["reference_length_quartile"],
            "reference_findings": reference_findings,
            "reference_impression": reference_impression,
            "output_A": cases[mapping[0]][idx]["model_output"],
            "output_B": cases[mapping[1]][idx]["model_output"],
            "output_C": cases[mapping[2]][idx]["model_output"],
            "major_abnormality_correct_A": "",
            "major_abnormality_correct_B": "",
            "major_abnormality_correct_C": "",
            "clinically_important_omission_A": "",
            "clinically_important_omission_B": "",
            "clinically_important_omission_C": "",
            "clinically_important_hallucination_A": "",
            "clinically_important_hallucination_B": "",
            "clinically_important_hallucination_C": "",
            "laterality_or_location_error_A": "",
            "laterality_or_location_error_B": "",
            "laterality_or_location_error_C": "",
            "critical_safety_error_A": "",
            "critical_safety_error_B": "",
            "critical_safety_error_C": "",
            "overall_acceptable_A": "",
            "overall_acceptable_B": "",
            "overall_acceptable_C": "",
            "reviewer_notes": "",
        }
        review_rows.append(row)

    with (OUT / "clinical_review_blinded_50.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        w.writeheader(); w.writerows(review_rows)
    # Keep the unblinding key separate from the review sheet.
    blind_rng = random.Random(SEED + 17)
    key_rows = []
    for idx in review_ids:
        mapping = ["Qwen", "InternVL3", "MedGemma"]
        blind_rng.shuffle(mapping)
        key_rows.append({"review_case_id": f"R{idx:03d}", "A": mapping[0], "B": mapping[1], "C": mapping[2]})
    with (OUT / "clinical_review_unblinding_key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["review_case_id", "A", "B", "C"])
        w.writeheader(); w.writerows(key_rows)
    progress(3, 4, "Blinded clinical-review packet prepared")

    with (OUT / "operational_pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(operational_rows[0].keys()))
        w.writeheader(); w.writerows(operational_rows)
    with (OUT / "utility_pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(utility_rows[0].keys()))
        w.writeheader(); w.writerows(utility_rows)

    summary = {
        "status": "WP3_THREE_MODEL_100CASE_ANALYSIS_OK",
        "seed": SEED,
        "bootstrap_iterations": BOOT,
        "models": ["MedGemma-4B", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"],
        "operational_pairwise": operational_rows,
        "utility_pairwise": utility_rows,
        "clinical_review_packet": {
            "n_cases": REVIEW_N,
            "selection": "Deterministic balanced sample across MeSH normal/non-normal metadata status and reference-length quartile.",
            "blinding": "Model identities randomized independently by case across output columns A/B/C. Unblinding key stored separately.",
            "adjudication_status": "Not performed. Blank fields are provided for human clinical review.",
        },
        "interpretation": {
            "energy_scope": "Direct NVIDIA GPU board operational energy; gross primary and idle-adjusted net secondary.",
            "operational_inference_unit": "Ten matched 10-case blocks.",
            "utility_inference_unit": "One hundred paired cases.",
            "utility_limit": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.",
            "multiplicity": "Holm adjustment is reported separately within the nine operational pairwise sign tests and six utility pairwise sign tests.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WP3 three-model 100-case paired analysis",
        "",
        "This analysis compares MedGemma-4B, Qwen2.5-VL-7B-Instruct, and InternVL3-8B on the same frozen stratified 100-case single-image Open-I cohort.",
        "",
        "Operational comparisons use ten matched 10-case blocks. Utility comparisons use 100 paired cases. Lexical metrics are screening endpoints only.",
        "",
        "A deterministic 50-case blinded clinical-review packet was also prepared. No clinical adjudication was performed by this script.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(4, 4, "Three-model analysis complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
