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
PAIR_SPECS = [("MedGemma", "Qwen"), ("MedGemma", "InternVL3"), ("Qwen", "InternVL3")]


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    p = pathlib.Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"schema_version": 1, "current": current, "total": total, "fraction": current / total, "phase": phase, "unit": "analysis stages", "updated_at_epoch": time.time()}), encoding="utf-8")
    os.replace(tmp, p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required input: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_rows(rows: list[dict[str, str]], expected: int, label: str) -> dict[int, dict[str, str]]:
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows for {label}, found {len(rows)}")
    out = {int(r["case_index"]): r for r in rows}
    if set(out) != set(range(1, expected + 1)):
        raise RuntimeError(f"Non-contiguous case indices for {label}")
    return out


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = p * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def exact_sign_p(diffs: list[float]) -> float:
    nz = [x for x in diffs if x != 0]
    if not nz:
        return 1.0
    n = len(nz)
    k = min(sum(x > 0 for x in nz), sum(x < 0 for x in nz))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired_bootstrap_ratio(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float, float]:
    obs = statistics.mean(a) / statistics.mean(b)
    n = len(a)
    reps = []
    for _ in range(BOOT):
        ids = [rng.randrange(n) for _ in range(n)]
        reps.append(statistics.mean(a[i] for i in ids) / statistics.mean(b[i] for i in ids))
    return obs, quantile(reps, 0.025), quantile(reps, 0.975)


def paired_bootstrap_diff(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float, float]:
    obs = statistics.mean(x - y for x, y in zip(a, b))
    n = len(a)
    reps = []
    for _ in range(BOOT):
        ids = [rng.randrange(n) for _ in range(n)]
        reps.append(statistics.mean(a[i] - b[i] for i in ids))
    return obs, quantile(reps, 0.025), quantile(reps, 0.975)


def holm(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda x: x[1])
    out: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = min(1.0, running)
    return out


def canonical_block_value(row: dict[str, str], metric: str) -> float:
    aliases = {
        "gross_wh_per_case": ("gross_wh_per_case", "gross_gpu_energy_wh_per_case"),
        "net_wh_per_case": ("net_wh_per_case", "net_gpu_energy_wh_per_case"),
        "median_case_seconds": ("median_case_seconds", "median_case_elapsed_seconds"),
    }
    for key in aliases[metric]:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise RuntimeError(f"No supported field for {metric}; columns={sorted(row)}")


def load_blocks() -> dict[str, dict[int, dict[str, str]]]:
    two = read_csv(TWO / "block_summary.csv")
    med = read_csv(MED / "block_summary.csv")
    out = {
        "Qwen": {int(r["block"]): r for r in two if r.get("model") == "Qwen2.5-VL-7B-Instruct"},
        "InternVL3": {int(r["block"]): r for r in two if r.get("model") == "InternVL3-8B"},
        "MedGemma": {int(r["block"]): r for r in med},
    }
    expected = set(range(1, 11))
    for name, rows in out.items():
        if set(rows) != expected:
            raise RuntimeError(f"Expected matched blocks 1..10 for {name}")
    return out


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def select_review_ids(manifest: dict[int, dict[str, str]]) -> list[int]:
    strata: dict[tuple[str, str], list[int]] = {}
    for idx, row in manifest.items():
        strata.setdefault((row["normal_metadata_stratum"], row["reference_length_quartile"]), []).append(idx)
    shuffled: dict[tuple[str, str], list[int]] = {}
    for key, source_ids in strata.items():
        ids = list(source_ids)
        local = random.Random(f"{SEED}:{key}")
        local.shuffle(ids)
        shuffled[key] = ids

    selected: list[int] = []
    used: dict[tuple[str, str], int] = {key: 0 for key in shuffled}
    base, rem = divmod(REVIEW_N, len(shuffled))
    for sidx, key in enumerate(sorted(shuffled)):
        target = base + (1 if sidx < rem else 0)
        take = min(target, len(shuffled[key]))
        selected.extend(shuffled[key][:take])
        used[key] = take

    # Some strata contain fewer cases than their equal-allocation target. Fill the
    # remaining slots deterministically in balanced round-robin fashion from strata
    # with residual capacity, instead of shrinking the clinical-review packet.
    ordered_keys = sorted(shuffled)
    while len(selected) < REVIEW_N:
        added = False
        for key in ordered_keys:
            pos = used[key]
            if pos < len(shuffled[key]):
                selected.append(shuffled[key][pos])
                used[key] += 1
                added = True
                if len(selected) == REVIEW_N:
                    break
        if not added:
            break

    if len(selected) != REVIEW_N:
        raise RuntimeError(f"Expected {REVIEW_N} review cases after capacity-aware allocation, found {len(selected)}")
    if len(set(selected)) != REVIEW_N:
        raise RuntimeError("Duplicate case selected for clinical review")
    return sorted(selected)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    blocks = load_blocks()

    operational = []
    op_ps = []
    for metric in ["gross_wh_per_case", "net_wh_per_case", "median_case_seconds"]:
        for a, b in PAIR_SPECS:
            av = [canonical_block_value(blocks[a][j], metric) for j in range(1, 11)]
            bv = [canonical_block_value(blocks[b][j], metric) for j in range(1, 11)]
            est, lo, hi = paired_bootstrap_ratio(av, bv, rng)
            diffs = [x - y for x, y in zip(av, bv)]
            p = exact_sign_p(diffs)
            key = f"{metric}:{a}/{b}"
            op_ps.append((key, p))
            operational.append({"metric": metric, "numerator_model": a, "denominator_model": b, "ratio_of_paired_block_means": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "numerator_higher_blocks": sum(x > 0 for x in diffs), "numerator_lower_blocks": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_blocks": 10, "holm_p_across_all_operational_pairwise_tests": None})
    adj = holm(op_ps)
    for r in operational:
        r["holm_p_across_all_operational_pairwise_tests"] = adj[f"{r['metric']}:{r['numerator_model']}/{r['denominator_model']}"]
    progress(1, 4, "Operational pairwise bootstrap complete")

    manifest = index_rows(read_csv(TWO / "case_manifest_100.csv"), 100, "manifest")
    cases = {
        "Qwen": index_rows(read_csv(TWO / "qwen_case_review_100.csv"), 100, "Qwen"),
        "InternVL3": index_rows(read_csv(TWO / "internvl_case_results_100.csv"), 100, "InternVL3"),
        "MedGemma": index_rows(read_csv(MED / "case_results_100.csv"), 100, "MedGemma"),
    }

    utility = []
    util_ps = []
    for metric in ["unigram_f1", "rouge_l_f1"]:
        for a, b in PAIR_SPECS:
            av = [float(cases[a][j][metric]) for j in range(1, 101)]
            bv = [float(cases[b][j][metric]) for j in range(1, 101)]
            est, lo, hi = paired_bootstrap_diff(av, bv, rng)
            diffs = [x - y for x, y in zip(av, bv)]
            p = exact_sign_p(diffs)
            key = f"{metric}:{a}-{b}"
            util_ps.append((key, p))
            utility.append({"metric": metric, "model_a": a, "model_b": b, "mean_a": statistics.mean(av), "mean_b": statistics.mean(bv), "mean_paired_difference_a_minus_b": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "a_higher_cases": sum(x > 0 for x in diffs), "b_higher_cases": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_cases": 100, "holm_p_across_all_utility_pairwise_tests": None})
    uadj = holm(util_ps)
    for r in utility:
        r["holm_p_across_all_utility_pairwise_tests"] = uadj[f"{r['metric']}:{r['model_a']}-{r['model_b']}"]
    progress(2, 4, "Utility pairwise bootstrap complete")

    review_ids = select_review_ids(manifest)
    blind = random.Random(SEED + 17)
    review_rows, key_rows = [], []
    for idx in review_ids:
        mapping = ["Qwen", "InternVL3", "MedGemma"]
        blind.shuffle(mapping)
        q, i, m = cases["Qwen"][idx], cases["InternVL3"][idx], cases["MedGemma"][idx]
        findings = q.get("reference_findings", "") or i.get("reference_findings", "") or m.get("reference_findings", "")
        impression = q.get("reference_impression", "") or i.get("reference_impression", "") or m.get("reference_impression", "")
        row = {"review_case_id": f"R{idx:03d}", "case_index": idx, "normal_metadata_stratum": manifest[idx]["normal_metadata_stratum"], "reference_length_quartile": manifest[idx]["reference_length_quartile"], "reference_findings": findings, "reference_impression": impression, "output_A": cases[mapping[0]][idx]["model_output"], "output_B": cases[mapping[1]][idx]["model_output"], "output_C": cases[mapping[2]][idx]["model_output"]}
        for field in ["major_abnormality_correct", "clinically_important_omission", "clinically_important_hallucination", "laterality_or_location_error", "critical_safety_error", "overall_acceptable"]:
            for letter in "ABC":
                row[f"{field}_{letter}"] = ""
        row["reviewer_notes"] = ""
        review_rows.append(row)
        key_rows.append({"review_case_id": f"R{idx:03d}", "A": mapping[0], "B": mapping[1], "C": mapping[2]})
    write_csv(OUT / "clinical_review_blinded_50.csv", review_rows)
    write_csv(OUT / "clinical_review_unblinding_key.csv", key_rows)
    progress(3, 4, "Blinded clinical-review packet prepared")

    write_csv(OUT / "operational_pairwise.csv", operational)
    write_csv(OUT / "utility_pairwise.csv", utility)
    summary = {"status": "WP3_THREE_MODEL_100CASE_ANALYSIS_OK", "seed": SEED, "bootstrap_iterations": BOOT, "models": ["MedGemma-4B", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"], "operational_pairwise": operational, "utility_pairwise": utility, "clinical_review_packet": {"n_cases": REVIEW_N, "selection": "Deterministic capacity-aware balanced sample across MeSH normal/non-normal metadata status and reference-length quartile; short strata are exhausted and residual slots are filled round-robin from strata with remaining capacity.", "blinding": "Model identities randomized independently by case across output columns A/B/C. Unblinding key stored separately.", "adjudication_status": "Not performed. Blank fields are provided for human clinical review."}, "interpretation": {"energy_scope": "Direct NVIDIA GPU board operational energy; gross primary and idle-adjusted net secondary.", "operational_inference_unit": "Ten matched 10-case blocks.", "utility_inference_unit": "One hundred paired cases.", "utility_limit": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.", "multiplicity": "Holm adjustment is reported separately within the nine operational pairwise sign tests and six utility pairwise sign tests."}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "analysis_report.md").write_text("# WP3 three-model 100-case paired analysis\n\nOperational comparisons use ten matched 10-case blocks. Utility comparisons use 100 paired cases. Lexical metrics are screening endpoints only. A deterministic capacity-aware 50-case blinded clinical-review packet was prepared; no clinical adjudication was performed.\n", encoding="utf-8")
    progress(4, 4, "Three-model analysis complete")
    print(summary["status"])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
