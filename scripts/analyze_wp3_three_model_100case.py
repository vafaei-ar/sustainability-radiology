from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import statistics
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
MED = ROOT / "results" / "wp3" / "medgemma_100case"
OUT = ROOT / "results" / "wp3" / "three_model_100case_analysis"
BOOT = 20000
SEED = 20260831
REVIEW_N = 50
PAIR_SPECS = [("MedGemma", "Qwen"), ("MedGemma", "InternVL3"), ("Qwen", "InternVL3")]
MODEL_LABELS = {
    "MedGemma": "MedGemma-4B",
    "Qwen": "Qwen2.5-VL-7B-Instruct",
    "InternVL3": "InternVL3-8B",
}
ADJUDICATION_FIELDS = [
    "major_abnormality_correct",
    "clinically_important_omission",
    "clinically_important_hallucination",
    "laterality_or_location_error",
    "critical_safety_error",
    "overall_acceptable",
]


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    p = pathlib.Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "current": current,
                "total": total,
                "fraction": current / total,
                "phase": phase,
                "unit": "analysis stages",
                "updated_at_epoch": time.time(),
            }
        ),
        encoding="utf-8",
    )
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
        w.writeheader()
        w.writerows(rows)


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


def qa_review_packet(review_rows: list[dict], key_rows: list[dict]) -> dict:
    errors: list[str] = []
    review_ids = [r["review_case_id"] for r in review_rows]
    case_indices = [int(r["case_index"]) for r in review_rows]
    if len(review_rows) != REVIEW_N:
        errors.append(f"review row count={len(review_rows)}")
    if len(set(review_ids)) != REVIEW_N:
        errors.append("review_case_id values are not unique")
    if len(set(case_indices)) != REVIEW_N:
        errors.append("case_index values are not unique")
    if {r["review_case_id"] for r in key_rows} != set(review_ids):
        errors.append("unblinding key IDs do not match review packet IDs")

    model_set = set(MODEL_LABELS)
    for row in review_rows:
        for letter in "ABC":
            if not str(row.get(f"output_{letter}", "")).strip():
                errors.append(f"empty output_{letter} for {row['review_case_id']}")
            for field in ADJUDICATION_FIELDS:
                if str(row.get(f"{field}_{letter}", "")).strip():
                    errors.append(f"nonblank adjudication field {field}_{letter} for {row['review_case_id']}")
    for row in key_rows:
        mapped = {row["A"], row["B"], row["C"]}
        if mapped != model_set:
            errors.append(f"invalid model permutation for {row['review_case_id']}")

    column_balance = {letter: Counter(row[letter] for row in key_rows) for letter in "ABC"}
    stratum_counts = Counter((r["normal_metadata_stratum"], r["reference_length_quartile"]) for r in review_rows)
    normal_counts = Counter(r["normal_metadata_stratum"] for r in review_rows)
    quartile_counts = Counter(r["reference_length_quartile"] for r in review_rows)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "review_cases": len(review_rows),
        "unique_case_indices": len(set(case_indices)),
        "all_outputs_nonempty": not any(x.startswith("empty output_") for x in errors),
        "all_adjudication_fields_blank": not any(x.startswith("nonblank adjudication") for x in errors),
        "unblinding_key_valid": not any("unblinding" in x or "permutation" in x for x in errors),
        "normal_metadata_counts": dict(sorted(normal_counts.items())),
        "reference_length_quartile_counts": dict(sorted(quartile_counts.items())),
        "joint_stratum_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(stratum_counts.items())},
        "blinded_column_model_counts": {letter: dict(sorted(counts.items())) for letter, counts in column_balance.items()},
    }


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def build_report(operational: list[dict], utility: list[dict], qa: dict) -> str:
    op_lines = [
        "| Endpoint | Comparison | Ratio | 95% bootstrap CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for r in operational:
        comparison = f"{MODEL_LABELS[r['numerator_model']]} / {MODEL_LABELS[r['denominator_model']]}"
        op_lines.append(
            f"| {r['metric']} | {comparison} | {r['ratio_of_paired_block_means']:.3f} | {fmt_ci(r['bootstrap_ci_low'], r['bootstrap_ci_high'])} | {r['holm_p_across_all_operational_pairwise_tests']:.4g} |"
        )

    util_lines = [
        "| Metric | Comparison | Mean A | Mean B | Paired difference A-B | 95% bootstrap CI | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in utility:
        comparison = f"{MODEL_LABELS[r['model_a']]} vs {MODEL_LABELS[r['model_b']]}"
        util_lines.append(
            f"| {r['metric']} | {comparison} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_paired_difference_a_minus_b']:.3f} | {fmt_ci(r['bootstrap_ci_low'], r['bootstrap_ci_high'])} | {r['holm_p_across_all_utility_pairwise_tests']:.4g} |"
        )

    med_q_gross = next(r for r in operational if r["metric"] == "gross_wh_per_case" and r["numerator_model"] == "MedGemma" and r["denominator_model"] == "Qwen")
    med_i_gross = next(r for r in operational if r["metric"] == "gross_wh_per_case" and r["numerator_model"] == "MedGemma" and r["denominator_model"] == "InternVL3")
    med_q_net = next(r for r in operational if r["metric"] == "net_wh_per_case" and r["numerator_model"] == "MedGemma" and r["denominator_model"] == "Qwen")
    med_q_time = next(r for r in operational if r["metric"] == "median_case_seconds" and r["numerator_model"] == "MedGemma" and r["denominator_model"] == "Qwen")
    med_q_uni = next(r for r in utility if r["metric"] == "unigram_f1" and r["model_a"] == "MedGemma" and r["model_b"] == "Qwen")
    med_q_rouge = next(r for r in utility if r["metric"] == "rouge_l_f1" and r["model_a"] == "MedGemma" and r["model_b"] == "Qwen")

    manuscript = (
        f"Across ten matched 10-case blocks, MedGemma-4B required {100 * (1 - med_q_gross['ratio_of_paired_block_means']):.1f}% less gross GPU-board energy per case than Qwen2.5-VL-7B-Instruct "
        f"(ratio {med_q_gross['ratio_of_paired_block_means']:.3f}, 95% bootstrap CI {fmt_ci(med_q_gross['bootstrap_ci_low'], med_q_gross['bootstrap_ci_high'])}) and "
        f"{100 * (1 - med_i_gross['ratio_of_paired_block_means']):.1f}% less than InternVL3-8B "
        f"(ratio {med_i_gross['ratio_of_paired_block_means']:.3f}, 95% CI {fmt_ci(med_i_gross['bootstrap_ci_low'], med_i_gross['bootstrap_ci_high'])}). "
        f"The idle-adjusted MedGemma/Qwen energy ratio was {med_q_net['ratio_of_paired_block_means']:.3f} (95% CI {fmt_ci(med_q_net['bootstrap_ci_low'], med_q_net['bootstrap_ci_high'])}). "
        f"Runtime did not clearly differ between MedGemma and Qwen (ratio {med_q_time['ratio_of_paired_block_means']:.3f}, 95% CI {fmt_ci(med_q_time['bootstrap_ci_low'], med_q_time['bootstrap_ci_high'])}). "
        f"On lexical screening across 100 paired cases, MedGemma exceeded Qwen by {med_q_uni['mean_paired_difference_a_minus_b']:.3f} in unigram F1 and {med_q_rouge['mean_paired_difference_a_minus_b']:.3f} in ROUGE-L. "
        "These lexical metrics are screening endpoints and do not establish clinical adequacy."
    )

    qa_lines = [
        f"Packet QA status: **{qa['status'].upper()}**.",
        f"The packet contains {qa['review_cases']} unique blinded cases; all model outputs are nonempty: {qa['all_outputs_nonempty']}; all adjudication fields are blank: {qa['all_adjudication_fields_blank']}; unblinding key valid: {qa['unblinding_key_valid']}.",
        f"Normal-metadata counts: `{json.dumps(qa['normal_metadata_counts'], sort_keys=True)}`.",
        f"Reference-length quartile counts: `{json.dumps(qa['reference_length_quartile_counts'], sort_keys=True)}`.",
        f"Blinded-column model balance: `{json.dumps(qa['blinded_column_model_counts'], sort_keys=True)}`.",
    ]

    reviewer = [
        "Review outputs A, B, and C without consulting the unblinding key.",
        "Use the reference findings and impression as the comparison text, while recognizing that the reference report is not an independent expert re-adjudication.",
        "Complete major-abnormality correctness, clinically important omission, clinically important hallucination, laterality/location error, critical safety error, and overall acceptability for each output.",
        "Keep uncertainty explicit in reviewer notes rather than forcing a favorable or unfavorable judgment when the reference is ambiguous.",
        "Do not use unigram F1 or ROUGE-L when adjudicating clinical acceptability.",
    ]

    figure_lines = [
        "The operational-pairwise CSV is the figure-ready source for a ratio forest plot. The null line is 1.0; ratios below 1 favor the numerator model for lower resource use.",
        "The utility-pairwise CSV is the figure-ready source for paired lexical-difference plots. Positive A-B differences favor model A, but these should remain labeled as lexical screening rather than clinical utility.",
    ]

    return "\n".join(
        [
            "# WP3 three-model 100-case publication bundle",
            "",
            "This bundled CPU-only step reruns the paired analysis, performs structural QA of the blinded clinical-review packet, and materializes manuscript-ready tables and wording from the frozen three-model results. No clinical adjudication is performed.",
            "",
            "## Operational comparisons",
            "",
            *op_lines,
            "",
            "Operational inference uses ten matched 10-case blocks. Energy is direct NVIDIA GPU-board operational energy; gross energy is primary and idle-adjusted energy is secondary.",
            "",
            "## Lexical screening comparisons",
            "",
            *util_lines,
            "",
            "Lexical inference uses 100 paired cases. Unigram F1 and ROUGE-L do not establish clinical adequacy.",
            "",
            "## Clinical-review packet QA",
            "",
            *qa_lines,
            "",
            "## Reviewer instructions",
            "",
            *[f"- {x}" for x in reviewer],
            "",
            "## Manuscript-ready results paragraph",
            "",
            manuscript,
            "",
            "## Figure-ready guidance",
            "",
            *figure_lines,
            "",
            "## Interpretation guardrails",
            "",
            "The 100-case benchmark is a deterministic stratified single-image Open-I workload, not a population-performance estimate. The MeSH normal/non-normal field is metadata, not clinical adjudication. Gross GPU-board energy should not be described as total-system energy. Clinical efficiency claims should wait for blinded human review and a prespecified acceptable-utility rule.",
            "",
        ]
    )


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
            operational.append(
                {
                    "metric": metric,
                    "numerator_model": a,
                    "denominator_model": b,
                    "ratio_of_paired_block_means": est,
                    "percent_change_numerator_vs_denominator": 100 * (est - 1),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "exact_sign_p_two_sided": p,
                    "numerator_higher_blocks": sum(x > 0 for x in diffs),
                    "numerator_lower_blocks": sum(x < 0 for x in diffs),
                    "ties": sum(x == 0 for x in diffs),
                    "paired_blocks": 10,
                    "holm_p_across_all_operational_pairwise_tests": None,
                }
            )
    adj = holm(op_ps)
    for r in operational:
        r["holm_p_across_all_operational_pairwise_tests"] = adj[f"{r['metric']}:{r['numerator_model']}/{r['denominator_model']}"]
    progress(1, 5, "Operational pairwise bootstrap complete")

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
            utility.append(
                {
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
                }
            )
    uadj = holm(util_ps)
    for r in utility:
        r["holm_p_across_all_utility_pairwise_tests"] = uadj[f"{r['metric']}:{r['model_a']}-{r['model_b']}"]
    progress(2, 5, "Utility pairwise bootstrap complete")

    review_ids = select_review_ids(manifest)
    blind = random.Random(SEED + 17)
    review_rows, key_rows = [], []
    for idx in review_ids:
        mapping = ["Qwen", "InternVL3", "MedGemma"]
        blind.shuffle(mapping)
        q, i, m = cases["Qwen"][idx], cases["InternVL3"][idx], cases["MedGemma"][idx]
        findings = q.get("reference_findings", "") or i.get("reference_findings", "") or m.get("reference_findings", "")
        impression = q.get("reference_impression", "") or i.get("reference_impression", "") or m.get("reference_impression", "")
        row = {
            "review_case_id": f"R{idx:03d}",
            "case_index": idx,
            "normal_metadata_stratum": manifest[idx]["normal_metadata_stratum"],
            "reference_length_quartile": manifest[idx]["reference_length_quartile"],
            "reference_findings": findings,
            "reference_impression": impression,
            "output_A": cases[mapping[0]][idx]["model_output"],
            "output_B": cases[mapping[1]][idx]["model_output"],
            "output_C": cases[mapping[2]][idx]["model_output"],
        }
        for field in ADJUDICATION_FIELDS:
            for letter in "ABC":
                row[f"{field}_{letter}"] = ""
        row["reviewer_notes"] = ""
        review_rows.append(row)
        key_rows.append({"review_case_id": f"R{idx:03d}", "A": mapping[0], "B": mapping[1], "C": mapping[2]})
    progress(3, 5, "Blinded clinical-review packet prepared")

    qa = qa_review_packet(review_rows, key_rows)
    if qa["status"] != "pass":
        raise RuntimeError("Clinical-review packet QA failed: " + "; ".join(qa["errors"]))
    progress(4, 5, "Clinical-review packet QA passed")

    write_csv(OUT / "clinical_review_blinded_50.csv", review_rows)
    write_csv(OUT / "clinical_review_unblinding_key.csv", key_rows)
    write_csv(OUT / "operational_pairwise.csv", operational)
    write_csv(OUT / "utility_pairwise.csv", utility)

    summary = {
        "status": "WP3_THREE_MODEL_100CASE_PUBLICATION_BUNDLE_OK",
        "seed": SEED,
        "bootstrap_iterations": BOOT,
        "models": ["MedGemma-4B", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"],
        "operational_pairwise": operational,
        "utility_pairwise": utility,
        "clinical_review_packet": {
            "n_cases": REVIEW_N,
            "selection": "Deterministic capacity-aware balanced sample across MeSH normal/non-normal metadata status and reference-length quartile; short strata are exhausted and residual slots are filled round-robin from strata with remaining capacity.",
            "blinding": "Model identities randomized independently by case across output columns A/B/C. Unblinding key stored separately.",
            "adjudication_status": "Not performed. Blank fields are provided for human clinical review.",
            "qa": qa,
        },
        "interpretation": {
            "energy_scope": "Direct NVIDIA GPU board operational energy; gross primary and idle-adjusted net secondary.",
            "operational_inference_unit": "Ten matched 10-case blocks.",
            "utility_inference_unit": "One hundred paired cases.",
            "utility_limit": "Unigram F1 and ROUGE-L are lexical screening metrics only and do not establish clinical adequacy.",
            "multiplicity": "Holm adjustment is reported separately within the nine operational pairwise sign tests and six utility pairwise sign tests.",
            "cohort_limit": "Deterministic stratified exactly-one-image Open-I single-image workload; not a population-performance estimate.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "analysis_report.md").write_text(build_report(operational, utility, qa), encoding="utf-8")
    progress(5, 5, "Publication bundle complete")
    print(summary["status"])
    print(json.dumps({"packet_qa": qa, "status": summary["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
