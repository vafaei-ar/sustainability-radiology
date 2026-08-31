from __future__ import annotations

import csv
import json
import os
import pathlib
import time

import run_wp3_medgemma_token_budget_100case as m

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp3" / "medgemma_token_budget_100case"
BLOCK_PATH = OUT / "block_summary.csv"
CASE_PATH = OUT / "case_results_300.csv"
PAIRWISE_PATH = OUT / "pairwise_comparisons.csv"
SUMMARY_PATH = OUT / "summary.json"


def progress(current: int, total: int, phase: str) -> None:
    raw = os.environ.get("RUNRELAY_PROGRESS_FILE")
    if not raw:
        return
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "current": current,
        "total": total,
        "fraction": current / total if total else None,
        "phase": phase,
        "unit": "recovery stages",
        "updated_at_epoch": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Missing completed-run artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_union_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    block_rows = read_csv(BLOCK_PATH)
    case_rows = read_csv(CASE_PATH)
    budgets = tuple(m.TOKEN_BUDGETS)

    if len(block_rows) != 30:
        raise RuntimeError(f"Expected 30 completed block rows, found {len(block_rows)}")
    if len(case_rows) != 300:
        raise RuntimeError(f"Expected 300 completed case rows, found {len(case_rows)}")
    for budget in budgets:
        bs = [r for r in block_rows if int(r["token_budget"]) == budget]
        cs = [r for r in case_rows if int(r["token_budget"]) == budget]
        if len(bs) != 10 or sorted(int(r["block"]) for r in bs) != list(range(1, 11)):
            raise RuntimeError(f"Incomplete block coverage for budget {budget}")
        indices = sorted(int(r["case_index"]) for r in cs)
        if len(cs) != 100 or indices != list(range(1, 101)):
            raise RuntimeError(f"Incomplete case coverage for budget {budget}")

    if not all(r.get("radgraph_rg_er_f1", "") not in ("", None) for r in case_rows):
        raise RuntimeError("Saved 300-case output is missing completed RG_ER scores; refusing to infer or rerun GPU work")
    for row in case_rows:
        float(row["radgraph_rg_er_f1"])
    progress(1, 3, "Validated 30 energy blocks, 300 case outputs, and 300 saved RadGraph scores")

    summaries = {}
    for budget in budgets:
        bs = [r for r in block_rows if int(r["token_budget"]) == budget]
        cs = [r for r in case_rows if int(r["token_budget"]) == budget]
        summaries[str(budget)] = {
            "gross_wh_per_case_mean": m.mean(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "gross_wh_per_case_median": m.median(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "gross_block_cv": m.cv(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "net_wh_per_case_mean": m.mean(r["net_gpu_energy_wh_per_case"] for r in bs),
            "median_case_seconds": m.median(r["median_case_elapsed_seconds"] for r in bs),
            "mean_output_tokens": m.mean(r["output_token_count"] for r in cs),
            "median_output_tokens": m.median(r["output_token_count"] for r in cs),
            "near_token_cap_fraction": m.mean(r["near_token_cap"] for r in cs),
            "mean_unigram_f1": m.mean(r["unigram_f1"] for r in cs),
            "mean_rouge_l_f1": m.mean(r["rouge_l_f1"] for r in cs),
            "mean_radgraph_rg_er_f1": m.mean(r["radgraph_rg_er_f1"] for r in cs),
        }

    comparisons = []
    for a, b in ((64, 128), (32, 128), (32, 64)):
        ar = sorted([r for r in block_rows if int(r["token_budget"]) == a], key=lambda r: int(r["block"]))
        br = sorted([r for r in block_rows if int(r["token_budget"]) == b], key=lambda r: int(r["block"]))
        av = [float(r["gross_gpu_energy_wh_per_case"]) for r in ar]
        bv = [float(r["gross_gpu_energy_wh_per_case"]) for r in br]
        est, lo, hi = m.bootstrap_ratio(av, bv, m.BOOTSTRAP_REPS, m.BOOTSTRAP_SEED + a + b)
        comparisons.append({
            "metric": "gross_gpu_energy_wh_per_case",
            "numerator_budget": a,
            "denominator_budget": b,
            "estimate": est,
            "ci95_low": lo,
            "ci95_high": hi,
            "effect_scale": "ratio",
            "exact_sign_p_two_sided": m.exact_sign_p(av, bv),
        })
    m.holm(comparisons, "exact_sign_p_two_sided", "holm_p_within_energy_family")

    by_budget = {
        budget: sorted([r for r in case_rows if int(r["token_budget"]) == budget], key=lambda r: int(r["case_index"]))
        for budget in budgets
    }
    utility_rows = []
    for a, b in ((64, 128), (32, 128), (32, 64)):
        av = [float(r["radgraph_rg_er_f1"]) for r in by_budget[a]]
        bv = [float(r["radgraph_rg_er_f1"]) for r in by_budget[b]]
        est, lo, hi = m.bootstrap_difference(av, bv, m.BOOTSTRAP_REPS, m.BOOTSTRAP_SEED + a + b + 1000)
        utility_rows.append({
            "metric": "radgraph_rg_er_f1",
            "numerator_budget": a,
            "denominator_budget": b,
            "estimate": est,
            "ci95_low": lo,
            "ci95_high": hi,
            "effect_scale": "mean paired difference",
            "exact_sign_p_two_sided": m.exact_sign_p(av, bv),
        })
    m.holm(utility_rows, "exact_sign_p_two_sided", "holm_p_within_radgraph_family")
    pairwise_rows = comparisons + utility_rows
    write_union_csv(PAIRWISE_PATH, pairwise_rows)
    progress(2, 3, "Reconstructed paired energy and RadGraph inference with endpoint-specific Holm correction")

    pareto = []
    for budget in budgets:
        s = summaries[str(budget)]
        dominated_by = []
        for other in budgets:
            if other == budget:
                continue
            o = summaries[str(other)]
            if (
                o["gross_wh_per_case_mean"] <= s["gross_wh_per_case_mean"]
                and o["mean_radgraph_rg_er_f1"] >= s["mean_radgraph_rg_er_f1"]
                and (
                    o["gross_wh_per_case_mean"] < s["gross_wh_per_case_mean"]
                    or o["mean_radgraph_rg_er_f1"] > s["mean_radgraph_rg_er_f1"]
                )
            ):
                dominated_by.append(other)
        pareto.append({"token_budget": budget, "dominated": bool(dominated_by), "dominated_by": dominated_by})

    block_orders = {}
    for block in range(1, 11):
        rows = sorted([r for r in block_rows if int(r["block"]) == block], key=lambda r: int(r["execution_order_within_block"]))
        block_orders[str(block)] = [int(r["token_budget"]) for r in rows]

    summary = {
        "status": "WP3_MEDGEMMA_TOKEN_BUDGET_100CASE_RECOVERED",
        "recovery_note": "All GPU measurements and F1-RadGraph RG_ER scoring completed in R8M5Q2K7. This analysis-only recovery reconstructs final statistics from the saved 30 block rows and 300 case rows after the original mixed-schema CSV serialization failure; no GPU inference was repeated.",
        "model": "MedGemma-4B",
        "repo_id": m.MODEL_ID,
        "revision": m.MODEL_REVISION,
        "cases": 100,
        "token_budgets": list(budgets),
        "blocks_per_budget": 10,
        "block_size": m.BLOCK_SIZE,
        "prompt": m.PROMPT,
        "block_execution_orders": block_orders,
        "measurement_scope": "Direct NVIDIA GPU board operational energy; model loading and warmup excluded; gross primary and idle-adjusted net secondary.",
        "primary_operational_endpoint": "gross GPU-board Wh per completed case",
        "primary_utility_endpoint": "F1-RadGraph RG_ER candidate-reference fidelity",
        "radgraph": {
            "status": "ok_saved_scores_validated",
            "version": m.RADGRAPH_VERSION,
            "model_type": m.RADGRAPH_MODEL_TYPE,
            "score_component": "RG_ER",
            "scores_validated": 300,
        },
        "budget_summaries": summaries,
        "pairwise_comparisons": pairwise_rows,
        "energy_utility_pareto": pareto,
        "interpretation_limit": "F1-RadGraph measures candidate-reference factual/structural agreement, not image-grounded diagnostic accuracy or radiologist-adjudicated clinical safety. Token caps can truncate reports; near-cap output frequency is reported explicitly.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    progress(3, 3, "Recovery outputs complete")
    print("WP3_MEDGEMMA_TOKEN_BUDGET_RECOVERY_OK")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
