from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import threading
import time

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model" / "case_manifest_100.csv"
OUT_DIR = ROOT / "results" / "wp3" / "medgemma_token_budget_100case"
HELPER_PATH = ROOT / "scripts" / "run_wp3_remaining_bf16_panel_pilot.py"
MODEL_ID = "google/medgemma-4b-it"
MODEL_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"
PROMPT = "Describe the chest radiograph findings concisely. Do not infer patient identity."
TOKEN_BUDGETS = (128, 64, 32)
BLOCK_SIZE = 10
IDLE_SECONDS = 8.0
ORDER_SEED = 20260831
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20260831
RADGRAPH_VERSION = "0.1.18"
RADGRAPH_MODEL_TYPE = "radgraph-xl"
WP3_PY = ROOT / ".venv-wp3" / "bin" / "python"
METRIC_TARGET = ROOT / ".wp3-metrics-packages"


def load_helper():
    spec = importlib.util.spec_from_file_location("wp3_medgemma_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load WP3 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "unit": "measurement stages",
        "updated_at_epoch": time.time(),
    }
    path = pathlib.Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def mean(values):
    vals = [float(x) for x in values]
    return statistics.mean(vals) if vals else None


def median(values):
    vals = [float(x) for x in values]
    return statistics.median(vals) if vals else None


def cv(values):
    vals = [float(x) for x in values]
    if len(vals) < 2:
        return None
    m = statistics.mean(vals)
    return statistics.stdev(vals) / m if m else None


def bootstrap_ratio(numer: list[float], denom: list[float], reps: int, seed: int):
    if len(numer) != len(denom) or not numer:
        raise ValueError("Paired vectors required")
    rng = random.Random(seed)
    n = len(numer)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        a = statistics.mean(numer[i] for i in idx)
        b = statistics.mean(denom[i] for i in idx)
        if b != 0:
            vals.append(a / b)
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return statistics.mean(numer) / statistics.mean(denom), lo, hi


def bootstrap_difference(numer: list[float], denom: list[float], reps: int, seed: int):
    if len(numer) != len(denom) or not numer:
        raise ValueError("Paired vectors required")
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(numer, denom)]
    n = len(diffs)
    vals = []
    for _ in range(reps):
        vals.append(statistics.mean(diffs[rng.randrange(n)] for _ in range(n)))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return statistics.mean(diffs), lo, hi


def exact_sign_p(a: list[float], b: list[float]) -> float:
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    k = min(sum(x > 0 for x in d), sum(x < 0 for x in d))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm(rows: list[dict], p_key: str, out_key: str) -> None:
    ordered = sorted(enumerate(rows), key=lambda x: float(x[1][p_key]))
    m = len(rows)
    running = 0.0
    adjusted = [1.0] * m
    for rank, (idx, row) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(row[p_key]))
        running = max(running, value)
        adjusted[idx] = running
    for row, adj in zip(rows, adjusted):
        row[out_key] = adj


def run_radgraph(pairs: list[dict]) -> tuple[str, list[float] | None, str | None]:
    if not WP3_PY.is_file():
        return "unavailable", None, ".venv-wp3 Python not found"
    METRIC_TARGET.mkdir(exist_ok=True)
    try:
        install = [
            str(WP3_PY), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--target", str(METRIC_TARGET), "--upgrade", "--no-deps",
            f"radgraph=={RADGRAPH_VERSION}",
            "appdirs", "dotmap", "jsonpickle", "h5py", "nltk", "defusedxml",
        ]
        install_run = subprocess.run(
            install, cwd=ROOT, check=False, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=600,
        )
        if install_run.returncode != 0:
            return "failed", None, (install_run.stdout or "")[-4000:]

        payload_path = OUT_DIR / ".radgraph_pairs.json"
        result_path = OUT_DIR / ".radgraph_result.json"
        payload_path.write_text(json.dumps(pairs) + "\n", encoding="utf-8")
        code = r'''
import json, pathlib, sys
from radgraph import F1RadGraph
pairs = json.loads(pathlib.Path(sys.argv[1]).read_text())
refs = [x["reference"] for x in pairs]
hyps = [x["candidate"] for x in pairs]
scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=-1)
mean_reward, reward_list, _, _ = scorer(hyps=hyps, refs=refs)
rg_e, rg_er, rg_bar_er = reward_list
pathlib.Path(sys.argv[2]).write_text(json.dumps({
    "mean_reward": [float(x) for x in mean_reward],
    "rg_e": [float(x) for x in rg_e],
    "rg_er": [float(x) for x in rg_er],
    "rg_bar_er": [float(x) for x in rg_bar_er],
}))
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(METRIC_TARGET) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        metric_run = subprocess.run(
            [str(WP3_PY), "-c", code, str(payload_path), str(result_path)],
            cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=2400, env=env,
        )
        if metric_run.returncode != 0:
            return "failed", None, (metric_run.stdout or "")[-6000:]
        obj = json.loads(result_path.read_text(encoding="utf-8"))
        scores = obj.get("rg_er")
        if not isinstance(scores, list) or len(scores) != len(pairs):
            return "raw_output_unparsed", None, f"Expected {len(pairs)} RG_ER scores"
        return "ok", [float(x) for x in scores], None
    except Exception as exc:
        return "failed", None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    helper = load_helper()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.is_file():
        raise RuntimeError("Frozen 100-case Open-I manifest missing")
    cases = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8", newline="")))
    if len(cases) != 100:
        raise RuntimeError(f"Expected 100 frozen cases, found {len(cases)}")
    if [int(r["case_index"]) for r in cases] != list(range(1, 101)):
        raise RuntimeError("Frozen case_index is not contiguous 1..100")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    refs = helper.extract_report_texts()
    snapshot = pathlib.Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(ROOT / ".wp3-models" / "MedGemma-4B-pinned"),
    )).resolve()
    pipe = pipeline(
        "image-text-to-text",
        model=str(snapshot),
        device=0,
        dtype=torch.bfloat16,
    )
    warm_img = Image.open(ROOT / cases[0]["local_image_path"]).convert("RGB")
    warm_out = pipe(helper.make_messages(warm_img), max_new_tokens=16, do_sample=False)
    if not helper.extract_generated_text(warm_out):
        raise RuntimeError("Empty MedGemma warmup output")
    del warm_img, warm_out
    torch.cuda.synchronize()

    rng = random.Random(ORDER_SEED)
    block_orders = {}
    for block_idx in range(1, 11):
        order = list(TOKEN_BUDGETS)
        rng.shuffle(order)
        block_orders[block_idx] = order

    block_rows: list[dict] = []
    case_rows: list[dict] = []
    total_stages = 31
    done = 0
    progress(done, total_stages, "MedGemma warmup complete")

    for block_idx in range(1, 11):
        block_cases = cases[(block_idx - 1) * BLOCK_SIZE:block_idx * BLOCK_SIZE]
        for order_pos, budget in enumerate(block_orders[block_idx], start=1):
            idle_rows = []
            idle_stop = threading.Event()
            idle_thread = threading.Thread(target=helper.sample_trace, args=(idle_stop, idle_rows), daemon=True)
            idle_thread.start()
            time.sleep(IDLE_SECONDS)
            idle_stop.set(); idle_thread.join()
            if not idle_rows:
                raise RuntimeError(f"No idle power samples for block {block_idx}, budget {budget}")
            idle_mean_power = statistics.mean(r[1] for r in idle_rows)

            trace = []
            stop = threading.Event()
            sampler = threading.Thread(target=helper.sample_trace, args=(stop, trace), daemon=True)
            sampler.start()
            block_start = time.perf_counter()
            elapsed_cases = []

            for case in block_cases:
                findings, impression = refs.get(case["source_report_id"], ("", ""))
                reference = " ".join(x for x in (findings, impression) if x).strip()
                img = Image.open(ROOT / case["local_image_path"]).convert("RGB")
                t0 = time.perf_counter()
                out = pipe(helper.make_messages(img), max_new_tokens=budget, do_sample=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                pred = helper.extract_generated_text(out)
                if not pred:
                    raise RuntimeError(f"Empty output for case {case['case_index']}, budget {budget}")
                token_count = len(pipe.tokenizer(pred, add_special_tokens=False)["input_ids"])
                elapsed_cases.append(elapsed)
                case_rows.append({
                    "token_budget": budget,
                    "block": block_idx,
                    "execution_order_within_block": order_pos,
                    "case_index": int(case["case_index"]),
                    "source_report_id": case["source_report_id"],
                    "source_image_id": case["source_image_id"],
                    "normal_metadata_stratum": case.get("normal_metadata_stratum", ""),
                    "reference_length_quartile": case.get("reference_length_quartile", ""),
                    "elapsed_seconds": elapsed,
                    "output_token_count": token_count,
                    "near_token_cap": int(token_count >= max(1, budget - 2)),
                    "model_word_count": len(helper.normalize_tokens(pred)),
                    "unigram_f1": helper.unigram_f1(pred, reference),
                    "rouge_l_f1": helper.rouge_l_f1(pred, reference),
                    "model_output": pred,
                    "reference_findings": findings,
                    "reference_impression": impression,
                })
                del img, out

            block_elapsed = time.perf_counter() - block_start
            stop.set(); sampler.join()
            gross_wh = helper.integrate_wh(trace)
            net_wh = max(0.0, gross_wh - idle_mean_power * block_elapsed / 3600.0)
            block_rows.append({
                "token_budget": budget,
                "block": block_idx,
                "execution_order_within_block": order_pos,
                "cases": len(block_cases),
                "gross_gpu_energy_wh_block": gross_wh,
                "gross_gpu_energy_wh_per_case": gross_wh / len(block_cases),
                "net_gpu_energy_wh_block": net_wh,
                "net_gpu_energy_wh_per_case": net_wh / len(block_cases),
                "idle_mean_power_w": idle_mean_power,
                "block_elapsed_seconds": block_elapsed,
                "median_case_elapsed_seconds": statistics.median(elapsed_cases),
                "mean_gpu_utilization_pct": statistics.mean(r[2] for r in trace) if trace else None,
                "peak_sampled_memory_mib": max(r[3] for r in trace) if trace else None,
            })
            done += 1
            progress(done, total_stages, f"Completed block {block_idx}/10 at max_new_tokens={budget}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    pairs = []
    for row in case_rows:
        reference = " ".join(x for x in (row["reference_findings"], row["reference_impression"]) if x).strip()
        pairs.append({"reference": reference, "candidate": row["model_output"]})
    radgraph_status, radgraph_scores, radgraph_error = run_radgraph(pairs)
    if radgraph_status == "ok" and radgraph_scores is not None:
        for row, score in zip(case_rows, radgraph_scores):
            row["radgraph_rg_er_f1"] = score
    else:
        for row in case_rows:
            row["radgraph_rg_er_f1"] = ""
    done += 1
    progress(done, total_stages, f"RadGraph {radgraph_status}")

    write_csv(OUT_DIR / "block_summary.csv", block_rows)
    write_csv(OUT_DIR / "case_results_300.csv", case_rows)

    summaries = {}
    for budget in TOKEN_BUDGETS:
        bs = [r for r in block_rows if int(r["token_budget"]) == budget]
        cs = [r for r in case_rows if int(r["token_budget"]) == budget]
        summaries[str(budget)] = {
            "gross_wh_per_case_mean": mean(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "gross_wh_per_case_median": median(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "gross_block_cv": cv(r["gross_gpu_energy_wh_per_case"] for r in bs),
            "net_wh_per_case_mean": mean(r["net_gpu_energy_wh_per_case"] for r in bs),
            "median_case_seconds": median(r["median_case_elapsed_seconds"] for r in bs),
            "mean_output_tokens": mean(r["output_token_count"] for r in cs),
            "median_output_tokens": median(r["output_token_count"] for r in cs),
            "near_token_cap_fraction": mean(r["near_token_cap"] for r in cs),
            "mean_unigram_f1": mean(r["unigram_f1"] for r in cs),
            "mean_rouge_l_f1": mean(r["rouge_l_f1"] for r in cs),
            "mean_radgraph_rg_er_f1": mean(r["radgraph_rg_er_f1"] for r in cs if r["radgraph_rg_er_f1"] != ""),
        }

    comparisons = []
    for a, b in ((64, 128), (32, 128), (32, 64)):
        a_blocks = sorted([r for r in block_rows if int(r["token_budget"]) == a], key=lambda r: int(r["block"]))
        b_blocks = sorted([r for r in block_rows if int(r["token_budget"]) == b], key=lambda r: int(r["block"]))
        ag = [float(r["gross_gpu_energy_wh_per_case"]) for r in a_blocks]
        bg = [float(r["gross_gpu_energy_wh_per_case"]) for r in b_blocks]
        ratio, lo, hi = bootstrap_ratio(ag, bg, BOOTSTRAP_REPS, BOOTSTRAP_SEED + a + b)
        comparisons.append({
            "metric": "gross_gpu_energy_wh_per_case",
            "numerator_budget": a,
            "denominator_budget": b,
            "estimate": ratio,
            "ci95_low": lo,
            "ci95_high": hi,
            "effect_scale": "ratio",
            "exact_sign_p_two_sided": exact_sign_p(ag, bg),
        })
    holm(comparisons, "exact_sign_p_two_sided", "holm_p_within_energy_family")

    utility_rows = []
    if radgraph_status == "ok":
        by_budget = {
            budget: sorted([r for r in case_rows if int(r["token_budget"]) == budget], key=lambda r: int(r["case_index"]))
            for budget in TOKEN_BUDGETS
        }
        for a, b in ((64, 128), (32, 128), (32, 64)):
            av = [float(r["radgraph_rg_er_f1"]) for r in by_budget[a]]
            bv = [float(r["radgraph_rg_er_f1"]) for r in by_budget[b]]
            diff, lo, hi = bootstrap_difference(av, bv, BOOTSTRAP_REPS, BOOTSTRAP_SEED + a + b + 1000)
            utility_rows.append({
                "metric": "radgraph_rg_er_f1",
                "numerator_budget": a,
                "denominator_budget": b,
                "estimate": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "effect_scale": "mean paired difference",
                "exact_sign_p_two_sided": exact_sign_p(av, bv),
            })
        holm(utility_rows, "exact_sign_p_two_sided", "holm_p_within_radgraph_family")

    pairwise_rows = comparisons + utility_rows
    write_csv(OUT_DIR / "pairwise_comparisons.csv", pairwise_rows)

    pareto = []
    if radgraph_status == "ok":
        for budget in TOKEN_BUDGETS:
            s = summaries[str(budget)]
            dominated_by = []
            for other in TOKEN_BUDGETS:
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

    summary = {
        "status": "WP3_MEDGEMMA_TOKEN_BUDGET_100CASE_COMPLETE",
        "model": "MedGemma-4B",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "cases": 100,
        "token_budgets": list(TOKEN_BUDGETS),
        "blocks_per_budget": 10,
        "block_size": BLOCK_SIZE,
        "prompt": PROMPT,
        "block_execution_orders": block_orders,
        "measurement_scope": "Direct NVIDIA GPU board operational energy; model loading and warmup excluded; gross primary and idle-adjusted net secondary.",
        "primary_operational_endpoint": "gross GPU-board Wh per completed case",
        "primary_utility_endpoint": "F1-RadGraph RG_ER candidate-reference fidelity when RadGraph completes",
        "radgraph": {
            "status": radgraph_status,
            "version": RADGRAPH_VERSION,
            "model_type": RADGRAPH_MODEL_TYPE,
            "score_component": "RG_ER",
            "error": radgraph_error,
        },
        "budget_summaries": summaries,
        "pairwise_comparisons": pairwise_rows,
        "energy_utility_pareto": pareto,
        "interpretation_limit": "F1-RadGraph measures candidate-reference factual/structural agreement, not image-grounded diagnostic accuracy or radiologist-adjudicated clinical safety. Token caps can truncate reports; near-cap output frequency is reported explicitly.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WP3_MEDGEMMA_TOKEN_BUDGET_100CASE_COMPLETE")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
