from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
MED = ROOT / "results" / "wp3" / "medgemma_100case"
OUT = ROOT / "results" / "wp3" / "three_model_100case_analysis"
BOOT = 20000
SEED = 20260831
REVIEW_N = 50
PAIR_SPECS = [("MedGemma", "Qwen"), ("MedGemma", "InternVL3"), ("Qwen", "InternVL3")]
MODELS = ("MedGemma", "Qwen", "InternVL3")
RADGRAPH_VERSION = "0.1.18"
METRIC_TARGET = ROOT / ".wp3-metrics-packages"
WP3_PY = ROOT / ".venv-wp3" / "bin" / "python"

FINDINGS = {
    "cardiomegaly": [r"cardiomegal", r"enlarged heart", r"cardiac enlargement", r"heart (?:is |appears )?enlarged"],
    "pneumothorax": [r"pneumothorax", r"pleural air"],
    "pleural_effusion": [r"pleural effusion", r"costophrenic .*blunt", r"blunting of .*costophrenic"],
    "consolidation": [r"consolidat", r"airspace disease", r"pneumonia", r"focal infiltrate"],
    "pulmonary_edema": [r"pulmonary edema", r"interstitial edema", r"vascular congestion", r"venous engorgement"],
    "atelectasis": [r"atelecta"],
    "lung_opacity": [r"lung opacity", r"pulmonary opacity", r"airspace opacity", r"opacit(?:y|ies)"],
    "lung_lesion": [r"lung mass", r"pulmonary mass", r"lung nodule", r"pulmonary nodule", r"lung lesion", r"pulmonary lesion"],
    "fracture": [r"fracture"],
    "support_device": [r"catheter", r"pacemaker", r"icd", r"endotracheal tube", r"chest tube", r"central line", r"prosthetic valve"],
    "hiatal_hernia": [r"hiatal hernia"],
}
NEGATION = re.compile(r"\b(no|without|absent|negative for|free of|not seen|no evidence of|clear of|resolved)\b", re.I)
UNCERTAIN = re.compile(r"\b(possible|possibly|probable|probably|may|might|suggest|suggesting|cannot exclude|could represent|likely)\b", re.I)


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
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
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
    used = {key: 0 for key in shuffled}
    base, rem = divmod(REVIEW_N, len(shuffled))
    for sidx, key in enumerate(sorted(shuffled)):
        target = base + (1 if sidx < rem else 0)
        take = min(target, len(shuffled[key]))
        selected.extend(shuffled[key][:take]); used[key] = take
    ordered_keys = sorted(shuffled)
    while len(selected) < REVIEW_N:
        added = False
        for key in ordered_keys:
            pos = used[key]
            if pos < len(shuffled[key]):
                selected.append(shuffled[key][pos]); used[key] += 1; added = True
                if len(selected) == REVIEW_N:
                    break
        if not added:
            break
    if len(selected) != REVIEW_N or len(set(selected)) != REVIEW_N:
        raise RuntimeError(f"Clinical-review selection failed: {len(selected)} rows, {len(set(selected))} unique")
    return sorted(selected)


def reference_text(row: dict[str, str]) -> str:
    return " ".join(x for x in (row.get("reference_findings", ""), row.get("reference_impression", "")) if x).strip()


def sentence_chunks(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if x.strip()]


def finding_state(text: str, patterns: list[str]) -> str:
    uncertain = False
    for sent in sentence_chunks(text.lower()):
        for pat in patterns:
            m = re.search(pat, sent, re.I)
            if not m:
                continue
            before = sent[max(0, m.start() - 65):m.start()]
            around = sent[max(0, m.start() - 65):min(len(sent), m.end() + 65)]
            if NEGATION.search(before) or re.search(r"\bno\b[^.;]{0,45}" + pat, sent, re.I):
                return "negative"
            if UNCERTAIN.search(around):
                uncertain = True
            else:
                return "positive"
    return "uncertain" if uncertain else "not_mentioned"


def proxy_case(reference: str, candidate: str) -> dict[str, float | int | str]:
    rs = {k: finding_state(reference, v) for k, v in FINDINGS.items()}
    cs = {k: finding_state(candidate, v) for k, v in FINDINGS.items()}
    rp = {k for k, v in rs.items() if v == "positive"}
    cp = {k for k, v in cs.items() if v == "positive"}
    union = {k for k in FINDINGS if rs[k] != "not_mentioned" or cs[k] != "not_mentioned"}
    precision = len(rp & cp) / len(cp) if cp else (1.0 if not rp else 0.0)
    recall = len(rp & cp) / len(rp) if rp else (1.0 if not cp else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    omitted = sorted(rp - cp)
    hallucinated = sorted(cp - rp)
    return {
        "finding_state_agreement": sum(rs[k] == cs[k] for k in FINDINGS) / len(FINDINGS),
        "mentioned_state_agreement": sum(rs[k] == cs[k] for k in union) / len(union) if union else 1.0,
        "positive_finding_f1": f1,
        "omission_proxy": int(bool(omitted)),
        "hallucination_proxy": int(bool(hallucinated)),
        "omitted_findings": ";".join(omitted),
        "hallucinated_findings": ";".join(hallucinated),
    }


def corrected_runtime_blocks(cases: dict[str, dict[int, dict[str, str]]]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for model in MODELS:
        by_block: dict[int, list[float]] = defaultdict(list)
        for row in cases[model].values():
            by_block[int(row["block"])].append(float(row["elapsed_seconds"]))
        out[model] = {}
        for block in range(1, 11):
            vals = by_block[block]
            if len(vals) != 10:
                raise RuntimeError(f"Expected 10 case runtimes for {model} block {block}, found {len(vals)}")
            out[model][block] = statistics.median(vals)
    return out


def try_radgraph(pairs: list[dict[str, str]]) -> tuple[str, list[float] | None, str | None]:
    if not WP3_PY.is_file():
        return "unavailable", None, ".venv-wp3 Python not found"
    METRIC_TARGET.mkdir(exist_ok=True)
    try:
        install = [str(WP3_PY), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--target", str(METRIC_TARGET), "--upgrade", "--no-deps", f"radgraph=={RADGRAPH_VERSION}", "appdirs", "dotmap", "jsonpickle", "h5py", "nltk"]
        subprocess.run(install, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        payload_path = OUT / ".radgraph_pairs.json"
        payload_path.write_text(json.dumps(pairs) + "\n", encoding="utf-8")
        result_path = OUT / ".radgraph_result.json"
        code = r'''
import json, pathlib, sys
from radgraph import F1RadGraph
pairs=json.loads(pathlib.Path(sys.argv[1]).read_text())
refs=[x["reference"] for x in pairs]; hyps=[x["candidate"] for x in pairs]
scorer=F1RadGraph(reward_level="all", model_type="modern-radgraph-xl")
res=scorer(hyps=hyps, refs=refs)
pathlib.Path(sys.argv[2]).write_text(json.dumps(res, default=float))
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(METRIC_TARGET) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([str(WP3_PY), "-c", code, str(payload_path), str(result_path)], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900, env=env)
        obj = json.loads(result_path.read_text(encoding="utf-8"))
        candidates = []
        def walk(x):
            if isinstance(x, list) and len(x) == len(pairs) and all(isinstance(v, (int, float)) for v in x):
                candidates.append([float(v) for v in x])
            elif isinstance(x, (list, tuple)):
                for y in x: walk(y)
            elif isinstance(x, dict):
                for y in x.values(): walk(y)
        walk(obj)
        if not candidates:
            return "raw_output_unparsed", None, "RadGraph completed but no 300-value reward vector was identified"
        return "ok", candidates[-1], None
    except Exception as exc:
        return "failed", None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    blocks = load_blocks()
    manifest = index_rows(read_csv(TWO / "case_manifest_100.csv"), 100, "manifest")
    cases = {
        "Qwen": index_rows(read_csv(TWO / "qwen_case_review_100.csv"), 100, "Qwen"),
        "InternVL3": index_rows(read_csv(TWO / "internvl_case_results_100.csv"), 100, "InternVL3"),
        "MedGemma": index_rows(read_csv(MED / "case_results_100.csv"), 100, "MedGemma"),
    }
    corrected_runtime = corrected_runtime_blocks(cases)

    operational = []
    op_ps = []
    for metric in ["gross_wh_per_case", "net_wh_per_case", "corrected_median_case_seconds"]:
        for a, b in PAIR_SPECS:
            if metric == "corrected_median_case_seconds":
                av = [corrected_runtime[a][j] for j in range(1, 11)]
                bv = [corrected_runtime[b][j] for j in range(1, 11)]
            else:
                av = [canonical_block_value(blocks[a][j], metric) for j in range(1, 11)]
                bv = [canonical_block_value(blocks[b][j], metric) for j in range(1, 11)]
            est, lo, hi = paired_bootstrap_ratio(av, bv, rng)
            diffs = [x - y for x, y in zip(av, bv)]
            p = exact_sign_p(diffs)
            key = f"{metric}:{a}/{b}"; op_ps.append((key, p))
            operational.append({"metric": metric, "numerator_model": a, "denominator_model": b, "ratio_of_paired_block_means": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "numerator_higher_blocks": sum(x > 0 for x in diffs), "numerator_lower_blocks": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_blocks": 10, "holm_p_across_all_operational_pairwise_tests": None})
    adj = holm(op_ps)
    for r in operational:
        r["holm_p_across_all_operational_pairwise_tests"] = adj[f"{r['metric']}:{r['numerator_model']}/{r['denominator_model']}"]
    progress(1, 6, "Operational paired analysis complete")

    utility = []
    util_ps = []
    for metric in ["unigram_f1", "rouge_l_f1"]:
        for a, b in PAIR_SPECS:
            av = [float(cases[a][j][metric]) for j in range(1, 101)]
            bv = [float(cases[b][j][metric]) for j in range(1, 101)]
            est, lo, hi = paired_bootstrap_diff(av, bv, rng); diffs = [x - y for x, y in zip(av, bv)]; p = exact_sign_p(diffs)
            key = f"{metric}:{a}-{b}"; util_ps.append((key, p))
            utility.append({"metric": metric, "model_a": a, "model_b": b, "mean_a": statistics.mean(av), "mean_b": statistics.mean(bv), "mean_paired_difference_a_minus_b": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "a_higher_cases": sum(x > 0 for x in diffs), "b_higher_cases": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_cases": 100, "holm_p_within_metric_family": None, "endpoint_class": "lexical_screening"})
    progress(2, 6, "Lexical utility analysis complete")

    proxy_by_model: dict[str, list[dict]] = {m: [] for m in MODELS}
    radgraph_pairs = []
    for idx in range(1, 101):
        ref = reference_text(cases["Qwen"][idx]) or reference_text(cases["InternVL3"][idx]) or reference_text(cases["MedGemma"][idx])
        if not ref:
            raise RuntimeError(f"No reference text for case {idx}")
        for model in MODELS:
            cand = cases[model][idx].get("model_output", "").strip()
            if not cand:
                raise RuntimeError(f"Empty output for {model} case {idx}")
            prox = proxy_case(ref, cand); prox["case_index"] = idx
            proxy_by_model[model].append(prox)
            radgraph_pairs.append({"case_index": idx, "model": model, "reference": ref, "candidate": cand})
    for metric in ["finding_state_agreement", "mentioned_state_agreement", "positive_finding_f1"]:
        for a, b in PAIR_SPECS:
            av = [float(x[metric]) for x in proxy_by_model[a]]; bv = [float(x[metric]) for x in proxy_by_model[b]]
            est, lo, hi = paired_bootstrap_diff(av, bv, rng); diffs = [x - y for x, y in zip(av, bv)]; p = exact_sign_p(diffs)
            key = f"{metric}:{a}-{b}"; util_ps.append((key, p))
            utility.append({"metric": metric, "model_a": a, "model_b": b, "mean_a": statistics.mean(av), "mean_b": statistics.mean(bv), "mean_paired_difference_a_minus_b": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "a_higher_cases": sum(x > 0 for x in diffs), "b_higher_cases": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_cases": 100, "holm_p_within_metric_family": None, "endpoint_class": "deterministic_reference_proxy"})
    progress(3, 6, "Deterministic clinical-fidelity proxy complete")

    radgraph_status, radgraph_scores, radgraph_error = try_radgraph(radgraph_pairs)
    radgraph_model_means = {}
    if radgraph_scores is not None:
        rg_by_model = {m: [] for m in MODELS}
        for pair, score in zip(radgraph_pairs, radgraph_scores):
            rg_by_model[pair["model"]].append(float(score))
        radgraph_model_means = {m: statistics.mean(v) for m, v in rg_by_model.items()}
        for a, b in PAIR_SPECS:
            av, bv = rg_by_model[a], rg_by_model[b]
            est, lo, hi = paired_bootstrap_diff(av, bv, rng); diffs = [x - y for x, y in zip(av, bv)]; p = exact_sign_p(diffs)
            key = f"radgraph_f1:{a}-{b}"; util_ps.append((key, p))
            utility.append({"metric": "radgraph_f1", "model_a": a, "model_b": b, "mean_a": statistics.mean(av), "mean_b": statistics.mean(bv), "mean_paired_difference_a_minus_b": est, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "exact_sign_p_two_sided": p, "a_higher_cases": sum(x > 0 for x in diffs), "b_higher_cases": sum(x < 0 for x in diffs), "ties": sum(x == 0 for x in diffs), "paired_cases": 100, "holm_p_within_metric_family": None, "endpoint_class": "validated_automated_radiology_metric"})
    uadj = holm(util_ps)
    for r in utility:
        r["holm_p_within_metric_family"] = uadj[f"{r['metric']}:{r['model_a']}-{r['model_b']}"]
    progress(4, 6, f"RadGraph stage complete: {radgraph_status}")

    review_ids = select_review_ids(manifest)
    blind = random.Random(SEED + 17)
    review_rows, key_rows = [], []
    for idx in review_ids:
        mapping = ["Qwen", "InternVL3", "MedGemma"]; blind.shuffle(mapping)
        q, i, m = cases["Qwen"][idx], cases["InternVL3"][idx], cases["MedGemma"][idx]
        findings = q.get("reference_findings", "") or i.get("reference_findings", "") or m.get("reference_findings", "")
        impression = q.get("reference_impression", "") or i.get("reference_impression", "") or m.get("reference_impression", "")
        row = {"review_case_id": f"R{idx:03d}", "case_index": idx, "normal_metadata_stratum": manifest[idx]["normal_metadata_stratum"], "reference_length_quartile": manifest[idx]["reference_length_quartile"], "reference_findings": findings, "reference_impression": impression, "output_A": cases[mapping[0]][idx]["model_output"], "output_B": cases[mapping[1]][idx]["model_output"], "output_C": cases[mapping[2]][idx]["model_output"]}
        for field in ["major_abnormality_correct", "clinically_important_omission", "clinically_important_hallucination", "laterality_or_location_error", "critical_safety_error", "overall_acceptable"]:
            for letter in "ABC": row[f"{field}_{letter}"] = ""
        row["reviewer_notes"] = ""; review_rows.append(row); key_rows.append({"review_case_id": f"R{idx:03d}", "A": mapping[0], "B": mapping[1], "C": mapping[2]})
    write_csv(OUT / "clinical_review_blinded_50.csv", review_rows); write_csv(OUT / "clinical_review_unblinding_key.csv", key_rows)

    qa_errors = []
    if len(review_rows) != REVIEW_N or len({r["case_index"] for r in review_rows}) != REVIEW_N: qa_errors.append("review case count/uniqueness failed")
    if any(not r["output_A"].strip() or not r["output_B"].strip() or not r["output_C"].strip() for r in review_rows): qa_errors.append("empty blinded output")
    if any(any(r[f"{field}_{letter}"] != "" for field in ["major_abnormality_correct", "clinically_important_omission", "clinically_important_hallucination", "laterality_or_location_error", "critical_safety_error", "overall_acceptable"] for letter in "ABC") for r in review_rows): qa_errors.append("adjudication field unexpectedly populated")
    valid_perm = all(set((k["A"], k["B"], k["C"])) == set(MODELS) for k in key_rows)
    if not valid_perm: qa_errors.append("invalid unblinding permutation")
    packet_qa = {"status": "pass" if not qa_errors else "fail", "errors": qa_errors, "review_cases": len(review_rows), "unique_case_indices": len({r["case_index"] for r in review_rows}), "all_outputs_nonempty": not any("empty" in e for e in qa_errors), "all_adjudication_fields_blank": not any("adjudication" in e for e in qa_errors), "unblinding_key_valid": valid_perm, "normal_metadata_counts": dict(Counter(r["normal_metadata_stratum"] for r in review_rows)), "reference_length_quartile_counts": dict(Counter(r["reference_length_quartile"] for r in review_rows)), "joint_stratum_counts": dict(Counter(f"{r['normal_metadata_stratum']}|{r['reference_length_quartile']}" for r in review_rows)), "blinded_column_model_counts": {letter: dict(Counter(k[letter] for k in key_rows)) for letter in "ABC"}}
    if qa_errors: raise RuntimeError("Clinical-review packet QA failed: " + "; ".join(qa_errors))
    progress(5, 6, "Blinded review packet regenerated and QA passed")

    write_csv(OUT / "operational_pairwise.csv", operational)
    write_csv(OUT / "utility_pairwise.csv", utility)

    proxy_summary = {}
    for model in MODELS:
        rows = proxy_by_model[model]
        proxy_summary[model] = {
            "finding_state_agreement_mean": statistics.mean(float(r["finding_state_agreement"]) for r in rows),
            "mentioned_state_agreement_mean": statistics.mean(float(r["mentioned_state_agreement"]) for r in rows),
            "positive_finding_f1_mean": statistics.mean(float(r["positive_finding_f1"]) for r in rows),
            "omission_proxy_rate": statistics.mean(int(r["omission_proxy"]) for r in rows),
            "hallucination_proxy_rate": statistics.mean(int(r["hallucination_proxy"]) for r in rows),
        }
        if model in radgraph_model_means: proxy_summary[model]["radgraph_f1_mean"] = radgraph_model_means[model]

    gross_energy = {m: statistics.mean(canonical_block_value(blocks[m][j], "gross_wh_per_case") for j in range(1, 11)) for m in MODELS}
    pareto_metric = "radgraph_f1_mean" if radgraph_model_means else "mentioned_state_agreement_mean"
    pareto = []
    for model in MODELS:
        util = float(proxy_summary[model][pareto_metric])
        pareto.append({"model": model, "gross_gpu_board_wh_per_case_mean": gross_energy[model], "utility_metric": pareto_metric, "utility_value": util, "wh_per_utility_unit": gross_energy[model] / util if util > 0 else None})
    for row in pareto:
        dominators = [o["model"] for o in pareto if o is not row and o["gross_gpu_board_wh_per_case_mean"] <= row["gross_gpu_board_wh_per_case_mean"] and o["utility_value"] >= row["utility_value"] and (o["gross_gpu_board_wh_per_case_mean"] < row["gross_gpu_board_wh_per_case_mean"] or o["utility_value"] > row["utility_value"])]
        row["pareto_dominated"] = bool(dominators); row["dominated_by"] = dominators

    summary = {
        "status": "WP3_THREE_MODEL_100CASE_AUTOMATED_FIDELITY_OK",
        "seed": SEED,
        "bootstrap_iterations": BOOT,
        "models": ["MedGemma-4B", "Qwen2.5-VL-7B-Instruct", "InternVL3-8B"],
        "operational_pairwise": operational,
        "utility_pairwise": utility,
        "automated_clinical_fidelity": {
            "scope": "Automated reference-based evaluation on all 100 frozen Open-I cases for each of three models (300 model-reference pairs).",
            "radgraph": {"status": radgraph_status, "version": RADGRAPH_VERSION, "model_type": "modern-radgraph-xl", "error": radgraph_error, "model_means": radgraph_model_means},
            "deterministic_proxy": {"finding_categories": list(FINDINGS), "model_summary": proxy_summary},
            "interpretation_limit": "These are automated report-fidelity metrics, not radiologist adjudication and not diagnostic accuracy estimates.",
            "reference_limit": "Open-I report text is the reference; deidentification placeholders, report incompleteness, and the exactly-one-image sampling restriction can affect apparent agreement.",
        },
        "energy_utility_pareto": pareto,
        "clinical_review_packet": {"n_cases": REVIEW_N, "selection": "Deterministic capacity-aware balanced sample across MeSH normal/non-normal metadata status and reference-length quartile.", "blinding": "Model identities randomized independently by case across output columns A/B/C. Unblinding key stored separately.", "adjudication_status": "Not performed. Packet retained for optional future expert review.", "qa": packet_qa},
        "interpretation": {"energy_scope": "Direct NVIDIA GPU board operational energy; gross primary and idle-adjusted net secondary.", "operational_inference_unit": "Ten matched 10-case blocks.", "runtime_note": "Runtime medians were recomputed from case-level elapsed_seconds using the standard median, correcting the prior even-n upper-middle implementation bug in historical block summaries.", "utility_inference_unit": "One hundred paired cases.", "utility_hierarchy": "F1-RadGraph is preferred when available; deterministic finding-state metrics are exploratory proxies; unigram F1 and ROUGE-L remain secondary lexical screening endpoints.", "clinical_limit": "No human radiologist adjudication was performed."},
        "method_sources": {"RadGraph": "Stanford-AIMI/radgraph, official open-source F1-RadGraph implementation, pinned package 0.1.18.", "GREEN": "Stanford-AIMI/GREEN, methodological context only; not executed in this task.", "CheXbert": "stanfordmlgroup/CheXbert, methodological context only; not executed because its released checkpoint has separate licensing/access requirements."},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# WP3 three-model 100-case analysis and automated clinical fidelity",
        "",
        "Operational comparisons use ten matched 10-case blocks. Gross direct NVIDIA GPU board energy is primary and idle-adjusted net energy is secondary. Runtime medians were recomputed from case-level elapsed times to remove the prior even-n median bug.",
        "",
        "Automated clinical fidelity was evaluated on all 300 model-reference pairs. The preferred automated endpoint is F1-RadGraph from the official Stanford-AIMI RadGraph package (pinned to version 0.1.18) when the runtime completed successfully. A transparent deterministic finding-state proxy was also computed for cardiomegaly, pneumothorax, pleural effusion, consolidation/pneumonia, pulmonary edema, atelectasis, lung opacity, lung lesion/nodule/mass, fracture, support devices, and hiatal hernia.",
        "",
        f"RadGraph status: **{radgraph_status}**.",
        "",
        "These automated metrics estimate reference-report fidelity. They are not radiologist adjudication, diagnostic accuracy, sensitivity, or specificity. The Open-I text report is treated as the reference and can itself be incomplete or contain deidentification placeholders.",
        "",
        "The 50-case blinded clinical-review packet was regenerated and passed structural QA but remains intentionally unadjudicated. It is retained for optional future expert validation rather than being filled by a non-clinician.",
        "",
        "Energy-utility Pareto results use mean gross GPU board Wh/case and the preferred automated utility endpoint. Any efficiency claim should remain conditional on automated report fidelity and should not be presented as clinical superiority.",
        "",
        "## Method sources",
        "- RadGraph: Stanford-AIMI/radgraph, official F1-RadGraph implementation, package version 0.1.18.",
        "- GREEN: Stanford-AIMI/GREEN, cited as methodological context only; not executed here.",
        "- CheXbert: stanfordmlgroup/CheXbert, cited as methodological context only; not executed here because the released checkpoint has separate licensing/access requirements.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(6, 6, "Automated clinical-fidelity publication bundle complete")
    print("WP3_THREE_MODEL_100CASE_AUTOMATED_FIDELITY_OK")
    print(json.dumps({"radgraph_status": radgraph_status, "proxy_summary": proxy_summary, "pareto": pareto, "packet_qa": packet_qa}, sort_keys=True))


if __name__ == "__main__":
    main()
