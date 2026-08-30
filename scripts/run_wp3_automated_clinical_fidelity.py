from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
import venv

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO = ROOT / "results" / "wp3" / "openi_100case_single_image_two_model"
MED = ROOT / "results" / "wp3" / "medgemma_100case"
OUT = ROOT / "results" / "wp3" / "automated_clinical_fidelity"
ENV = ROOT / ".venv-wp3-metrics"
RADGRAPH_VERSION = "0.1.18"
MODELS = ("MedGemma", "Qwen", "InternVL3")

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
    "support_device": [r"catheter", r"pacemaker", r"icd", r"endotracheal tube", r"chest tube", r"central line", r"prosthetic valve", r"support device"],
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
    tmp.write_text(json.dumps({"schema_version": 1, "current": current, "total": total, "fraction": current / total, "phase": phase, "unit": "evaluation stages", "updated_at_epoch": time.time()}), encoding="utf-8")
    os.replace(tmp, p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing required input: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index100(rows: list[dict[str, str]], label: str) -> dict[int, dict[str, str]]:
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 rows for {label}, found {len(rows)}")
    out = {int(r["case_index"]): r for r in rows}
    if set(out) != set(range(1, 101)):
        raise RuntimeError(f"Non-contiguous case_index for {label}")
    return out


def ensure_metric_env() -> pathlib.Path:
    py = ENV / "bin" / "python"
    marker = ENV / ".radgraph-0.1.18-ready"
    if not py.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(ENV)
    if not marker.exists():
        cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "radgraph==" + RADGRAPH_VERSION, "appdirs", "dotmap", "jsonpickle", "h5py", "nltk"]
        subprocess.run(cmd, cwd=ROOT, check=True)
        marker.write_text("radgraph==0.1.18\n", encoding="utf-8")
    return py


def sentence_chunks(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if x.strip()]


def finding_state(text: str, patterns: list[str]) -> str:
    found_uncertain = False
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
                found_uncertain = True
            else:
                return "positive"
    return "uncertain" if found_uncertain else "not_mentioned"


def proxy_case(reference: str, candidate: str) -> dict:
    ref_states = {k: finding_state(reference, v) for k, v in FINDINGS.items()}
    cand_states = {k: finding_state(candidate, v) for k, v in FINDINGS.items()}
    ref_positive = {k for k, v in ref_states.items() if v == "positive"}
    cand_positive = {k for k, v in cand_states.items() if v == "positive"}
    important_omissions = sorted(ref_positive - cand_positive)
    hallucinations = sorted(cand_positive - ref_positive)
    exact = sum(ref_states[k] == cand_states[k] for k in FINDINGS)
    mentioned_ref = {k for k, v in ref_states.items() if v != "not_mentioned"}
    mentioned_cand = {k for k, v in cand_states.items() if v != "not_mentioned"}
    union = mentioned_ref | mentioned_cand
    state_agreement = exact / len(FINDINGS)
    mentioned_state_agreement = (sum(ref_states[k] == cand_states[k] for k in union) / len(union)) if union else 1.0
    precision = len(ref_positive & cand_positive) / len(cand_positive) if cand_positive else (1.0 if not ref_positive else 0.0)
    recall = len(ref_positive & cand_positive) / len(ref_positive) if ref_positive else (1.0 if not cand_positive else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "finding_state_agreement": state_agreement,
        "mentioned_state_agreement": mentioned_state_agreement,
        "positive_finding_precision": precision,
        "positive_finding_recall": recall,
        "positive_finding_f1": f1,
        "important_omission_proxy": int(bool(important_omissions)),
        "hallucination_proxy": int(bool(hallucinations)),
        "omitted_findings": ";".join(important_omissions),
        "hallucinated_findings": ";".join(hallucinations),
        "reference_states_json": json.dumps(ref_states, sort_keys=True),
        "candidate_states_json": json.dumps(cand_states, sort_keys=True),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def mean(xs):
    vals = [float(x) for x in xs]
    return statistics.mean(vals) if vals else None


def bootstrap_mean_diff(a: list[float], b: list[float], seed: int = 20260831, boot: int = 10000) -> tuple[float, float, float]:
    import random
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = statistics.mean(diffs)
    reps = []
    n = len(diffs)
    for _ in range(boot):
        reps.append(statistics.mean(diffs[rng.randrange(n)] for _ in range(n)))
    reps.sort()
    lo = reps[int(0.025 * (boot - 1))]
    hi = reps[int(0.975 * (boot - 1))]
    return obs, lo, hi


def load_data():
    q = index100(read_csv(TWO / "qwen_case_review_100.csv"), "Qwen")
    i = index100(read_csv(TWO / "internvl_case_results_100.csv"), "InternVL3")
    m = index100(read_csv(MED / "case_results_100.csv"), "MedGemma")
    return {"Qwen": q, "InternVL3": i, "MedGemma": m}


def build_pairs(cases):
    pairs = []
    for idx in range(1, 101):
        q = cases["Qwen"][idx]
        reference = " ".join(x for x in (q.get("reference_findings", ""), q.get("reference_impression", "")) if x).strip()
        if not reference:
            # The model case tables are matched; fall back to another model only if needed.
            for model in ("InternVL3", "MedGemma"):
                r = cases[model][idx]
                reference = " ".join(x for x in (r.get("reference_findings", ""), r.get("reference_impression", "")) if x).strip()
                if reference:
                    break
        if not reference:
            raise RuntimeError(f"No reference text for case {idx}")
        for model in MODELS:
            candidate = cases[model][idx].get("model_output", "").strip()
            if not candidate:
                raise RuntimeError(f"Empty model output for {model} case {idx}")
            pairs.append({"case_index": idx, "model": model, "reference": reference, "candidate": candidate})
    return pairs


def run_radgraph(py: pathlib.Path, input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    code = r'''
import csv, json, pathlib, sys
from radgraph import F1RadGraph
inp = pathlib.Path(sys.argv[1]); outp = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(inp.open(encoding="utf-8")))
refs = [r["reference"] for r in rows]
hyps = [r["candidate"] for r in rows]
scorer = F1RadGraph(reward_level="all", model_type="modern-radgraph-xl")
res = scorer(hyps=hyps, refs=refs)
# API versions return either (mean_reward, reward_list) or a dictionary-like object.
if isinstance(res, tuple) and len(res) == 2:
    mean_reward, reward_list = res
else:
    mean_reward, reward_list = None, res
payload = {"mean_reward": mean_reward, "reward_list": reward_list, "radgraph_version": "0.1.18", "model_type": "modern-radgraph-xl"}
outp.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
'''
    subprocess.run([str(py), "-c", code, str(input_path), str(output_path)], cwd=ROOT, check=True)


def extract_radgraph_scores(payload: dict, n: int) -> list[float] | None:
    obj = payload.get("reward_list")
    # F1RadGraph all-level commonly returns three reward arrays; use the full graph reward when identifiable.
    if isinstance(obj, list) and len(obj) == n and all(isinstance(x, (int, float)) for x in obj):
        return [float(x) for x in obj]
    if isinstance(obj, (list, tuple)) and len(obj) in (3, 4):
        candidates = [x for x in obj if isinstance(x, list) and len(x) == n and all(isinstance(v, (int, float)) for v in x)]
        if candidates:
            return [float(x) for x in candidates[-1]]
    if isinstance(obj, dict):
        for key in ("reward", "radgraph", "full", "f1", "scores"):
            x = obj.get(key)
            if isinstance(x, list) and len(x) == n:
                return [float(v) for v in x]
    return None


def energy_means() -> dict[str, float]:
    two = read_csv(TWO / "block_summary.csv")
    med = read_csv(MED / "block_summary.csv")
    q = [float(r["gross_wh_per_case"]) for r in two if r.get("model") == "Qwen2.5-VL-7B-Instruct"]
    i = [float(r["gross_wh_per_case"]) for r in two if r.get("model") == "InternVL3-8B"]
    m = [float(r.get("gross_gpu_energy_wh_per_case") or r.get("gross_wh_per_case")) for r in med]
    if not (len(q) == len(i) == len(m) == 10):
        raise RuntimeError("Expected ten gross-energy blocks per model")
    return {"Qwen": statistics.mean(q), "InternVL3": statistics.mean(i), "MedGemma": statistics.mean(m)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_data()
    pairs = build_pairs(cases)
    write_csv(OUT / "report_pairs_300.csv", pairs)
    progress(1, 5, "Frozen 300 model-reference pairs prepared")

    proxy_rows = []
    for p in pairs:
        row = {"case_index": p["case_index"], "model": p["model"]}
        row.update(proxy_case(p["reference"], p["candidate"]))
        proxy_rows.append(row)
    write_csv(OUT / "proxy_case_metrics.csv", proxy_rows)
    progress(2, 5, "Deterministic finding-state proxy complete")

    radgraph_status = "not_run"
    radgraph_error = None
    radgraph_scores = None
    try:
        py = ensure_metric_env()
        run_radgraph(py, OUT / "report_pairs_300.csv", OUT / "radgraph_raw.json")
        payload = json.loads((OUT / "radgraph_raw.json").read_text(encoding="utf-8"))
        radgraph_scores = extract_radgraph_scores(payload, len(pairs))
        radgraph_status = "ok" if radgraph_scores is not None else "raw_output_unparsed"
    except Exception as exc:
        radgraph_status = "failed"
        radgraph_error = f"{type(exc).__name__}: {exc}"
        (OUT / "radgraph_raw.json").write_text(json.dumps({"status": "failed", "error": radgraph_error}, indent=2) + "\n", encoding="utf-8")
    progress(3, 5, f"RadGraph stage: {radgraph_status}")

    if radgraph_scores is not None:
        for row, score in zip(proxy_rows, radgraph_scores):
            row["radgraph_f1"] = score
    else:
        for row in proxy_rows:
            row["radgraph_f1"] = ""
    write_csv(OUT / "case_metrics.csv", proxy_rows)

    summary_rows = []
    for model in MODELS:
        rows = [r for r in proxy_rows if r["model"] == model]
        summary_rows.append({
            "model": model,
            "n_cases": len(rows),
            "finding_state_agreement_mean": mean(r["finding_state_agreement"] for r in rows),
            "mentioned_state_agreement_mean": mean(r["mentioned_state_agreement"] for r in rows),
            "positive_finding_f1_mean": mean(r["positive_finding_f1"] for r in rows),
            "omission_proxy_rate": mean(r["important_omission_proxy"] for r in rows),
            "hallucination_proxy_rate": mean(r["hallucination_proxy"] for r in rows),
            "radgraph_f1_mean": mean(r["radgraph_f1"] for r in rows if r["radgraph_f1"] != "") if radgraph_scores is not None else "",
        })
    write_csv(OUT / "model_summary.csv", summary_rows)

    paired = []
    for metric in ("finding_state_agreement", "mentioned_state_agreement", "positive_finding_f1") + (("radgraph_f1",) if radgraph_scores is not None else tuple()):
        for a, b in (("MedGemma", "Qwen"), ("MedGemma", "InternVL3"), ("Qwen", "InternVL3")):
            av = [float(r[metric]) for r in proxy_rows if r["model"] == a]
            bv = [float(r[metric]) for r in proxy_rows if r["model"] == b]
            d, lo, hi = bootstrap_mean_diff(av, bv)
            paired.append({"metric": metric, "model_a": a, "model_b": b, "mean_difference_a_minus_b": d, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "paired_cases": 100})
    write_csv(OUT / "paired_utility.csv", paired)

    energy = energy_means()
    pareto = []
    for s in summary_rows:
        model = s["model"]
        utility_name = "radgraph_f1_mean" if radgraph_scores is not None else "mentioned_state_agreement_mean"
        utility = float(s[utility_name])
        pareto.append({"model": model, "gross_gpu_board_wh_per_case_mean": energy[model], "utility_metric": utility_name, "utility_value": utility, "wh_per_utility_unit": energy[model] / utility if utility > 0 else ""})
    # A model is dominated if another has lower-or-equal energy and higher-or-equal utility, with at least one strict.
    for row in pareto:
        dominated_by = []
        for other in pareto:
            if other is row:
                continue
            if other["gross_gpu_board_wh_per_case_mean"] <= row["gross_gpu_board_wh_per_case_mean"] and other["utility_value"] >= row["utility_value"] and (other["gross_gpu_board_wh_per_case_mean"] < row["gross_gpu_board_wh_per_case_mean"] or other["utility_value"] > row["utility_value"]):
                dominated_by.append(other["model"])
        row["pareto_dominated"] = int(bool(dominated_by))
        row["dominated_by"] = ";".join(dominated_by)
    write_csv(OUT / "energy_utility_pareto.csv", pareto)
    progress(4, 5, "Model summaries, paired utility, and Pareto table complete")

    runtime = {}
    if ENV.exists():
        try:
            py = ENV / "bin" / "python"
            runtime["python"] = subprocess.check_output([str(py), "--version"], text=True, stderr=subprocess.STDOUT).strip()
            runtime["radgraph"] = RADGRAPH_VERSION
        except Exception:
            pass
    report = {
        "status": "WP3_AUTOMATED_CLINICAL_FIDELITY_OK",
        "cases": 100,
        "model_reference_pairs": 300,
        "models": list(MODELS),
        "radgraph_status": radgraph_status,
        "radgraph_error": radgraph_error,
        "radgraph_version_requested": RADGRAPH_VERSION,
        "radgraph_model_type": "modern-radgraph-xl",
        "proxy_scope": "Deterministic reference-grounded finding-state agreement across eleven predefined chest-radiograph finding categories with simple negation/uncertainty handling.",
        "proxy_limit": "The rule-based finding-state proxy is not a validated radiologist adjudication instrument and must not be described as clinical accuracy.",
        "radgraph_limit": "RadGraph is an automated report factuality/entity-relation metric; it is not a substitute for expert radiologist review.",
        "reference_limit": "Open-I report text is treated as the reference. Report incompleteness, deidentification placeholders, and single-image selection can affect apparent agreement.",
        "energy_scope": "Gross direct NVIDIA GPU board operational energy per case from the frozen BF16 benchmark; model loading and warmup excluded.",
        "utility_primary": "F1-RadGraph when successfully parsed; otherwise the deterministic mentioned-finding state agreement is used only as an exploratory proxy.",
        "model_summary": summary_rows,
        "pareto": pareto,
        "runtime": runtime,
        "citations": [
            {"name": "RadGraph", "source": "Stanford-AIMI/radgraph", "version": RADGRAPH_VERSION, "note": "Official open-source F1-RadGraph implementation."},
            {"name": "GREEN", "source": "Stanford-AIMI/GREEN", "note": "Methodological context only; not executed in this task."},
            {"name": "CheXbert", "source": "stanfordmlgroup/CheXbert", "note": "Methodological context only; not executed because the released checkpoint has separate licensing/access requirements."},
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# WP3 automated reference-based clinical fidelity",
        "",
        "This analysis evaluates all 300 model-reference report pairs from the frozen 100-case single-image Open-I benchmark.",
        "",
        "The preferred automated utility endpoint is F1-RadGraph from the official Stanford-AIMI RadGraph package (pinned to 0.1.18) when the runtime and model download complete successfully. A transparent deterministic finding-state proxy is computed for every pair regardless of RadGraph availability. The proxy covers cardiomegaly, pneumothorax, pleural effusion, consolidation/pneumonia, pulmonary edema, atelectasis, lung opacity, lung lesion/nodule/mass, fracture, support devices, and hiatal hernia.",
        "",
        "These are automated reference-based metrics. They are not radiologist adjudication and must not be labeled clinical accuracy, sensitivity, specificity, or diagnostic performance.",
        "",
        "Energy-utility Pareto results use gross direct NVIDIA GPU board Wh/case from the frozen BF16 benchmark. Gross energy is primary because idle-adjusted net energy depends more strongly on baseline procedure.",
        "",
        f"RadGraph status: **{radgraph_status}**.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(5, 5, "Automated clinical-fidelity bundle complete")
    print("WP3_AUTOMATED_CLINICAL_FIDELITY_OK")
    print(json.dumps({"radgraph_status": radgraph_status, "models": summary_rows, "pareto": pareto}, sort_keys=True))


if __name__ == "__main__":
    main()
