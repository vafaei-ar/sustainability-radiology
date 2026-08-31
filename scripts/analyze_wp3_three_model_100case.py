from __future__ import annotations

import csv
import json
import os
import pathlib
import subprocess

import analyze_wp3_three_model_100case_impl as impl


RADGRAPH_F1_MODEL_TYPE = "radgraph-xl"


def compatibility_runtime_blocks(cases):
    """Provide legacy block medians only so the shared implementation can finish.

    These values are removed from all saved publication outputs below because
    Qwen case-level exports do not retain elapsed times and the historical
    even-n block medians are known to be biased upward.
    """
    blocks = impl.load_blocks()
    aliases = ("median_case_seconds", "median_case_elapsed_seconds")
    out = {}
    for model in impl.MODELS:
        out[model] = {}
        for block in range(1, 11):
            row = blocks[model][block]
            value = None
            for key in aliases:
                if row.get(key) not in (None, ""):
                    value = float(row[key])
                    break
            if value is None:
                raise RuntimeError(f"No legacy block runtime field for {model} block {block}")
            out[model][block] = value
    return out


def fixed_try_radgraph(pairs):
    """Run official F1-RadGraph and return the case-level RG_ER component.

    The official F1RadGraph(reward_level='all') API returns three per-case
    vectors: RG_E, RG_ER, and RG_bar_ER. We use RG_ER, the component the
    upstream README notes has been widely reported in radiology report
    generation studies. The scorer model type follows the upstream F1 example
    (`radgraph-xl`), rather than the entity-extraction-only modern model used
    in the earlier exploratory attempt.
    """
    if not impl.WP3_PY.is_file():
        return "unavailable", None, ".venv-wp3 Python not found"

    impl.METRIC_TARGET.mkdir(exist_ok=True)
    try:
        install = [
            str(impl.WP3_PY), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-input",
            "--target", str(impl.METRIC_TARGET), "--upgrade", "--no-deps",
            f"radgraph=={impl.RADGRAPH_VERSION}",
            "appdirs", "dotmap", "jsonpickle", "h5py", "nltk",
        ]
        install_run = subprocess.run(
            install,
            cwd=impl.ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        if install_run.returncode != 0:
            tail = (install_run.stdout or "")[-4000:]
            return "failed", None, f"RadGraph package install failed (exit {install_run.returncode}): {tail}"

        payload_path = impl.OUT / ".radgraph_pairs.json"
        payload_path.write_text(json.dumps(pairs) + "\n", encoding="utf-8")
        result_path = impl.OUT / ".radgraph_result.json"
        code = r'''
import json, pathlib, sys
from radgraph import F1RadGraph
pairs = json.loads(pathlib.Path(sys.argv[1]).read_text())
refs = [x["reference"] for x in pairs]
hyps = [x["candidate"] for x in pairs]
scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=-1)
mean_reward, reward_list, _, _ = scorer(hyps=hyps, refs=refs)
rg_e, rg_er, rg_bar_er = reward_list
payload = {
    "mean_reward": [float(x) for x in mean_reward],
    "rg_e": [float(x) for x in rg_e],
    "rg_er": [float(x) for x in rg_er],
    "rg_bar_er": [float(x) for x in rg_bar_er],
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload))
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(impl.METRIC_TARGET) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["CUDA_VISIBLE_DEVICES"] = ""
        metric_run = subprocess.run(
            [str(impl.WP3_PY), "-c", code, str(payload_path), str(result_path)],
            cwd=impl.ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            env=env,
        )
        if metric_run.returncode != 0:
            tail = (metric_run.stdout or "")[-6000:]
            return "failed", None, f"F1-RadGraph execution failed (exit {metric_run.returncode}): {tail}"

        obj = json.loads(result_path.read_text(encoding="utf-8"))
        scores = obj.get("rg_er")
        if not isinstance(scores, list) or len(scores) != len(pairs):
            return "raw_output_unparsed", None, (
                f"F1-RadGraph completed but RG_ER length was "
                f"{len(scores) if isinstance(scores, list) else 'invalid'}; expected {len(pairs)}"
            )
        return "ok", [float(x) for x in scores], None
    except Exception as exc:
        return "failed", None, f"{type(exc).__name__}: {exc}"


def quarantine_runtime_outputs() -> None:
    op_path = impl.OUT / "operational_pairwise.csv"
    with op_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["metric"] != "corrected_median_case_seconds"]
    adjusted = impl.holm([
        (f"{r['metric']}:{r['numerator_model']}/{r['denominator_model']}", float(r["exact_sign_p_two_sided"]))
        for r in rows
    ])
    for r in rows:
        key = f"{r['metric']}:{r['numerator_model']}/{r['denominator_model']}"
        r["holm_p_across_all_operational_pairwise_tests"] = adjusted[key]
    impl.write_csv(op_path, rows)

    summary_path = impl.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["operational_pairwise"] = rows
    summary.setdefault("interpretation", {})["runtime_note"] = (
        "Runtime comparison omitted. The Qwen 100-case case-level export does not retain per-case elapsed times, "
        "and the available historical 10-case block medians use a known even-n upper-middle implementation rather "
        "than the standard median. Energy endpoints are unaffected."
    )
    radgraph = summary.get("automated_clinical_fidelity", {}).get("radgraph", {})
    if radgraph:
        radgraph["score_component"] = "RG_ER"
        radgraph["f1_model_type"] = RADGRAPH_F1_MODEL_TYPE
        radgraph["score_component_note"] = (
            "RG_ER is the entity-plus-relation F1 component returned by F1RadGraph(reward_level='all'); "
            "the upstream RadGraph README notes that RG_ER has been widely reported in radiology report-generation studies."
        )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_path = impl.OUT / "analysis_report.md"
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "Operational comparisons use ten matched 10-case blocks. Gross direct NVIDIA GPU board energy is primary and idle-adjusted net energy is secondary. Runtime medians were recomputed from case-level elapsed times to remove the prior even-n median bug.",
        "Operational comparisons use ten matched 10-case blocks. Gross direct NVIDIA GPU board energy is primary and idle-adjusted net energy is secondary. Runtime comparison is omitted because the Qwen case-level export does not retain elapsed times and the available historical block medians have a known even-n median implementation bug."
    )
    text = text.replace(
        "The preferred automated endpoint is F1-RadGraph from the official Stanford-AIMI RadGraph package (pinned to version 0.1.18) when the runtime completed successfully.",
        "The preferred automated endpoint is the RG_ER component of F1-RadGraph from the official Stanford-AIMI RadGraph package (pinned to version 0.1.18, scorer model type radgraph-xl) when the runtime completed successfully."
    )
    report_path.write_text(text, encoding="utf-8")


impl.corrected_runtime_blocks = compatibility_runtime_blocks
impl.try_radgraph = fixed_try_radgraph

if __name__ == "__main__":
    impl.main()
    quarantine_runtime_outputs()
