from __future__ import annotations

import csv
import json

import analyze_wp3_three_model_100case_impl as impl


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
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report_path = impl.OUT / "analysis_report.md"
    text = report_path.read_text(encoding="utf-8")
    text = text.replace(
        "Operational comparisons use ten matched 10-case blocks. Gross direct NVIDIA GPU board energy is primary and idle-adjusted net energy is secondary. Runtime medians were recomputed from case-level elapsed times to remove the prior even-n median bug.",
        "Operational comparisons use ten matched 10-case blocks. Gross direct NVIDIA GPU board energy is primary and idle-adjusted net energy is secondary. Runtime comparison is omitted because the Qwen case-level export does not retain elapsed times and the available historical block medians have a known even-n median implementation bug."
    )
    report_path.write_text(text, encoding="utf-8")


impl.corrected_runtime_blocks = compatibility_runtime_blocks

if __name__ == "__main__":
    impl.main()
    quarantine_runtime_outputs()
