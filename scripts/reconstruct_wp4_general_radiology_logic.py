from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp4" / "general_radiology_logic"
CONTROL_DIR = pathlib.Path("/home/asadr/datasets/trinetx/66350692f55db9228fba3206_20240514_224202103_Control")
SEARCH_ROOTS = [
    pathlib.Path("/home/asadr/works"),
    pathlib.Path("/home/asadr"),
]
TARGET_NAME = "sus_radio.ipynb"
TARGET_DISEASES = {
    "Breast cancer": "BC",
    "COPD": "COPD",
    "Chronic kidney disease": "CKD",
    "Ischemic heart disease": "IHD",
    "Colon/Rectal cancer": "CRC",
}
RELEVANT_TABLE_NAMES = {
    "diagnosis.csv",
    "procedure.csv",
    "patient.csv",
    "demographics.csv",
    "standardized_terminology.csv",
    "cohort_details.csv",
}
KEYWORDS = [
    "breast cancer", "copd", "chronic kidney", "ischemic heart", "ischaemic heart",
    "colon", "rectal", "crc", "bc", "ckd", "ihd", "trinetx", "control",
    "icd", "cpt", "hcpcs", "procedure", "radiolog", "imaging", "ct", "mri",
    "mra", "cta", "pet", "x-ray", "xray", "ultrasound", "mammograph", "gbd", "ihme",
    "n_patients", "n_procedures", "denominator", "utilization", "prevalence", "incidence",
]
PRUNE_DIRS = {
    ".git", ".cache", ".conda", ".venv", "venv", "node_modules", "datasets",
    ".wp3-models", ".wp3-data", "site-packages", "__pycache__",
}


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
        "unit": "WP4 reconstruction stages",
        "updated_at_epoch": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_notebooks() -> list[pathlib.Path]:
    found: set[pathlib.Path] = set()
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
            if TARGET_NAME in filenames:
                found.add((pathlib.Path(dirpath) / TARGET_NAME).resolve())
    return sorted(found)


def cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(x) for x in source)
    return ""


def relevant_cell(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in KEYWORDS)


def extract_notebook_logic(path: pathlib.Path) -> dict[str, object]:
    nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    selected: list[dict[str, object]] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if not isinstance(cell, dict):
            continue
        text = cell_text(cell)
        if not text or not relevant_cell(text):
            continue
        # Source only. Notebook outputs are deliberately ignored because they may contain restricted values.
        selected.append({
            "cell_index": idx,
            "cell_type": str(cell.get("cell_type", "")),
            "source": text[:20000],
        })
    joined = "\n\n".join(str(x["source"]) for x in selected)
    disease_hits = {
        name: bool(re.search(re.escape(name), joined, flags=re.I) or re.search(rf"\b{re.escape(abbr)}\b", joined))
        for name, abbr in TARGET_DISEASES.items()
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "cell_count": len(nb.get("cells", [])),
        "relevant_cell_count": len(selected),
        "disease_mentions": disease_hits,
        "relevant_cells": selected,
    }


def read_csv_header(path: pathlib.Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",\t|;").delimiter
            except Exception:
                delim = ","
            return [str(x).strip() for x in next(csv.reader(f, delimiter=delim), [])]
    except Exception:
        return []


def relevant_table_schemas() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not CONTROL_DIR.is_dir():
        return rows
    for path in sorted(CONTROL_DIR.rglob("*.csv")):
        rel = path.relative_to(CONTROL_DIR).as_posix()
        name = path.name.lower()
        role_hit = name in RELEVANT_TABLE_NAMES or any(
            key in rel.lower() for key in ["diagnos", "procedure", "patient", "demograph", "terminology", "cohort_details"]
        )
        if not role_hit:
            continue
        rows.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "schema_fields": read_csv_header(path),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress(0, 4, "Searching exact notebook filename on authorized workstation roots")
    notebooks = find_notebooks()
    progress(1, 4, f"Found {len(notebooks)} sus_radio.ipynb candidate(s)")

    parsed: list[dict[str, object]] = []
    for path in notebooks:
        try:
            parsed.append(extract_notebook_logic(path))
        except Exception as exc:
            parsed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    progress(2, 4, "Parsed notebook source cells without reading outputs")

    schemas = relevant_table_schemas()
    progress(3, 4, "Read headers for relevant TriNetX Control tables")

    logic_ready = any(
        isinstance(item.get("relevant_cells"), list) and len(item.get("relevant_cells", [])) > 0
        for item in parsed
    )
    summary = {
        "status": "WP4_GENERAL_RADIOLOGY_LOGIC_RECONSTRUCTION_OK",
        "purpose": "Recover the original general-radiology disease/procedure/scaling logic before estimating utilization from the broad TriNetX Control export.",
        "target_diseases": TARGET_DISEASES,
        "searched_roots": [str(p) for p in SEARCH_ROOTS],
        "notebook_filename": TARGET_NAME,
        "notebooks_found": len(notebooks),
        "notebooks": parsed,
        "control_directory": str(CONTROL_DIR),
        "control_directory_exists": CONTROL_DIR.is_dir(),
        "relevant_table_schemas": schemas,
        "logic_ready_for_aggregate_extraction": logic_ready,
        "privacy": "Only notebook source text and table headers are exported. Notebook outputs and TriNetX row values are not read into artifacts.",
        "next_step_if_ready": "Implement disease-by-modality aggregate extraction using the recovered definitions and export only n_patients, n_procedures, utilization, and prespecified strata.",
    }
    (OUT / "logic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (OUT / "relevant_table_schemas.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "schema_fields"])
        w.writeheader()
        for row in schemas:
            w.writerow({**row, "schema_fields": json.dumps(row["schema_fields"])})

    md: list[str] = [
        "# WP4 general-radiology notebook logic reconstruction",
        "",
        f"Notebook candidates found: {len(notebooks)}.",
        f"Logic ready for aggregate extraction: {logic_ready}.",
        "",
        "The target diseases are Breast cancer (BC), COPD, Chronic kidney disease (CKD), Ischemic heart disease (IHD), and Colon/Rectal cancer (CRC).",
        "",
        "Notebook outputs were intentionally ignored. Only source cells matching disease, imaging/procedure, TriNetX, or GBD terms are reproduced below.",
    ]
    for item in parsed:
        md.extend(["", f"## {item.get('path')}"])
        if "error" in item:
            md.append(f"Error: {item['error']}")
            continue
        md.append(f"SHA256: `{item.get('sha256')}`")
        for cell in item.get("relevant_cells", []):
            md.extend([
                "",
                f"### Cell {cell['cell_index']} ({cell['cell_type']})",
                "```python" if cell["cell_type"] == "code" else "```text",
                str(cell["source"]).rstrip(),
                "```",
            ])
    (OUT / "notebook_logic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    progress(4, 4, "WP4 general-radiology logic reconstruction complete")
    print("WP4_GENERAL_RADIOLOGY_LOGIC_RECONSTRUCTION_OK")
    print(json.dumps({
        "notebooks_found": len(notebooks),
        "logic_ready_for_aggregate_extraction": logic_ready,
        "relevant_table_schema_count": len(schemas),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
