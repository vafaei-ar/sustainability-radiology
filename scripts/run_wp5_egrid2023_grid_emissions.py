from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "wp5" / "egrid2023"
CACHE = ROOT / "data" / "wp5" / "egrid2023"
XLSX = CACHE / "egrid2023_data_rev2.xlsx"
STATE_CSV = OUT / "state_co2e_intensity.csv"
SUBREGION_CSV = OUT / "subregion_co2e_intensity.csv"
SUMMARY_JSON = OUT / "summary.json"
REPORT_MD = OUT / "validation_report.md"

SOURCE_URL = "https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx"
DATA_YEAR = 2023
REVISION = "Revision 2"
REVISION_RELEASE_DATE = "2025-06-12"
LB_PER_MWH_TO_KG_PER_KWH = 0.45359237 / 1000.0

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


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
        "unit": "eGRID stages",
        "updated_at_epoch": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if XLSX.is_file() and XLSX.stat().st_size > 100_000:
        return
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "sustainability-radiology/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, XLSX.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    if XLSX.stat().st_size <= 100_000:
        raise RuntimeError("Downloaded eGRID workbook is unexpectedly small")


def col_index(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref)
    if not m:
        raise ValueError(cell_ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + ord(ch) - 64
    return n - 1


def shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    out: list[str] = []
    for si in root.findall(f"{{{NS_MAIN}}}si"):
        parts = [t.text or "" for t in si.iter(f"{{{NS_MAIN}}}t")]
        out.append("".join(parts))
    return out


def workbook_sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rels = {
        r.attrib["Id"]: r.attrib["Target"]
        for r in rel.findall(f"{{{NS_REL_PKG}}}Relationship")
    }
    out = []
    for s in wb.find(f"{{{NS_MAIN}}}sheets") or []:
        name = s.attrib["name"]
        rid = s.attrib[f"{{{NS_REL_DOC}}}id"]
        target = rels[rid]
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = "xl/" + target.lstrip("./")
        out.append((name, path))
    return out


def sheet_rows(z: zipfile.ZipFile, path: str, sst: list[str]) -> list[list[object]]:
    root = ET.fromstring(z.read(path))
    rows: list[list[object]] = []
    sheet_data = root.find(f"{{{NS_MAIN}}}sheetData")
    if sheet_data is None:
        return rows
    for r in sheet_data.findall(f"{{{NS_MAIN}}}row"):
        vals: dict[int, object] = {}
        max_idx = -1
        for c in r.findall(f"{{{NS_MAIN}}}c"):
            idx = col_index(c.attrib.get("r", "A1"))
            max_idx = max(max_idx, idx)
            typ = c.attrib.get("t")
            if typ == "inlineStr":
                node = c.find(f"{{{NS_MAIN}}}is")
                val = "" if node is None else "".join((t.text or "") for t in node.iter(f"{{{NS_MAIN}}}t"))
            else:
                v = c.find(f"{{{NS_MAIN}}}v")
                raw = "" if v is None or v.text is None else v.text
                if typ == "s" and raw:
                    val = sst[int(raw)]
                elif typ == "str":
                    val = raw
                else:
                    try:
                        val = float(raw) if raw != "" else ""
                    except ValueError:
                        val = raw
            vals[idx] = val
        if max_idx >= 0:
            rows.append([vals.get(i, "") for i in range(max_idx + 1)])
    return rows


def norm_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def find_table(rows: list[list[object]], code_candidates: set[str]) -> tuple[list[str], list[list[object]]]:
    for i, row in enumerate(rows):
        normalized = [norm_code(x) for x in row]
        if any(code in normalized for code in code_candidates):
            headers = [str(x).strip() for x in row]
            return headers, rows[i + 1 :]
    raise RuntimeError(f"Could not locate header row containing any of {sorted(code_candidates)}")


def header_index(headers: list[str], candidates: set[str]) -> int:
    normalized = [norm_code(x) for x in headers]
    for candidate in candidates:
        if candidate in normalized:
            return normalized.index(candidate)
    raise RuntimeError(f"Missing required header: {sorted(candidates)}")


def extract_level(rows: list[list[object]], level: str) -> list[dict[str, object]]:
    if level == "state":
        code_headers = {"STC2ERTA", "STCO2ERTA"}
        id_headers = {"PSTATABB", "STABBR", "STATEABBREVIATION"}
        name_headers = {"PSTATABBN", "STNAME", "STATENAME"}
    else:
        code_headers = {"SRC2ERTA", "SRCO2ERTA"}
        id_headers = {"SUBRGN", "SUBREGION", "EGRIDSUBREGIONACRONYM"}
        name_headers = {"SRNAME", "EGRIDSUBREGIONNAME"}

    headers, data = find_table(rows, code_headers)
    idx_rate = header_index(headers, code_headers)
    idx_id = header_index(headers, id_headers)
    try:
        idx_name = header_index(headers, name_headers)
    except RuntimeError:
        idx_name = -1

    out: list[dict[str, object]] = []
    for row in data:
        if idx_id >= len(row):
            continue
        identifier = str(row[idx_id]).strip()
        if not identifier or identifier.lower() in {"nan", "none"}:
            continue
        if idx_rate >= len(row) or row[idx_rate] == "":
            continue
        try:
            rate = float(row[idx_rate])
        except (TypeError, ValueError):
            continue
        name = str(row[idx_name]).strip() if idx_name >= 0 and idx_name < len(row) else ""
        out.append(
            {
                "id": identifier,
                "name": name,
                "co2e_lb_per_mwh": rate,
                "co2e_kg_per_kwh": rate * LB_PER_MWH_TO_KG_PER_KWH,
            }
        )
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, object]], id_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([id_name, "name", "co2e_lb_per_mwh", "co2e_kg_per_kwh", "data_year", "revision"])
        for r in rows:
            w.writerow([
                r["id"],
                r["name"],
                f"{float(r['co2e_lb_per_mwh']):.6f}",
                f"{float(r['co2e_kg_per_kwh']):.9f}",
                DATA_YEAR,
                REVISION,
            ])


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    vals = sorted(float(r["co2e_kg_per_kwh"]) for r in rows)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    min_row = min(rows, key=lambda r: float(r["co2e_kg_per_kwh"]))
    max_row = max(rows, key=lambda r: float(r["co2e_kg_per_kwh"]))
    return {
        "n": n,
        "min_kg_per_kwh": float(min_row["co2e_kg_per_kwh"]),
        "min_id": min_row["id"],
        "median_kg_per_kwh": med,
        "max_kg_per_kwh": float(max_row["co2e_kg_per_kwh"]),
        "max_id": max_row["id"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress(0, 4, "Downloading EPA eGRID2023 Revision 2 workbook")
    download()
    digest = sha256(XLSX)
    progress(1, 4, "Downloaded and hashed EPA workbook")

    with zipfile.ZipFile(XLSX) as z:
        sst = shared_strings(z)
        sheets = workbook_sheets(z)
        state_rows = None
        subregion_rows = None
        state_sheet = None
        subregion_sheet = None
        for name, path in sheets:
            rows = sheet_rows(z, path, sst)
            flat = {norm_code(v) for row in rows[:20] for v in row}
            if state_rows is None and ({"STC2ERTA", "STCO2ERTA"} & flat):
                state_rows = rows
                state_sheet = name
            if subregion_rows is None and ({"SRC2ERTA", "SRCO2ERTA"} & flat):
                subregion_rows = rows
                subregion_sheet = name
        if state_rows is None or subregion_rows is None:
            raise RuntimeError(f"Could not identify required eGRID sheets; sheets={[n for n, _ in sheets]}")
        states = extract_level(state_rows, "state")
        subregions = extract_level(subregion_rows, "subregion")

    progress(2, 4, "Parsed state and eGRID-subregion CO2e output rates")

    state_ids = {str(r["id"]) for r in states}
    subregion_ids = {str(r["id"]) for r in subregions}
    if len(states) != 52:
        raise RuntimeError(f"Expected 52 state-level records from eGRID2023, found {len(states)}")
    if len(subregions) != 27:
        raise RuntimeError(f"Expected 27 eGRID-subregion records, found {len(subregions)}")
    if len(state_ids) != len(states) or len(subregion_ids) != len(subregions):
        raise RuntimeError("Duplicate state or subregion identifiers detected")
    if not all(float(r["co2e_kg_per_kwh"]) >= 0 for r in states + subregions):
        raise RuntimeError("Negative CO2e output emission rate detected")
    progress(3, 4, "Validated record counts, uniqueness, and units")

    write_csv(STATE_CSV, states, "state")
    write_csv(SUBREGION_CSV, subregions, "egrid_subregion")
    state_summary = summarize(states)
    subregion_summary = summarize(subregions)
    summary = {
        "status": "WP5_EGRID2023_REV2_OK",
        "source": {
            "publisher": "US EPA",
            "dataset": "eGRID2023",
            "revision": REVISION,
            "revision_release_date": REVISION_RELEASE_DATE,
            "url": SOURCE_URL,
            "sha256": digest,
            "local_cache": str(XLSX.relative_to(ROOT)),
            "state_sheet": state_sheet,
            "subregion_sheet": subregion_sheet,
        },
        "conversion": {
            "source_unit": "lb CO2e / MWh",
            "target_unit": "kg CO2e / kWh",
            "factor": LB_PER_MWH_TO_KG_PER_KWH,
        },
        "state": state_summary,
        "subregion": subregion_summary,
        "method": "Annual total output emission rates from eGRID2023 Revision 2. These are generation-source output rates and do not include transmission/distribution losses.",
        "intended_use": "Primary annual geographic electricity carbon-intensity layer for operational AI inference modeling.",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    REPORT_MD.write_text(
        "# WP5 eGRID2023 Revision 2 validation\n\n"
        f"- Source: {SOURCE_URL}\n"
        f"- Revision release date: {REVISION_RELEASE_DATE}\n"
        f"- Workbook SHA256: `{digest}`\n"
        f"- Parsed state sheet: `{state_sheet}`\n"
        f"- Parsed subregion sheet: `{subregion_sheet}`\n"
        f"- State records: {len(states)}\n"
        f"- eGRID subregion records: {len(subregions)}\n"
        f"- Unit conversion: 1 lb/MWh = {LB_PER_MWH_TO_KG_PER_KWH:.11f} kg/kWh\n\n"
        "The exported carbon-intensity endpoint is annual total CO2-equivalent output emission rate. "
        "It is suitable for annual operational electricity modeling. EPA eGRID output rates are calculated at generation sources and do not include transmission and distribution losses; any later loss adjustment must therefore be explicit and separate.\n\n"
        f"State range: {state_summary['min_kg_per_kwh']:.6f} to {state_summary['max_kg_per_kwh']:.6f} kg CO2e/kWh.\n\n"
        f"Subregion range: {subregion_summary['min_kg_per_kwh']:.6f} to {subregion_summary['max_kg_per_kwh']:.6f} kg CO2e/kWh.\n",
        encoding="utf-8",
    )
    progress(4, 4, "eGRID state/subregion carbon-intensity layer complete")
    print("WP5_EGRID2023_REV2_OK")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
