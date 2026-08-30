#!/usr/bin/env python3
"""Read-only audit of experiment-impact-tracker `total_power` semantics.

This pass extracts the exact function blocks and unit-bearing expressions needed
to determine whether `data_interface.total_power` is cumulative energy and its
unit. It reads only allow-listed local source trees and writes source excerpts,
metadata, and a machine-generated conclusion flag. No user data are read.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from pathlib import Path

OUT = Path("results/wp1/experiment_impact_tracker_total_power_audit.txt")
ROOTS = [Path("/home/asadr/works/sustainability"), Path("/home/asadr/.local/lib")]
KEY_NAMES = {
    "data_interface.py",
    "data_info_and_router.py",
    "compute_tracker.py",
    "common.py",
    "nvidia.py",
    "intel.py",
    "rapl.py",
    "setup.py",
}
TERMS = (
    "def gather_additional_info",
    "total_power",
    "estimated_carbon_impact_kg",
    "exp_len_hours",
    "pue",
    "kwh",
    "kilowatt",
    "joule",
    "watt",
    "energy",
    "power_draw",
    "power_usage",
    "co2",
    "carbon",
    "/ 1000",
    "/1000",
    "3600",
)


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def emit_window(out: list[str], path: Path, center: int, before: int = 10, after: int = 18) -> None:
    lines = read(path)
    a = max(1, center - before)
    b = min(len(lines), center + after)
    out.append(f"WINDOW\t{path}\tL{a}-L{b}")
    for i in range(a, b + 1):
        out.append(f"SOURCE\t{path}\tL{i}\t{lines[i-1]}")
    out.append("---")


def emit_function(out: list[str], path: Path, function_name: str) -> bool:
    lines = read(path)
    start = None
    indent = None
    rx = re.compile(rf"^(\s*)def\s+{re.escape(function_name)}\s*\(")
    for i, line in enumerate(lines):
        m = rx.match(line)
        if m:
            start = i
            indent = len(m.group(1))
            break
    if start is None:
        return False
    end = min(len(lines), start + 120)
    for j in range(start + 1, min(len(lines), start + 120)):
        stripped = lines[j].strip()
        if not stripped:
            continue
        leading = len(lines[j]) - len(lines[j].lstrip())
        if leading <= indent and (lines[j].lstrip().startswith("def ") or lines[j].lstrip().startswith("class ")):
            end = j
            break
    out.append(f"FUNCTION\t{path}\t{function_name}\tL{start+1}-L{end}")
    for i in range(start, end):
        out.append(f"SOURCE\t{path}\tL{i+1}\t{lines[i]}")
    out.append("---")
    return True


def candidate_files() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in ROOTS:
        if not root.exists():
            continue
        for pattern in ("**/experiment_impact_tracker/**/*.py", "**/experiment-impact-tracker/**/*.py"):
            for p in root.glob(pattern):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    if p.name in KEY_NAMES or "data_interface" in p.name or "tracker" in p.name:
                        found.append(p)
    return sorted(found)


def main() -> None:
    out: list[str] = ["AUDIT\texperiment-impact-tracker total_power semantics v2"]

    for dist_name in ("experiment-impact-tracker", "experiment_impact_tracker"):
        try:
            out.append(f"DIST_VERSION\t{dist_name}\t{importlib.metadata.version(dist_name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    spec = importlib.util.find_spec("experiment_impact_tracker")
    out.append(f"IMPORT_SPEC\texperiment_impact_tracker\t{spec.origin if spec else 'NOT_FOUND'}")

    files = candidate_files()
    out.append(f"CANDIDATE_FILES\t{len(files)}")
    for p in files:
        out.append(f"LOCAL_SOURCE_FILE\t{p}")

    # Exact function blocks most likely to define total_power.
    for p in files:
        if p.name == "data_interface.py":
            for fn in ("gather_additional_info", "get_total_power", "get_carbon_impact"):
                emit_function(out, p, fn)

    # Context windows for every unit-bearing or total_power expression.
    seen_windows: set[tuple[Path, int]] = set()
    for p in files:
        try:
            lines = read(p)
        except Exception as exc:
            out.append(f"READ_ERROR\t{p}\t{type(exc).__name__}: {exc}")
            continue
        for i, line in enumerate(lines, start=1):
            low = line.lower()
            if any(term in low for term in TERMS):
                key = (p, i // 12)
                if key in seen_windows:
                    continue
                seen_windows.add(key)
                emit_window(out, p, i)

    joined = "\n".join(out).lower()
    evidence_energy = any(x in joined for x in ("kwh", "kilowatt-hour", "kilowatt hour"))
    evidence_total = "total_power" in joined
    evidence_conversion = any(x in joined for x in ("/ 1000", "/1000", "3600"))
    out.append(f"EVIDENCE_TOTAL_POWER\t{evidence_total}")
    out.append(f"EVIDENCE_KWH_TOKEN\t{evidence_energy}")
    out.append(f"EVIDENCE_UNIT_CONVERSION_TOKEN\t{evidence_conversion}")
    out.append("CONCLUSION\tMANUAL_SOURCE_REVIEW_REQUIRED")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"EIT_TOTAL_POWER_AUDIT_V2_OK files={len(files)} lines={len(out)}")


if __name__ == "__main__":
    main()
