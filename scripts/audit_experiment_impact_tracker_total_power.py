#!/usr/bin/env python3
"""Read-only audit of experiment-impact-tracker `total_power` semantics.

The goal is to establish whether the historical benchmark field copied from
`experiment_impact_tracker.data_interface.total_power` represents energy and,
if so, its unit. The script inspects the installed package when available and
then searches a small allow-listed set of local roots for package source. It
writes only package metadata, source paths, and relevant source-code excerpts.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
from pathlib import Path

OUT = Path("results/wp1/experiment_impact_tracker_total_power_audit.txt")
ROOTS = [
    Path("/home/asadr/works"),
    Path("/home/asadr/.local/lib"),
]
TERMS = ("total_power", "kwh", "kilowatt", "joule", "energy", "power")
MAX_SOURCE_FILES = 30
MAX_LINES_PER_FILE = 120


def relevant_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [f"READ_ERROR\t{path}\t{type(exc).__name__}: {exc}"]
    rows: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        if any(term in low for term in TERMS):
            start = max(1, i - 2)
            end = min(len(lines), i + 2)
            for j in range(start, end + 1):
                rows.append(f"SOURCE\t{path}\tL{j}\t{lines[j-1]}")
            rows.append("---")
            if len(rows) >= MAX_LINES_PER_FILE:
                rows.append(f"TRUNCATED\t{path}\t{MAX_LINES_PER_FILE}")
                break
    return rows


def add_installed_package(out: list[str]) -> None:
    for dist_name in ("experiment-impact-tracker", "experiment_impact_tracker"):
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        out.append(f"DIST_VERSION\t{dist_name}\t{version}")
        try:
            dist = importlib.metadata.distribution(dist_name)
            out.append(f"DIST_LOCATION\t{dist_name}\t{dist.locate_file('')}")
        except Exception as exc:
            out.append(f"DIST_ERROR\t{dist_name}\t{type(exc).__name__}: {exc}")

    spec = importlib.util.find_spec("experiment_impact_tracker")
    if spec is None:
        out.append("IMPORT_SPEC\texperiment_impact_tracker\tNOT_FOUND")
        return
    out.append(f"IMPORT_SPEC\texperiment_impact_tracker\t{spec.origin}")
    locations = list(spec.submodule_search_locations or [])
    for loc in locations:
        base = Path(loc)
        for candidate in sorted(base.rglob("*.py")):
            if candidate.name in {"data_interface.py", "tracker.py", "impact_tracker.py"}:
                out.append(f"INSTALLED_SOURCE_FILE\t{candidate}")
                out.extend(relevant_lines(candidate))


def search_local_sources(out: list[str]) -> None:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in ROOTS:
        if not root.exists():
            out.append(f"SEARCH_ROOT_MISSING\t{root}")
            continue
        out.append(f"SEARCH_ROOT\t{root}")
        for pattern in ("**/experiment_impact_tracker/**/*.py", "**/experiment-impact-tracker/**/*.py"):
            try:
                for path in root.glob(pattern):
                    if path.is_file() and path not in seen:
                        seen.add(path)
                        candidates.append(path)
                        if len(candidates) >= MAX_SOURCE_FILES:
                            break
            except Exception as exc:
                out.append(f"SEARCH_ERROR\t{root}\t{type(exc).__name__}: {exc}")
            if len(candidates) >= MAX_SOURCE_FILES:
                break
        if len(candidates) >= MAX_SOURCE_FILES:
            break

    for path in candidates:
        if path.name == "data_interface.py" or "tracker" in path.name.lower() or "impact" in path.name.lower():
            out.append(f"LOCAL_SOURCE_FILE\t{path}")
            out.extend(relevant_lines(path))
    out.append(f"LOCAL_SOURCE_FILES_FOUND\t{len(candidates)}")


def main() -> None:
    out: list[str] = []
    out.append("AUDIT\texperiment-impact-tracker total_power semantics")
    add_installed_package(out)
    search_local_sources(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"EIT_TOTAL_POWER_AUDIT_OK lines={len(out)}")


if __name__ == "__main__":
    main()
