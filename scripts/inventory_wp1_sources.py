#!/usr/bin/env python3
"""Inventory exact WP1 benchmark source filenames under the user's works tree.

This diagnostic is intentionally read-only outside the active project. It records
only file paths, sizes, and SHA256 digests for three non-clinical benchmark CSV
filenames needed by WP1.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

NAMES = {"A100.csv", "RTX5000.csv", "RTX_6000_Ada.csv"}
SEARCH_ROOTS = [Path("/home/asadr/works")]
OUT = Path("results/wp1/source_inventory.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


lines: list[str] = []
for root in SEARCH_ROOTS:
    lines.append(f"SEARCH_ROOT\t{root}")
    if not root.is_dir():
        lines.append(f"MISSING_ROOT\t{root}")
        continue
    for path in sorted(root.rglob("*.csv")):
        if path.name not in NAMES:
            continue
        try:
            digest = sha256(path)
            size = path.stat().st_size
            lines.append(f"FOUND\t{path}\t{size}\t{digest}")
        except OSError as exc:
            lines.append(f"ERROR\t{path}\t{type(exc).__name__}: {exc}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"WP1_SOURCE_INVENTORY rows={sum(line.startswith('FOUND') for line in lines)}")
