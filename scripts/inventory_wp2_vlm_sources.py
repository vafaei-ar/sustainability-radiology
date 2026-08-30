#!/usr/bin/env python3
"""Read-only inventory of historical VLM benchmark sources for WP2."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path("/home/asadr/works")
OUT = Path("results/wp2/source_inventory.txt")
TARGETS = {
    "RTX_6000_Ada_multimodel.csv",
    "InternVL2.py",
    "florence-2.py",
    "PaliGemma.py",
    "paligemma.py",
    "moondream.py",
    "moondream2.py",
}
KEYWORDS = ("multimodal", "internvl", "florence", "paligemma", "moondream")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = [f"SEARCH_ROOT\t{ROOT}"]
    seen: set[Path] = set()
    if ROOT.exists():
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            low = str(path).lower()
            if name in TARGETS or any(k in low for k in KEYWORDS):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    rows.append(f"FOUND\t{path}\t{path.stat().st_size}\t{sha256(path)}")
                except Exception as exc:
                    rows.append(f"ERROR\t{path}\t{type(exc).__name__}: {exc}")
    rows.append(f"FOUND_COUNT\t{len(seen)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"WP2_SOURCE_INVENTORY rows={len(seen)}")


if __name__ == "__main__":
    main()
