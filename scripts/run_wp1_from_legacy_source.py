#!/usr/bin/env python3
"""Stage the three historical WP1 benchmark CSVs from the known legacy project tree, verify them, then run the audit."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
import runpy

SOURCES = {
    "A100": (Path("/home/asadr/works/sus/models_gpus/A100.csv"), Path("data/legacy/narrow/A100.csv"), "8f9491ac35f4ca54154e031912a258d5c406a2a6db6d739a82760afcf402a390"),
    "RTX5000": (Path("/home/asadr/works/sus/models_gpus/RTX5000.csv"), Path("data/legacy/narrow/RTX5000.csv"), "4b5e9235479b7d0e44a97429a69a919c649b00f420b45aeb4a337c65764491b5"),
    "RTX_6000_Ada": (Path("/home/asadr/works/sus/models_gpus/RTX_6000_Ada.csv"), Path("data/legacy/narrow/RTX_6000_Ada.csv"), "ab3670401d28e696588fe07f4f4d06990644d8ab8d4e35f1bd83b30027b8640b"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


for key, (src, dst, expected) in SOURCES.items():
    if not src.is_file():
        raise FileNotFoundError(f"Missing legacy source for {key}: {src}")
    observed_source = sha256(src)
    if observed_source != expected:
        raise RuntimeError(f"Legacy source checksum mismatch for {src}: expected {expected}, observed {observed_source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    observed_copy = sha256(dst)
    if observed_copy != expected:
        raise RuntimeError(f"Staged copy checksum mismatch for {dst}: expected {expected}, observed {observed_copy}")

ns = runpy.run_path("scripts/run_wp1_benchmark_audit.py")
ns["EXPECTED_SHA256"].update({key: spec[2] for key, spec in SOURCES.items()})
ns["main"]()
