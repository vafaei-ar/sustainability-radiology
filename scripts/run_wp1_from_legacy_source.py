#!/usr/bin/env python3
"""Stage the three historical WP1 benchmark CSVs from a verified legacy source, then run the audit.

The source archive tree shows two historical locations for these frozen summary
CSVs. This wrapper checks only those exact allow-listed locations, verifies the
expected SHA256 digest, copies the matching file into the active project, and
runs the project-local WP1 analysis.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
import runpy

EXPECTED = {
    "A100": "8f9491ac35f4ca54154e031912a258d5c406a2a6db6d739a82760afcf402a390",
    "RTX5000": "4b5e9235479b7d0e44a97429a69a919c649b00f420b45aeb4a337c65764491b5",
    "RTX_6000_Ada": "ab3670401d28e696588fe07f4f4d06990644d8ab8d4e35f1bd83b30027b8640b",
}

CANDIDATE_DIRS = [
    Path("/home/asadr/works/sus/models_gpus"),
    Path("/home/asadr/works/sus/GPU_estimates/sustainability"),
]
DEST = Path("data/legacy/narrow")
FILENAMES = {
    "A100": "A100.csv",
    "RTX5000": "RTX5000.csv",
    "RTX_6000_Ada": "RTX_6000_Ada.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_verified_source(key: str) -> Path:
    filename = FILENAMES[key]
    expected = EXPECTED[key]
    checked: list[str] = []
    for directory in CANDIDATE_DIRS:
        path = directory / filename
        checked.append(str(path))
        if not path.is_file():
            continue
        observed = sha256(path)
        if observed == expected:
            return path
        raise RuntimeError(
            f"Legacy source checksum mismatch for {path}: expected {expected}, observed {observed}"
        )
    raise FileNotFoundError(
        f"No verified legacy source found for {key}. Checked: {', '.join(checked)}"
    )


DEST.mkdir(parents=True, exist_ok=True)
for key in FILENAMES:
    src = resolve_verified_source(key)
    dst = DEST / FILENAMES[key]
    shutil.copy2(src, dst)
    observed_copy = sha256(dst)
    if observed_copy != EXPECTED[key]:
        raise RuntimeError(
            f"Staged copy checksum mismatch for {dst}: expected {EXPECTED[key]}, observed {observed_copy}"
        )
    print(f"WP1_SOURCE {key} {src}")

ns = runpy.run_path("scripts/run_wp1_benchmark_audit.py")
ns["EXPECTED_SHA256"].update(EXPECTED)
ns["main"]()
