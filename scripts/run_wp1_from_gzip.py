#!/usr/bin/env python3
"""Materialize frozen WP1 benchmark CSVs from committed gzip inputs, then run audit."""
from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path
import runpy

FILES = {
    "A100": ("A100.csv.gz", "A100.csv", "8f9491ac35f4ca54154e031912a258d5c406a2a6db6d739a82760afcf402a390"),
    "RTX5000": ("RTX5000.csv.gz", "RTX5000.csv", "4b5e9235479b7d0e44a97429a69a919c649b00f420b45aeb4a337c65764491b5"),
    "RTX_6000_Ada": ("RTX_6000_Ada.csv.gz", "RTX_6000_Ada.csv", "ab3670401d28e696588fe07f4f4d06990644d8ab8d4e35f1bd83b30027b8640b"),
}
ROOT = Path("data/legacy/narrow")

for _, (gz_name, csv_name, expected) in FILES.items():
    src = ROOT / gz_name
    dst = ROOT / csv_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing committed input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    observed = hashlib.sha256(dst.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"Checksum mismatch for {dst}: {observed}")

ns = runpy.run_path("scripts/run_wp1_benchmark_audit.py")
ns["EXPECTED_SHA256"].update({key: spec[2] for key, spec in FILES.items()})
ns["main"]()
