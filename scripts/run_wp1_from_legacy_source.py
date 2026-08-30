#!/usr/bin/env python3
"""Stage the three historical WP1 benchmark CSVs from the inventoried canonical source, then run the audit."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
import runpy

EXPECTED = {
    "A100": "30ac79634ffdf914c21fbcd7c37b42c877aea0957e1983227f92c8da58f40bb4",
    "RTX5000": "e2fcf2759ef3d0c940c13dea7435ec7bee0c094aeb7c88ef8e3c6861c73f5f56",
    "RTX_6000_Ada": "1ec0faa176e1a5b982971b674f0a6b891b78ffa2cf057ad070e9268f620809d1",
}

SOURCE_DIR = Path("/home/asadr/works/repos/sustainability-stroke/external_data/models_gpus")
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


DEST.mkdir(parents=True, exist_ok=True)
for key, filename in FILENAMES.items():
    src = SOURCE_DIR / filename
    if not src.is_file():
        raise FileNotFoundError(f"Missing inventoried WP1 source for {key}: {src}")
    observed_source = sha256(src)
    if observed_source != EXPECTED[key]:
        raise RuntimeError(
            f"WP1 source checksum mismatch for {src}: expected {EXPECTED[key]}, observed {observed_source}"
        )
    dst = DEST / filename
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
