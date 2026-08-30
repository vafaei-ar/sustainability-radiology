#!/usr/bin/env python3
"""Read-only audit of historical VLM tracker logs for device/power evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/asadr/works/multimodal")
OUT = Path("results/wp2/tracker_device_audit.txt")
MODEL_TERMS = ("florence", "internvl", "paligemma", "moondream")
KEY_TERMS = ("gpu", "nvidia", "power", "device", "cpu", "dram", "carbon", "region")
MAX_FILES = 120
MAX_RECORDS_PER_FILE = 4


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def parse_json_lines(path: Path):
    records = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [], f"READ_ERROR {type(exc).__name__}: {exc}"
    stripped = text.strip()
    if not stripped:
        return [], "EMPTY"
    try:
        obj = json.loads(stripped)
        if isinstance(obj, list):
            records = obj[:MAX_RECORDS_PER_FILE] + (obj[-MAX_RECORDS_PER_FILE:] if len(obj) > MAX_RECORDS_PER_FILE else [])
        else:
            records = [obj]
        return records, None
    except Exception:
        pass
    lines = [ln for ln in text.splitlines() if ln.strip()]
    picks = lines[:MAX_RECORDS_PER_FILE] + (lines[-MAX_RECORDS_PER_FILE:] if len(lines) > MAX_RECORDS_PER_FILE else [])
    for ln in picks:
        try:
            records.append(json.loads(ln))
        except Exception:
            continue
    return records, None if records else "UNPARSEABLE"


def main():
    out = ["AUDIT\tWP2 tracker device and power evidence", f"ROOT\t{ROOT}"]
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)
    candidates = []
    for p in ROOT.rglob("data.json"):
        low = str(p).lower()
        if any(t in low for t in MODEL_TERMS):
            candidates.append(p)
    for p in ROOT.rglob("*.jsonl"):
        low = str(p).lower()
        if "impacttracker" in low and any(t in low for t in MODEL_TERMS):
            candidates.append(p)
    candidates = sorted(dict.fromkeys(candidates))[:MAX_FILES]
    out.append(f"FILES\t{len(candidates)}")
    for path in candidates:
        out.append(f"FILE\t{path}")
        records, err = parse_json_lines(path)
        if err:
            out.append(f"STATUS\t{err}")
            continue
        seen = set()
        for rec in records:
            for key, value in flatten(rec):
                lk = key.lower()
                if any(term in lk for term in KEY_TERMS):
                    pair = (key, repr(value))
                    if pair not in seen:
                        seen.add(pair)
                        val = repr(value)
                        if len(val) > 500:
                            val = val[:500] + "..."
                        out.append(f"EVIDENCE\t{key}\t{val}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"WP2_TRACKER_DEVICE_AUDIT_OK files={len(candidates)}")


if __name__ == "__main__":
    main()
