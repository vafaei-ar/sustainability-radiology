#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML required: {exc}")

CONFIG = Path("config/prospective_vlm.yaml")
OUT = Path("results/wp3/prospective_vlm_protocol_validation.md")


def require(obj, path):
    cur = obj
    for key in path:
        if key not in cur:
            raise AssertionError("missing: " + ".".join(path))
        cur = cur[key]
    return cur


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())

    models = require(cfg, ["models", "primary_panel"])
    assert len(models) >= 4
    assert len(set(models)) == len(models)

    tasks = require(cfg, ["tasks"])
    task_ids = [x["id"] for x in tasks]
    assert len(task_ids) >= 4
    assert len(set(task_ids)) == len(task_ids)

    tiers = require(cfg, ["input_burden", "tiers"])
    tier_ids = [x["id"] for x in tiers]
    assert set(["single_2d", "multi_image", "sampled_volume"]).issubset(tier_ids)

    assert require(cfg, ["hardware", "warm_model"]) is True
    assert require(cfg, ["hardware", "exclude_model_load_from_tracked_interval"]) is True
    assert int(require(cfg, ["hardware", "warmup_cases"])) >= 1

    assert require(cfg, ["prompting", "temperature"]) == 0
    assert require(cfg, ["prompting", "do_sample"]) is False
    assert int(require(cfg, ["prompting", "max_new_tokens"])) > 0

    assert require(cfg, ["measurement", "tracked_boundary"]) == "preprocessing_plus_model_inference_plus_generation"
    assert require(cfg, ["measurement", "exclude_one_time_model_load"]) is True
    assert require(cfg, ["normalization", "primary"]) == "energy_per_completed_case"
    assert int(require(cfg, ["replication", "repeated_runs_per_cell"])) >= 3
    assert require(cfg, ["clinical_utility", "primary_framework"]) == "performance_energy_pareto"
    assert require(cfg, ["interpretation", "florence2_historical_cpu_runs_not_ranked_against_gpu_vlms"]) is True

    lines = [
        "# Prospective VLM protocol validation",
        "",
        "Status: **PASS**",
        "",
        f"Primary models: {len(models)}",
        f"Tasks: {len(tasks)}",
        f"Input-burden tiers: {len(tiers)}",
        f"Repeated runs per cell: {require(cfg, ['replication', 'repeated_runs_per_cell'])}",
        f"Phase-1 target cases: {require(cfg, ['cases', 'phase1_target'])}",
        "",
        "Validated safeguards:",
        "- model loading excluded from the primary tracked interval",
        "- warmup required before measurement",
        "- deterministic decoding requested",
        "- preprocessing, inference, and generation included in the tracked boundary",
        "- case-level normalization fixed",
        "- energy interpreted jointly with clinical utility",
        "- historical Florence-2 CPU runs excluded from GPU VLM ranking",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("PROSPECTIVE_VLM_PROTOCOL_VALIDATION_OK")


if __name__ == "__main__":
    main()
