from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "analyze_wp3_three_model_100case.py"


def load_target():
    spec = importlib.util.spec_from_file_location("wp3_three_model_analysis", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = load_target()
    original_read_csv = mod.read_csv
    med_block_path = mod.MED / "block_summary.csv"

    def read_csv_with_canonical_block_fields(path: pathlib.Path):
        rows = original_read_csv(path)
        if path == med_block_path:
            for row in rows:
                row["gross_wh_per_case"] = row["gross_gpu_energy_wh_per_case"]
                row["net_wh_per_case"] = row["net_gpu_energy_wh_per_case"]
                row["median_case_seconds"] = row["median_case_elapsed_seconds"]
        return rows

    mod.read_csv = read_csv_with_canonical_block_fields
    mod.main()


if __name__ == "__main__":
    main()
