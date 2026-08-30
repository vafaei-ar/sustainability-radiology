from __future__ import annotations

import statistics
from collections import defaultdict

import analyze_wp3_three_model_100case_impl as impl


def corrected_runtime_blocks(cases):
    out = {}
    for model in impl.MODELS:
        by_block = defaultdict(list)
        for case_index, row in cases[model].items():
            raw_block = row.get("block")
            block = int(raw_block) if raw_block not in (None, "") else ((int(case_index) - 1) // 10 + 1)
            by_block[block].append(float(row["elapsed_seconds"]))
        out[model] = {}
        for block in range(1, 11):
            vals = by_block[block]
            if len(vals) != 10:
                raise RuntimeError(f"Expected 10 case runtimes for {model} block {block}, found {len(vals)}")
            out[model][block] = statistics.median(vals)
    return out


impl.corrected_runtime_blocks = corrected_runtime_blocks

if __name__ == "__main__":
    impl.main()
