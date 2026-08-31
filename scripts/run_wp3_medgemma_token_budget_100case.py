from __future__ import annotations

"""Analysis-only recovery entrypoint for the completed token-budget measurements.

R8M5Q2K7 completed all 30 GPU measurement blocks and F1-RadGraph scoring before
failing during mixed-schema CSV serialization. This temporary entrypoint makes
the existing named RunRelay task recover the final statistical outputs from the
saved 30 block rows and 300 case rows. It does not load a model or repeat GPU
inference.
"""

from recover_wp3_medgemma_token_budget_analysis import main


if __name__ == "__main__":
    main()
