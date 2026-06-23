"""Run exploratory data analysis for the MPST dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.config import (  # noqa: E402
    EDA_SUMMARY_PATH,
    FIGURES_DIR,
    RAW_MPST_PATH,
    REPORTS_DIR,
    TOP_TAGS_PATH,
)
from metadata_intelligence.data_loader import load_mpst_data  # noqa: E402
from metadata_intelligence.eda import save_eda_outputs  # noqa: E402


def main() -> None:
    df = load_mpst_data(RAW_MPST_PATH)
    summary, top_tags = save_eda_outputs(df, REPORTS_DIR, FIGURES_DIR)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    top_tags.to_csv(TOP_TAGS_PATH, index=False)

    print(f"Saved EDA summary to {EDA_SUMMARY_PATH}")
    print(f"Saved top tags to {TOP_TAGS_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
