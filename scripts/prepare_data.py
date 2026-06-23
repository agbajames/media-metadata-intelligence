"""Prepare the MPST dataset for later modelling phases."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.config import (  # noqa: E402
    PROCESSED_MPST_PATH,
    RAW_MPST_PATH,
    TAG_CLASSES_PATH,
)
from metadata_intelligence.data_loader import load_mpst_data  # noqa: E402
from metadata_intelligence.labels import add_tag_count, get_unique_tags, parse_tags  # noqa: E402
from metadata_intelligence.preprocessing import combine_title_synopsis  # noqa: E402


def main() -> None:
    df = load_mpst_data(RAW_MPST_PATH)
    df["parsed_tags"] = df["tags"].apply(parse_tags)
    df = add_tag_count(df)
    df["combined_text"] = combine_title_synopsis(df)

    PROCESSED_MPST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_MPST_PATH, index=False)
    TAG_CLASSES_PATH.write_text(
        json.dumps(get_unique_tags(df), indent=2),
        encoding="utf-8",
    )

    print(f"Saved processed data to {PROCESSED_MPST_PATH}")
    print(f"Saved tag classes to {TAG_CLASSES_PATH}")


if __name__ == "__main__":
    main()
