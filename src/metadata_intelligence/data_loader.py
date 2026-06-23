"""Data loading and validation for the MPST dataset."""

from pathlib import Path

import pandas as pd

from metadata_intelligence.config import REQUIRED_COLUMNS


def load_mpst_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the MPST CSV dataset."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"MPST dataset not found at {data_path}. "
            "Place the CSV at data/raw/mpst_full_data.csv."
        )

    df = pd.read_csv(data_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "MPST dataset is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    return df.drop_duplicates().reset_index(drop=True)
