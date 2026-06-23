from pathlib import Path

import pandas as pd
import pytest

from metadata_intelligence.data_loader import load_mpst_data


def test_load_mpst_data_validates_required_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "missing_columns.csv"
    pd.DataFrame({"title": ["Movie"]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_mpst_data(data_path)


def test_load_mpst_data_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MPST dataset not found"):
        load_mpst_data(tmp_path / "missing.csv")


def test_load_mpst_data_drops_exact_duplicates(tmp_path: Path) -> None:
    data_path = tmp_path / "mpst.csv"
    row = {
        "imdb_id": "tt1",
        "title": "Movie",
        "plot_synopsis": "A plot.",
        "tags": "cult, revenge",
        "split": "train",
        "synopsis_source": "imdb",
    }
    pd.DataFrame([row, row]).to_csv(data_path, index=False)

    loaded = load_mpst_data(data_path)

    assert len(loaded) == 1
