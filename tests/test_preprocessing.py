import pandas as pd

from metadata_intelligence.labels import binarise_tags, parse_tags
from metadata_intelligence.preprocessing import basic_clean_text, combine_title_synopsis


def test_parse_tags_handles_comma_separated_tags() -> None:
    assert parse_tags("cult, revenge, murder") == ["cult", "revenge", "murder"]


def test_parse_tags_handles_empty_and_missing_values() -> None:
    assert parse_tags("") == []
    assert parse_tags(None) == []
    assert parse_tags(float("nan")) == []


def test_combine_title_synopsis_creates_usable_text() -> None:
    df = pd.DataFrame(
        {
            "title": ["My Movie"],
            "plot_synopsis": ["Line one.\nLine two."],
        }
    )

    combined = combine_title_synopsis(df)

    assert len(combined) == 1
    assert combined.iloc[0] == "my movie [sep] line one. line two."


def test_basic_clean_text_is_conservative() -> None:
    assert basic_clean_text("  Hello\tWORLD\x00  ") == "hello world"


def test_binarise_tags_returns_expected_number_of_rows() -> None:
    df = pd.DataFrame({"tags": ["cult, revenge", "murder", ""]})

    label_matrix, binarizer = binarise_tags(df)

    assert label_matrix.shape[0] == 3
    assert set(binarizer.classes_) == {"cult", "murder", "revenge"}
