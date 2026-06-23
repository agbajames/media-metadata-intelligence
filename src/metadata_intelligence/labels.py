"""Utilities for parsing and encoding multi-label MPST tags."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def parse_tags(tag_string: str | float | None) -> list[str]:
    """Parse a comma-separated tag string into a clean list of tags."""

    if tag_string is None or pd.isna(tag_string):
        return []

    return [
        tag.strip()
        for tag in str(tag_string).split(",")
        if tag.strip()
    ]


def _ensure_parsed_tags(df: pd.DataFrame) -> pd.Series:
    if "parsed_tags" in df.columns:
        return df["parsed_tags"]
    return df["tags"].apply(parse_tags)


def get_unique_tags(df: pd.DataFrame) -> list[str]:
    """Return sorted unique tags present in a dataset."""

    parsed_tags = _ensure_parsed_tags(df)
    return sorted({tag for tags in parsed_tags for tag in tags})


def add_tag_count(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataset with a tag_count column."""

    result = df.copy()
    parsed_tags = _ensure_parsed_tags(result)
    if "parsed_tags" not in result.columns:
        result["parsed_tags"] = parsed_tags
    result["tag_count"] = parsed_tags.apply(len)
    return result


def binarise_tags(df: pd.DataFrame):
    """Binarise parsed tags using scikit-learn's MultiLabelBinarizer."""

    parsed_tags = _ensure_parsed_tags(df)
    binarizer = MultiLabelBinarizer()
    label_matrix = binarizer.fit_transform(parsed_tags)
    return label_matrix, binarizer
