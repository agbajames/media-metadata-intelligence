"""Text preprocessing utilities."""

import re

import pandas as pd


CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def basic_clean_text(text: str | float | None) -> str:
    """Apply conservative text cleaning suitable for later NLP models."""

    if text is None or pd.isna(text):
        return ""

    cleaned = CONTROL_CHARS_PATTERN.sub(" ", str(text))
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip().lower()


def combine_title_synopsis(df: pd.DataFrame) -> pd.Series:
    """Combine title and synopsis into a single text field."""

    titles = df["title"].fillna("").astype(str)
    synopses = df["plot_synopsis"].fillna("").astype(str)
    combined = titles + " [SEP] " + synopses
    return combined.apply(basic_clean_text)
