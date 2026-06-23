"""Exploratory data analysis helpers for the MPST dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from metadata_intelligence.labels import add_tag_count, get_unique_tags, parse_tags


def top_tag_counts(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the top tag frequencies as a dataframe."""

    tag_counts = (
        df["tags"]
        .apply(parse_tags)
        .explode()
        .dropna()
        .value_counts()
        .head(n)
        .rename_axis("tag")
        .reset_index(name="count")
    )
    return tag_counts


def build_eda_summary(df: pd.DataFrame) -> dict:
    """Build a JSON-serialisable EDA summary."""

    df_with_counts = add_tag_count(df)
    tag_counts = df_with_counts["tag_count"]

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().to_dict().items()
        },
        "split_distribution": {
            str(split): int(count)
            for split, count in df["split"].value_counts(dropna=False).to_dict().items()
        },
        "unique_tag_count": int(len(get_unique_tags(df))),
        "tags_per_movie": {
            "average": float(tag_counts.mean()) if len(tag_counts) else 0.0,
            "median": float(tag_counts.median()) if len(tag_counts) else 0.0,
            "max": int(tag_counts.max()) if len(tag_counts) else 0,
        },
    }


def save_top_tags_plot(top_tags: pd.DataFrame, figures_dir: str | Path) -> Path:
    """Save a horizontal bar chart of the top tags."""

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "top_tags.png"

    plt.figure(figsize=(10, 6))
    if not top_tags.empty:
        plot_data = top_tags.sort_values("count")
        plt.barh(plot_data["tag"], plot_data["count"])
    plt.title("Top 20 Tags")
    plt.xlabel("Movie Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_tag_count_distribution_plot(df: pd.DataFrame, figures_dir: str | Path) -> Path:
    """Save a histogram of tag counts per movie."""

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tag_count_distribution.png"

    df_with_counts = add_tag_count(df)
    plt.figure(figsize=(8, 5))
    plt.hist(df_with_counts["tag_count"], bins=range(0, int(df_with_counts["tag_count"].max()) + 2))
    plt.title("Distribution of Tag Counts per Movie")
    plt.xlabel("Tags per Movie")
    plt.ylabel("Movie Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_eda_outputs(
    df: pd.DataFrame,
    reports_dir: str | Path,
    figures_dir: str | Path,
) -> tuple[dict, pd.DataFrame]:
    """Build and save EDA summary artifacts."""

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_eda_summary(df)
    top_tags = top_tag_counts(df)
    save_top_tags_plot(top_tags, figures_dir)
    save_tag_count_distribution_plot(df, figures_dir)
    return summary, top_tags
