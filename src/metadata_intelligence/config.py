"""Project configuration and filesystem paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_MPST_PATH = RAW_DATA_DIR / "mpst_full_data.csv"
PROCESSED_MPST_PATH = PROCESSED_DATA_DIR / "mpst_processed.parquet"
TAG_CLASSES_PATH = PROCESSED_DATA_DIR / "tag_classes.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
TOP_TAGS_PATH = REPORTS_DIR / "top_tags.csv"

REQUIRED_COLUMNS = [
    "imdb_id",
    "title",
    "plot_synopsis",
    "tags",
    "split",
    "synopsis_source",
]
