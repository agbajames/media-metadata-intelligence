"""Project configuration and filesystem paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SEARCH_ARTIFACTS_DIR = ARTIFACTS_DIR / "search"

RAW_MPST_PATH = RAW_DATA_DIR / "mpst_full_data.csv"
PROCESSED_MPST_PATH = PROCESSED_DATA_DIR / "mpst_processed.parquet"
TAG_CLASSES_PATH = PROCESSED_DATA_DIR / "tag_classes.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
TOP_TAGS_PATH = REPORTS_DIR / "top_tags.csv"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_metrics.json"
BASELINE_THRESHOLDS_PATH = REPORTS_DIR / "baseline_thresholds.csv"
BASELINE_PER_TAG_REPORT_PATH = REPORTS_DIR / "baseline_per_tag_report.csv"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_EMBEDDINGS_PATH = SEARCH_ARTIFACTS_DIR / "mpst_embeddings.npy"
SEARCH_METADATA_PATH = SEARCH_ARTIFACTS_DIR / "mpst_search_metadata.parquet"
SEARCH_INDEX_PATH = SEARCH_ARTIFACTS_DIR / "mpst_nn_index.joblib"

REQUIRED_COLUMNS = [
    "imdb_id",
    "title",
    "plot_synopsis",
    "tags",
    "split",
    "synopsis_source",
]
