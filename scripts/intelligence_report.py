"""Produce a combined tag prediction and semantic similarity report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.artifacts import load_model  # noqa: E402
from metadata_intelligence.config import (  # noqa: E402
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_METADATA_PATH,
    TAG_CLASSES_PATH,
)
from metadata_intelligence.modeling import (  # noqa: E402
    MODEL_VERSION,
    predict_tag_probabilities,
    predict_tags_with_threshold,
)
from metadata_intelligence.preprocessing import basic_clean_text  # noqa: E402
from metadata_intelligence.semantic_search import (  # noqa: E402
    ensure_search_artifacts_exist,
    format_search_results,
    load_embedding_model,
    search_similar_content,
)


def load_selected_threshold(default: float = 0.5) -> float:
    """Load the tuned Phase 2 threshold if available."""

    if not BASELINE_METRICS_PATH.exists():
        return default
    metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    return float(metrics.get("selected_threshold", default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tag predictions and similar-content results."
    )
    parser.add_argument("--title", required=True, help="Movie title.")
    parser.add_argument("--synopsis", required=True, help="Movie synopsis.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Tag probability threshold. Defaults to tuned threshold if available.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar movies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found at {BASELINE_MODEL_PATH}. "
            "Run python scripts/train_baseline.py first."
        )
    if not TAG_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Tag classes not found at {TAG_CLASSES_PATH}. "
            "Run python scripts/train_baseline.py first."
        )
    ensure_search_artifacts_exist()

    threshold = args.threshold if args.threshold is not None else load_selected_threshold()
    combined_text = basic_clean_text(f"{args.title} [SEP] {args.synopsis}")

    tag_model = load_model(BASELINE_MODEL_PATH)
    tag_classes = json.loads(TAG_CLASSES_PATH.read_text(encoding="utf-8"))
    tag_probabilities = predict_tag_probabilities(tag_model, [combined_text])
    predicted_tags = predict_tags_with_threshold(
        tag_probabilities,
        tag_classes,
        threshold=threshold,
    )[0]

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    embeddings = np.load(SEARCH_EMBEDDINGS_PATH)
    metadata_df = pd.read_parquet(SEARCH_METADATA_PATH)
    search_index = load_model(SEARCH_INDEX_PATH)
    similar_results = search_similar_content(
        combined_text,
        embedding_model,
        search_index,
        embeddings,
        metadata_df,
        top_k=args.top_k,
    )

    output = {
        "input": {
            "title": args.title,
            "synopsis": args.synopsis,
        },
        "predicted_tags": predicted_tags,
        "similar_content": format_search_results(similar_results),
        "metadata": {
            "tagging_model": MODEL_VERSION,
            "retrieval_model": EMBEDDING_MODEL_NAME,
            "threshold": threshold,
            "top_k": args.top_k,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
