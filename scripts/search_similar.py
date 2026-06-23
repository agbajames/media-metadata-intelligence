"""Search for semantically similar MPST movies."""

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
    EMBEDDING_MODEL_NAME,
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_METADATA_PATH,
)
from metadata_intelligence.preprocessing import basic_clean_text  # noqa: E402
from metadata_intelligence.semantic_search import (  # noqa: E402
    ensure_search_artifacts_exist,
    format_search_results,
    load_embedding_model,
    search_similar_content,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find semantically similar MPST movies.")
    parser.add_argument("--title", required=True, help="Query movie title.")
    parser.add_argument("--synopsis", required=True, help="Query movie synopsis.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar movies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_search_artifacts_exist()

    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    embeddings = np.load(SEARCH_EMBEDDINGS_PATH)
    metadata_df = pd.read_parquet(SEARCH_METADATA_PATH)
    index = load_model(SEARCH_INDEX_PATH)

    query_text = basic_clean_text(f"{args.title} [SEP] {args.synopsis}")
    results = search_similar_content(
        query_text,
        model,
        index,
        embeddings,
        metadata_df,
        top_k=args.top_k,
    )
    output = {
        "query": {
            "title": args.title,
            "input_fields": ["title", "synopsis"],
        },
        "similar_content": format_search_results(results),
        "metadata": {
            "retrieval_model": EMBEDDING_MODEL_NAME,
            "index_type": "sklearn_nearest_neighbors_cosine",
            "top_k": args.top_k,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
