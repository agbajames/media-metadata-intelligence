"""Build reusable semantic search artifacts for MPST movies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.artifacts import save_model  # noqa: E402
from metadata_intelligence.config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    PROCESSED_MPST_PATH,
    SEARCH_ARTIFACTS_DIR,
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_METADATA_PATH,
)
from metadata_intelligence.semantic_search import (  # noqa: E402
    build_nearest_neighbors_index,
    embed_texts,
    load_embedding_model,
)


METADATA_COLUMNS = [
    "imdb_id",
    "title",
    "plot_synopsis",
    "tags",
    "parsed_tags",
    "split",
    "synopsis_source",
]


def main() -> None:
    if not PROCESSED_MPST_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_MPST_PATH}. "
            "Run python scripts/prepare_data.py first."
        )

    df = pd.read_parquet(PROCESSED_MPST_PATH)
    if "combined_text" not in df.columns:
        raise ValueError("Processed data is missing required column: combined_text")

    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    texts = df["combined_text"].fillna("").astype(str).tolist()
    embeddings = embed_texts(texts, model)
    index = build_nearest_neighbors_index(embeddings, metric="cosine")

    SEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SEARCH_EMBEDDINGS_PATH, embeddings)
    metadata_columns = [column for column in METADATA_COLUMNS if column in df.columns]
    df[metadata_columns].to_parquet(SEARCH_METADATA_PATH, index=False)
    save_model(index, SEARCH_INDEX_PATH)

    print(f"Indexed movies: {len(df)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Saved embeddings to {SEARCH_EMBEDDINGS_PATH}")
    print(f"Saved metadata to {SEARCH_METADATA_PATH}")
    print(f"Saved nearest-neighbor index to {SEARCH_INDEX_PATH}")


if __name__ == "__main__":
    main()
