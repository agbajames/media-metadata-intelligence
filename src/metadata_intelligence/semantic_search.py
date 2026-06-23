"""Semantic embedding and nearest-neighbor retrieval utilities."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from metadata_intelligence.config import (
    EMBEDDING_MODEL_NAME,
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_METADATA_PATH,
)
from metadata_intelligence.labels import parse_tags


SEARCH_ARTIFACT_PATHS = [
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_METADATA_PATH,
    SEARCH_INDEX_PATH,
]
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load a sentence-transformers embedding model."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Sequence[str], model: Any, batch_size: int = 32) -> np.ndarray:
    """Generate sentence embeddings for a sequence of texts."""

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_nearest_neighbors_index(
    embeddings: np.ndarray,
    metric: str = "cosine",
) -> NearestNeighbors:
    """Build a sklearn NearestNeighbors index over embeddings."""

    index = NearestNeighbors(metric=metric, algorithm="brute")
    index.fit(embeddings)
    return index


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to a bounded similarity score."""

    return max(0.0, min(1.0, 1.0 - float(distance)))


def make_short_synopsis_preview(text: object, max_chars: int = 180) -> str:
    """Create a compact synopsis preview without cutting awkwardly far past the limit."""

    if text is None or pd.isna(text):
        return ""

    preview = " ".join(str(text).split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + "..."


def normalise_tags(value: object) -> list[str]:
    """Convert tag values from raw CSV or parquet storage into a list."""

    if isinstance(value, list):
        return [str(tag) for tag in value]
    if isinstance(value, np.ndarray):
        return [str(tag) for tag in value.tolist()]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
    return parse_tags(value)


def normalise_title_for_dedup(title: str) -> str:
    """Normalise a movie title for duplicate filtering."""

    return WHITESPACE_PATTERN.sub(" ", str(title).strip().lower())


def deduplicate_search_results(records: list[dict], top_k: int) -> list[dict]:
    """Keep the highest-similarity unique search records by imdb_id and title."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    sorted_records = sorted(
        records,
        key=lambda record: float(record["similarity_score"]),
        reverse=True,
    )
    seen_imdb_ids: set[str] = set()
    seen_titles: set[str] = set()
    deduplicated = []

    for record in sorted_records:
        imdb_id = str(record.get("imdb_id", "")).strip()
        title_key = normalise_title_for_dedup(str(record.get("title", "")))
        if imdb_id and imdb_id in seen_imdb_ids:
            continue
        if title_key and title_key in seen_titles:
            continue

        result = dict(record)
        result["rank"] = len(deduplicated) + 1
        deduplicated.append(result)
        if imdb_id:
            seen_imdb_ids.add(imdb_id)
        if title_key:
            seen_titles.add(title_key)
        if len(deduplicated) == top_k:
            break

    return deduplicated


def search_similar_content(
    query_text: str,
    model: Any,
    index: NearestNeighbors,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Embed a query and return the most similar indexed movies."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if len(metadata_df) != len(embeddings):
        raise ValueError("metadata_df and embeddings must contain the same number of rows.")

    candidate_count = min(max(top_k * 5, top_k + 20), len(metadata_df))
    query_embedding = embed_texts([query_text], model)
    distances, indices = index.kneighbors(query_embedding, n_neighbors=candidate_count)

    records = []
    for rank, (distance, row_index) in enumerate(zip(distances[0], indices[0], strict=True), start=1):
        metadata = metadata_df.iloc[int(row_index)]
        tags_value = metadata.get("parsed_tags", metadata.get("tags", ""))
        records.append(
            {
                "rank": rank,
                "imdb_id": metadata.get("imdb_id", ""),
                "title": metadata.get("title", ""),
                "similarity_score": cosine_distance_to_similarity(float(distance)),
                "tags": normalise_tags(tags_value),
                "synopsis_source": metadata.get("synopsis_source", ""),
                "short_synopsis_preview": make_short_synopsis_preview(
                    metadata.get("plot_synopsis", "")
                ),
            }
        )

    return pd.DataFrame(deduplicate_search_results(records, top_k=top_k))


def format_search_results(results_df_or_records: pd.DataFrame | Sequence[dict]) -> list[dict]:
    """Format search results into JSON-friendly records."""

    if isinstance(results_df_or_records, pd.DataFrame):
        records = results_df_or_records.to_dict(orient="records")
    else:
        records = list(results_df_or_records)

    formatted = []
    for record in records:
        formatted.append(
            {
                "rank": int(record["rank"]),
                "imdb_id": str(record["imdb_id"]),
                "title": str(record["title"]),
                "similarity_score": round(float(record["similarity_score"]), 4),
                "tags": normalise_tags(record.get("tags", [])),
                "synopsis_source": str(record.get("synopsis_source", "")),
                "short_synopsis_preview": make_short_synopsis_preview(
                    record.get("short_synopsis_preview", "")
                ),
            }
        )
    return formatted


def missing_search_artifacts(paths: Sequence[str | Path] = SEARCH_ARTIFACT_PATHS) -> list[Path]:
    """Return search artifact paths that do not exist."""

    return [Path(path) for path in paths if not Path(path).exists()]


def ensure_search_artifacts_exist(paths: Sequence[str | Path] = SEARCH_ARTIFACT_PATHS) -> None:
    """Raise a clear error if search artifacts have not been built."""

    missing = missing_search_artifacts(paths)
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Search artifacts are missing: "
            f"{missing_list}. Run python scripts/build_search_index.py first."
        )
