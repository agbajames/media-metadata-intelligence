"""Cached resource loading for the FastAPI service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock

import numpy as np
import pandas as pd

from metadata_intelligence.artifacts import load_model
from metadata_intelligence.config import (
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    EMBEDDING_MODEL_NAME,
    SEARCH_EMBEDDINGS_PATH,
    SEARCH_INDEX_PATH,
    SEARCH_METADATA_PATH,
    TAG_CLASSES_PATH,
)
from metadata_intelligence.semantic_search import load_embedding_model


@dataclass(frozen=True)
class TaggingResources:
    model: object
    tag_classes: list[str]


@dataclass(frozen=True)
class SearchResources:
    model: object
    embeddings: np.ndarray
    metadata_df: pd.DataFrame
    index: object


_tagging_resources: TaggingResources | None = None
_search_resources: SearchResources | None = None
_tagging_lock = Lock()
_search_lock = Lock()


def tagging_model_available() -> bool:
    """Return whether required tag prediction artifacts exist."""

    return BASELINE_MODEL_PATH.exists() and TAG_CLASSES_PATH.exists()


def search_index_available() -> bool:
    """Return whether required semantic search artifacts exist."""

    return (
        SEARCH_EMBEDDINGS_PATH.exists()
        and SEARCH_METADATA_PATH.exists()
        and SEARCH_INDEX_PATH.exists()
    )


def get_tagging_resources() -> TaggingResources:
    """Load and cache baseline tag model resources."""

    global _tagging_resources

    if _tagging_resources is not None:
        return _tagging_resources

    with _tagging_lock:
        if _tagging_resources is not None:
            return _tagging_resources

        missing = [
            str(path)
            for path in [BASELINE_MODEL_PATH, TAG_CLASSES_PATH]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Tagging artifacts are missing: "
                f"{', '.join(missing)}. Run python scripts/train_baseline.py first."
            )

        model = load_model(BASELINE_MODEL_PATH)
        tag_classes = json.loads(TAG_CLASSES_PATH.read_text(encoding="utf-8"))
        _tagging_resources = TaggingResources(model=model, tag_classes=tag_classes)
        return _tagging_resources


def get_search_resources() -> SearchResources:
    """Load and cache semantic search resources."""

    global _search_resources

    if _search_resources is not None:
        return _search_resources

    with _search_lock:
        if _search_resources is not None:
            return _search_resources

        missing = [
            str(path)
            for path in [SEARCH_EMBEDDINGS_PATH, SEARCH_METADATA_PATH, SEARCH_INDEX_PATH]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "Search artifacts are missing: "
                f"{', '.join(missing)}. Run python scripts/build_search_index.py first."
            )

        _search_resources = SearchResources(
            model=load_embedding_model(EMBEDDING_MODEL_NAME),
            embeddings=np.load(SEARCH_EMBEDDINGS_PATH),
            metadata_df=pd.read_parquet(SEARCH_METADATA_PATH),
            index=load_model(SEARCH_INDEX_PATH),
        )
        return _search_resources


def get_default_threshold(default: float = 0.5) -> float:
    """Load the tuned baseline threshold if available."""

    if not BASELINE_METRICS_PATH.exists():
        return default
    metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    return float(metrics.get("selected_threshold", default))
