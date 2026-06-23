"""UI-independent helpers for the Streamlit demo."""

from __future__ import annotations

from typing import Any

import pandas as pd

from metadata_intelligence.api.dependencies import (
    SearchResources,
    TaggingResources,
    get_default_threshold,
    search_index_available,
    tagging_model_available,
)
from metadata_intelligence.config import EMBEDDING_MODEL_NAME
from metadata_intelligence.modeling import (
    MODEL_VERSION,
    predict_tag_probabilities,
    predict_tags_with_threshold,
)
from metadata_intelligence.preprocessing import basic_clean_text
from metadata_intelligence.semantic_search import (
    format_search_results,
    search_similar_content,
)


def generate_demo_report(
    title: str,
    synopsis: str,
    threshold: float,
    top_k_similar: int,
    tagging_resources: TaggingResources,
    search_resources: SearchResources,
    top_k_tags: int | None = None,
) -> dict[str, Any]:
    """Generate combined tag prediction and similar-content output."""

    combined_text = basic_clean_text(f"{title} [SEP] {synopsis}")
    probabilities = predict_tag_probabilities(tagging_resources.model, [combined_text])
    predicted_tags = predict_tags_with_threshold(
        probabilities,
        tagging_resources.tag_classes,
        threshold=threshold,
        top_k=top_k_tags,
    )[0]
    similar_results = search_similar_content(
        combined_text,
        search_resources.model,
        search_resources.index,
        search_resources.embeddings,
        search_resources.metadata_df,
        top_k=top_k_similar,
    )

    return {
        "input": {
            "title": title,
            "synopsis": synopsis,
        },
        "predicted_tags": predicted_tags,
        "similar_content": format_search_results(similar_results),
        "metadata": {
            "tagging_model": MODEL_VERSION,
            "retrieval_model": EMBEDDING_MODEL_NAME,
            "threshold": threshold,
            "top_k_similar": top_k_similar,
            "top_k_tags": top_k_tags,
        },
    }


def convert_predicted_tags_to_dataframe(predicted_tags: list[dict]) -> pd.DataFrame:
    """Convert predicted tags into a display-friendly dataframe."""

    df = pd.DataFrame(predicted_tags)
    if df.empty:
        return pd.DataFrame(columns=["tag", "confidence"])
    df["confidence"] = df["confidence"].astype(float)
    return df[["tag", "confidence"]]


def convert_similar_content_to_dataframe(similar_content: list[dict]) -> pd.DataFrame:
    """Convert similar-content records into a display-friendly dataframe."""

    df = pd.DataFrame(similar_content)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "title",
                "similarity_score",
                "tags",
                "short_synopsis_preview",
            ]
        )
    df["tags"] = df["tags"].apply(lambda tags: ", ".join(tags))
    return df[
        [
            "rank",
            "title",
            "similarity_score",
            "tags",
            "short_synopsis_preview",
        ]
    ]


def artifact_status() -> dict[str, bool]:
    """Return artifact availability for display in the demo."""

    return {
        "tagging_model_available": tagging_model_available(),
        "search_index_available": search_index_available(),
    }


def default_threshold() -> float:
    """Return the tuned baseline threshold if available."""

    return get_default_threshold(default=0.5)
