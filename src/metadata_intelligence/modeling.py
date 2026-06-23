"""Baseline modelling utilities for multi-label tag prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


MODEL_VERSION = "baseline_tfidf_logreg_v1"


def build_tfidf_logreg_pipeline() -> Pipeline:
    """Build the TF-IDF plus OneVsRest logistic regression baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                    )
                ),
            ),
        ]
    )


def train_baseline_model(texts: Sequence[str], y_train: np.ndarray) -> Pipeline:
    """Fit the baseline model on training texts and multi-label targets."""

    model = build_tfidf_logreg_pipeline()
    model.fit(texts, y_train)
    return model


def predict_tag_probabilities(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """Predict per-tag probabilities for input texts."""

    return model.predict_proba(texts)


def predict_tags_with_threshold(
    probabilities: np.ndarray,
    tag_classes: Sequence[str],
    threshold: float,
    top_k: int | None = None,
) -> list[list[dict[str, float | str]]]:
    """Return tag/confidence predictions above a global threshold."""

    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    predictions = []
    for row in probabilities:
        tag_predictions = [
            {"tag": tag, "confidence": float(confidence)}
            for tag, confidence in zip(tag_classes, row, strict=True)
            if confidence >= threshold
        ]
        tag_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        if top_k is not None:
            tag_predictions = tag_predictions[:top_k]
        predictions.append(tag_predictions)

    return predictions


def build_prediction_output(
    title: str,
    predicted_tags: list[dict[str, float | str]],
    threshold: float,
    model_version: str = MODEL_VERSION,
) -> dict:
    """Build the structured prediction payload used by CLI and future APIs."""

    return {
        "title": title,
        "predicted_tags": predicted_tags,
        "metadata": {
            "model_version": model_version,
            "threshold": threshold,
            "input_fields": ["title", "synopsis"],
        },
    }
