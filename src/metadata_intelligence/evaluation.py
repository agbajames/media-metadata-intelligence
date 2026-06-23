"""Evaluation helpers for multi-label tag prediction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def evaluate_multilabel_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core multi-label evaluation metrics."""

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate a set of global thresholds against probability predictions."""

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = evaluate_multilabel_predictions(y_true, y_pred)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def classification_report_by_tag(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag_classes: Sequence[str],
) -> pd.DataFrame:
    """Return precision, recall, and F1 metrics for each tag."""

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(tag_classes),
        zero_division=0,
        output_dict=True,
    )

    rows = []
    for tag in tag_classes:
        tag_report = report[tag]
        rows.append(
            {
                "tag": tag,
                "precision": float(tag_report["precision"]),
                "recall": float(tag_report["recall"]),
                "f1_score": float(tag_report["f1-score"]),
                "support": int(tag_report["support"]),
            }
        )
    return pd.DataFrame(rows)
