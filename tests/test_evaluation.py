import numpy as np

from metadata_intelligence.evaluation import (
    classification_report_by_tag,
    evaluate_multilabel_predictions,
    evaluate_thresholds,
)


def test_evaluation_metrics_return_expected_keys() -> None:
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1]])

    metrics = evaluate_multilabel_predictions(y_true, y_pred)

    assert set(metrics) == {
        "micro_f1",
        "macro_f1",
        "hamming_loss",
        "micro_precision",
        "micro_recall",
        "macro_precision",
        "macro_recall",
    }


def test_threshold_evaluation_works_over_multiple_thresholds() -> None:
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.8, 0.3], [0.4, 0.7]])

    results = evaluate_thresholds(y_true, y_prob, thresholds=[0.25, 0.5])

    assert list(results["threshold"]) == [0.25, 0.5]
    assert "micro_f1" in results.columns


def test_classification_report_by_tag_returns_one_row_per_tag() -> None:
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 0], [1, 1]])

    report = classification_report_by_tag(y_true, y_pred, ["murder", "comedy"])

    assert list(report["tag"]) == ["murder", "comedy"]
    assert {"precision", "recall", "f1_score", "support"}.issubset(report.columns)
