"""Train and evaluate the TF-IDF + logistic regression baseline."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.artifacts import save_json, save_model  # noqa: E402
from metadata_intelligence.config import (  # noqa: E402
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_PER_TAG_REPORT_PATH,
    BASELINE_THRESHOLDS_PATH,
    PROCESSED_MPST_PATH,
    TAG_CLASSES_PATH,
)
from metadata_intelligence.evaluation import (  # noqa: E402
    classification_report_by_tag,
    evaluate_multilabel_predictions,
    evaluate_thresholds,
)
from metadata_intelligence.labels import get_unique_tags, parse_tags  # noqa: E402
from metadata_intelligence.modeling import (  # noqa: E402
    MODEL_VERSION,
    predict_tag_probabilities,
    train_baseline_model,
)


def normalise_parsed_tags(value: object) -> list[str]:
    """Convert stored parsed tags into a plain list of strings."""

    if isinstance(value, list):
        return [str(tag) for tag in value]
    if isinstance(value, np.ndarray):
        return [str(tag) for tag in value.tolist()]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return parse_tags(value)
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed]
    return []


def main() -> None:
    if not PROCESSED_MPST_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_MPST_PATH}. "
            "Run python scripts/prepare_data.py first."
        )

    df = pd.read_parquet(PROCESSED_MPST_PATH)
    required_columns = {"combined_text", "parsed_tags", "split"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Processed data is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    df["parsed_tags"] = df["parsed_tags"].apply(normalise_parsed_tags)
    tag_classes = get_unique_tags(df)
    mlb = MultiLabelBinarizer(classes=tag_classes)
    y_all = mlb.fit_transform(df["parsed_tags"])

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Expected train, val, and test rows in the split column.")

    x_train = df.loc[train_mask, "combined_text"].astype(str).tolist()
    x_val = df.loc[val_mask, "combined_text"].astype(str).tolist()
    x_test = df.loc[test_mask, "combined_text"].astype(str).tolist()
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    model = train_baseline_model(x_train, y_train)

    val_prob = predict_tag_probabilities(model, x_val)
    thresholds = np.round(np.arange(0.10, 0.61, 0.05), 2)
    threshold_results = evaluate_thresholds(y_val, val_prob, thresholds)
    selected_row = threshold_results.sort_values(
        ["micro_f1", "macro_f1", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    selected_threshold = float(selected_row["threshold"])

    val_pred = (val_prob >= selected_threshold).astype(int)
    validation_metrics = evaluate_multilabel_predictions(y_val, val_pred)

    test_prob = predict_tag_probabilities(model, x_test)
    test_pred = (test_prob >= selected_threshold).astype(int)
    test_metrics = evaluate_multilabel_predictions(y_test, test_pred)
    per_tag_report = classification_report_by_tag(y_test, test_pred, tag_classes)

    metrics = {
        "model_name": "TF-IDF + OneVsRest Logistic Regression",
        "model_version": MODEL_VERSION,
        "selected_threshold": selected_threshold,
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics": test_metrics,
        "train_size": int(train_mask.sum()),
        "validation_size": int(val_mask.sum()),
        "test_size": int(test_mask.sum()),
        "number_of_tags": int(len(tag_classes)),
    }

    save_model(model, BASELINE_MODEL_PATH)
    save_json(metrics, BASELINE_METRICS_PATH)
    save_json(tag_classes, TAG_CLASSES_PATH)
    BASELINE_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    threshold_results.to_csv(BASELINE_THRESHOLDS_PATH, index=False)
    per_tag_report.to_csv(BASELINE_PER_TAG_REPORT_PATH, index=False)

    print(f"Saved model to {BASELINE_MODEL_PATH}")
    print(f"Saved metrics to {BASELINE_METRICS_PATH}")
    print(f"Saved threshold results to {BASELINE_THRESHOLDS_PATH}")
    print(f"Saved per-tag report to {BASELINE_PER_TAG_REPORT_PATH}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Test micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Hamming loss: {test_metrics['hamming_loss']:.4f}")


if __name__ == "__main__":
    main()
