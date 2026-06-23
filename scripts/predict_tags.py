"""Predict MPST tags for a title and synopsis with the baseline model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metadata_intelligence.artifacts import load_model  # noqa: E402
from metadata_intelligence.config import (  # noqa: E402
    BASELINE_METRICS_PATH,
    BASELINE_MODEL_PATH,
    TAG_CLASSES_PATH,
)
from metadata_intelligence.modeling import (  # noqa: E402
    MODEL_VERSION,
    build_prediction_output,
    predict_tag_probabilities,
    predict_tags_with_threshold,
)
from metadata_intelligence.preprocessing import basic_clean_text  # noqa: E402


def load_selected_threshold(default: float = 0.35) -> float:
    """Load the tuned threshold if metrics are available."""

    if not BASELINE_METRICS_PATH.exists():
        return default

    metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    return float(metrics.get("selected_threshold", default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MPST tags for one movie.")
    parser.add_argument("--title", required=True, help="Movie title.")
    parser.add_argument("--synopsis", required=True, help="Movie synopsis.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold. Defaults to tuned threshold if available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional maximum number of tags to return.",
    )
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
            "Run python scripts/prepare_data.py or python scripts/train_baseline.py first."
        )

    threshold = args.threshold if args.threshold is not None else load_selected_threshold()
    model = load_model(BASELINE_MODEL_PATH)
    tag_classes = json.loads(TAG_CLASSES_PATH.read_text(encoding="utf-8"))

    combined_text = basic_clean_text(f"{args.title} [SEP] {args.synopsis}")
    probabilities = predict_tag_probabilities(model, [combined_text])
    predicted_tags = predict_tags_with_threshold(
        probabilities,
        tag_classes,
        threshold=threshold,
        top_k=args.top_k,
    )[0]
    output = build_prediction_output(
        title=args.title,
        predicted_tags=predicted_tags,
        threshold=threshold,
        model_version=MODEL_VERSION,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
