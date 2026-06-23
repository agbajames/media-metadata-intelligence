import numpy as np

from metadata_intelligence.modeling import (
    MODEL_VERSION,
    build_prediction_output,
    predict_tag_probabilities,
    predict_tags_with_threshold,
    train_baseline_model,
)


def test_prediction_thresholding_returns_tag_confidence_pairs() -> None:
    probabilities = np.array([[0.2, 0.8, 0.6]])
    tag_classes = ["cult", "murder", "violence"]

    predictions = predict_tags_with_threshold(probabilities, tag_classes, threshold=0.5)

    assert predictions == [
        [
            {"tag": "murder", "confidence": 0.8},
            {"tag": "violence", "confidence": 0.6},
        ]
    ]


def test_model_pipeline_can_fit_tiny_multilabel_dataset() -> None:
    texts = [
        "detective murder mystery",
        "romantic comedy wedding",
        "violent revenge thriller",
        "funny romantic friendship",
    ]
    y_train = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
    )

    model = train_baseline_model(texts, y_train)
    probabilities = predict_tag_probabilities(model, ["murder detective"])

    assert probabilities.shape == (1, 2)


def test_structured_prediction_output_is_stable() -> None:
    output = build_prediction_output(
        title="Example Film",
        predicted_tags=[{"tag": "murder", "confidence": 0.82}],
        threshold=0.35,
    )

    assert output == {
        "title": "Example Film",
        "predicted_tags": [{"tag": "murder", "confidence": 0.82}],
        "metadata": {
            "model_version": MODEL_VERSION,
            "threshold": 0.35,
            "input_fields": ["title", "synopsis"],
        },
    }
