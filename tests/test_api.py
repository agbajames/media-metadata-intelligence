from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from metadata_intelligence.api import app as api_app
from metadata_intelligence.api.app import app


client = TestClient(app)


class FakeTagModel:
    def predict_proba(self, texts):
        return np.array([[0.8, 0.3]])


def fake_search_results(*args, **kwargs):
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "imdb_id": "tt1",
                "title": "Similar Movie",
                "similarity_score": 0.75,
                "tags": ["murder"],
                "synopsis_source": "imdb",
                "short_synopsis_preview": "A short preview.",
            }
        ]
    )


def test_root_returns_service_information() -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Media Metadata Intelligence API"
    assert "/predict-tags" in body["available_endpoints"]


def test_health_returns_expected_keys(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "tagging_model_available", lambda: True)
    monkeypatch.setattr(api_app, "search_index_available", lambda: False)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "status",
        "service",
        "version",
        "tagging_model_available",
        "search_index_available",
    }
    assert body["status"] == "degraded"


def test_request_validation_rejects_empty_title() -> None:
    response = client.post(
        "/predict-tags",
        json={"title": "   ", "synopsis": "A valid synopsis."},
    )

    assert response.status_code == 422


def test_request_validation_rejects_empty_synopsis() -> None:
    response = client.post(
        "/search-similar",
        json={"title": "A title", "synopsis": ""},
    )

    assert response.status_code == 422


def test_request_validation_rejects_invalid_threshold() -> None:
    response = client.post(
        "/predict-tags",
        json={"title": "A title", "synopsis": "A synopsis", "threshold": 1.5},
    )

    assert response.status_code == 422


def test_request_validation_rejects_invalid_top_k() -> None:
    response = client.post(
        "/search-similar",
        json={"title": "A title", "synopsis": "A synopsis", "top_k": 25},
    )

    assert response.status_code == 422


def test_predict_tags_response_shape_is_stable(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "get_tagging_resources",
        lambda: SimpleNamespace(model=FakeTagModel(), tag_classes=["murder", "comedy"]),
    )
    monkeypatch.setattr(api_app, "get_default_threshold", lambda: 0.5)

    response = client.post(
        "/predict-tags",
        json={"title": "Example", "synopsis": "A murder mystery."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Example"
    assert body["predicted_tags"] == [{"tag": "murder", "confidence": 0.8}]
    assert body["metadata"]["threshold"] == 0.5
    assert body["metadata"]["input_fields"] == ["title", "synopsis"]


def test_search_similar_response_shape_is_stable(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "get_search_resources",
        lambda: SimpleNamespace(
            model=object(),
            embeddings=np.array([[1.0, 0.0]]),
            metadata_df=pd.DataFrame(),
            index=object(),
        ),
    )
    monkeypatch.setattr(api_app, "search_similar_content", fake_search_results)

    response = client.post(
        "/search-similar",
        json={"title": "Example", "synopsis": "A mystery.", "top_k": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"]["title"] == "Example"
    assert body["similar_content"][0]["title"] == "Similar Movie"
    assert body["metadata"]["top_k"] == 1


def test_intelligence_report_response_shape_is_stable(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "get_tagging_resources",
        lambda: SimpleNamespace(model=FakeTagModel(), tag_classes=["murder", "comedy"]),
    )
    monkeypatch.setattr(
        api_app,
        "get_search_resources",
        lambda: SimpleNamespace(
            model=object(),
            embeddings=np.array([[1.0, 0.0]]),
            metadata_df=pd.DataFrame(),
            index=object(),
        ),
    )
    monkeypatch.setattr(api_app, "get_default_threshold", lambda: 0.5)
    monkeypatch.setattr(api_app, "search_similar_content", fake_search_results)

    response = client.post(
        "/intelligence-report",
        json={
            "title": "Example",
            "synopsis": "A murder mystery.",
            "top_k_similar": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["input"]["title"] == "Example"
    assert body["predicted_tags"][0]["tag"] == "murder"
    assert body["similar_content"][0]["imdb_id"] == "tt1"
    assert body["metadata"]["top_k_similar"] == 1
