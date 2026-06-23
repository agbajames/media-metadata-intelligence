from types import SimpleNamespace

import numpy as np
import pandas as pd

from metadata_intelligence import demo
from metadata_intelligence.demo import (
    convert_predicted_tags_to_dataframe,
    convert_similar_content_to_dataframe,
    generate_demo_report,
)


class FakeTagModel:
    def predict_proba(self, texts):
        return np.array([[0.8, 0.2]])


def fake_search_results(*args, **kwargs):
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "imdb_id": "tt1",
                "title": "Similar Movie",
                "similarity_score": 0.75,
                "tags": ["murder", "suspenseful"],
                "synopsis_source": "imdb",
                "short_synopsis_preview": "A short preview.",
            }
        ]
    )


def test_generate_demo_report_with_mocked_resources(monkeypatch) -> None:
    monkeypatch.setattr(demo, "search_similar_content", fake_search_results)
    tagging_resources = SimpleNamespace(
        model=FakeTagModel(),
        tag_classes=["murder", "comedy"],
    )
    search_resources = SimpleNamespace(
        model=object(),
        embeddings=np.array([[1.0, 0.0]]),
        metadata_df=pd.DataFrame(),
        index=object(),
    )

    report = generate_demo_report(
        title="Example",
        synopsis="A detective investigates a murder.",
        threshold=0.5,
        top_k_similar=1,
        tagging_resources=tagging_resources,
        search_resources=search_resources,
    )

    assert report["input"]["title"] == "Example"
    assert report["predicted_tags"] == [{"tag": "murder", "confidence": 0.8}]
    assert report["similar_content"][0]["title"] == "Similar Movie"
    assert report["metadata"]["threshold"] == 0.5


def test_convert_predicted_tags_to_dataframe() -> None:
    df = convert_predicted_tags_to_dataframe(
        [{"tag": "murder", "confidence": 0.8}]
    )

    assert list(df.columns) == ["tag", "confidence"]
    assert df.iloc[0]["tag"] == "murder"


def test_convert_similar_content_to_dataframe_joins_tags() -> None:
    df = convert_similar_content_to_dataframe(
        [
            {
                "rank": 1,
                "title": "Similar Movie",
                "similarity_score": 0.75,
                "tags": ["murder", "dark"],
                "short_synopsis_preview": "Preview.",
            }
        ]
    )

    assert list(df.columns) == [
        "rank",
        "title",
        "similarity_score",
        "tags",
        "short_synopsis_preview",
    ]
    assert df.iloc[0]["tags"] == "murder, dark"
