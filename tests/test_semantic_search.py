from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metadata_intelligence.semantic_search import (
    build_nearest_neighbors_index,
    cosine_distance_to_similarity,
    deduplicate_search_results,
    ensure_search_artifacts_exist,
    format_search_results,
    make_short_synopsis_preview,
    normalise_title_for_dedup,
    search_similar_content,
)


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        embeddings = []
        for text in texts:
            if "detective" in text:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return np.array(embeddings, dtype=np.float32)


def test_cosine_distance_converts_to_similarity_score() -> None:
    assert cosine_distance_to_similarity(0.0) == 1.0
    assert cosine_distance_to_similarity(0.25) == 0.75
    assert cosine_distance_to_similarity(1.5) == 0.0


def test_search_result_formatting_returns_expected_keys() -> None:
    records = [
        {
            "rank": 1,
            "imdb_id": "tt1",
            "title": "Movie",
            "similarity_score": 0.81234,
            "tags": ["murder", "dark"],
            "synopsis_source": "imdb",
            "short_synopsis_preview": "A detective investigates.",
        }
    ]

    formatted = format_search_results(records)

    assert set(formatted[0]) == {
        "rank",
        "imdb_id",
        "title",
        "similarity_score",
        "tags",
        "synopsis_source",
        "short_synopsis_preview",
    }
    assert formatted[0]["similarity_score"] == 0.8123


def test_top_k_behaviour_on_tiny_synthetic_embedding_matrix() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "imdb_id": ["tt1", "tt2", "tt3"],
            "title": ["Detective One", "Detective Two", "Comedy"],
            "tags": ["murder", "murder, suspenseful", "comedy"],
            "synopsis_source": ["imdb", "imdb", "imdb"],
            "plot_synopsis": ["Detective story.", "Another detective story.", "Funny story."],
        }
    )
    index = build_nearest_neighbors_index(embeddings)

    results = search_similar_content(
        "detective mystery",
        FakeEmbeddingModel(),
        index,
        embeddings,
        metadata,
        top_k=2,
    )

    assert len(results) == 2
    assert list(results["rank"]) == [1, 2]
    assert results.iloc[0]["title"] == "Detective One"


def test_normalise_title_for_dedup_collapses_case_and_whitespace() -> None:
    assert normalise_title_for_dedup("  The   Same MOVIE  ") == "the same movie"


def test_deduplicate_search_results_removes_duplicate_imdb_ids() -> None:
    records = [
        {"rank": 1, "imdb_id": "tt1", "title": "Movie A", "similarity_score": 0.9},
        {"rank": 2, "imdb_id": "tt1", "title": "Movie B", "similarity_score": 0.8},
        {"rank": 3, "imdb_id": "tt2", "title": "Movie C", "similarity_score": 0.7},
    ]

    deduped = deduplicate_search_results(records, top_k=5)

    assert [record["imdb_id"] for record in deduped] == ["tt1", "tt2"]


def test_deduplicate_search_results_removes_duplicate_normalised_titles() -> None:
    records = [
        {"rank": 1, "imdb_id": "tt1", "title": "Same Movie", "similarity_score": 0.9},
        {"rank": 2, "imdb_id": "tt2", "title": " same   movie ", "similarity_score": 0.8},
        {"rank": 3, "imdb_id": "tt3", "title": "Different Movie", "similarity_score": 0.7},
    ]

    deduped = deduplicate_search_results(records, top_k=5)

    assert [record["title"] for record in deduped] == ["Same Movie", "Different Movie"]


def test_deduplicate_search_results_keeps_highest_similarity_duplicate() -> None:
    records = [
        {"rank": 1, "imdb_id": "tt1", "title": "Same Movie", "similarity_score": 0.4},
        {"rank": 2, "imdb_id": "tt2", "title": "same movie", "similarity_score": 0.9},
        {"rank": 3, "imdb_id": "tt3", "title": "Other Movie", "similarity_score": 0.8},
    ]

    deduped = deduplicate_search_results(records, top_k=5)

    assert deduped[0]["imdb_id"] == "tt2"
    assert deduped[0]["similarity_score"] == 0.9


def test_deduplicate_search_results_returns_up_to_top_k_unique_records() -> None:
    records = [
        {"rank": 1, "imdb_id": "tt1", "title": "Movie A", "similarity_score": 0.9},
        {"rank": 2, "imdb_id": "tt2", "title": "Movie B", "similarity_score": 0.8},
        {"rank": 3, "imdb_id": "tt3", "title": "Movie C", "similarity_score": 0.7},
    ]

    deduped = deduplicate_search_results(records, top_k=2)

    assert len(deduped) == 2
    assert [record["rank"] for record in deduped] == [1, 2]


def test_short_synopsis_preview_is_truncated_safely() -> None:
    preview = make_short_synopsis_preview("word " * 100, max_chars=25)

    assert len(preview) <= 25
    assert preview.endswith("...")


def test_missing_search_artifacts_produce_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_search_index.py"):
        ensure_search_artifacts_exist([tmp_path / "missing.npy"])
