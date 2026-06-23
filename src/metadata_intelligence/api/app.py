"""FastAPI application for the Media Metadata Intelligence System."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from metadata_intelligence.api.dependencies import (
    get_default_threshold,
    get_search_resources,
    get_tagging_resources,
    search_index_available,
    tagging_model_available,
)
from metadata_intelligence.api.schemas import (
    HealthResponse,
    IntelligenceReportRequest,
    IntelligenceReportResponse,
    SimilarSearchRequest,
    SimilarSearchResponse,
    TagPredictionRequest,
    TagPredictionResponse,
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


SERVICE_NAME = "Media Metadata Intelligence API"
SERVICE_VERSION = "0.1.0"


app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "API for multi-label media tag prediction, semantic search and "
        "similar-content recommendation."
    ),
    version=SERVICE_VERSION,
)


def _combined_text(title: str, synopsis: str) -> str:
    return basic_clean_text(f"{title} [SEP] {synopsis}")


def _resource_error(error: RuntimeError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(error))


@app.get("/")
def root() -> dict:
    """Return service information and available endpoints."""

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "available_endpoints": [
            "/health",
            "/predict-tags",
            "/search-similar",
            "/intelligence-report",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and artifact availability."""

    tagging_available = tagging_model_available()
    search_available = search_index_available()
    status = "ok" if tagging_available and search_available else "degraded"
    return HealthResponse(
        status=status,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        tagging_model_available=tagging_available,
        search_index_available=search_available,
    )


@app.post("/predict-tags", response_model=TagPredictionResponse)
def predict_tags(request: TagPredictionRequest) -> dict:
    """Predict MPST tags for a title and synopsis."""

    try:
        resources = get_tagging_resources()
    except RuntimeError as error:
        raise _resource_error(error) from error

    threshold = request.threshold if request.threshold is not None else get_default_threshold()
    probabilities = predict_tag_probabilities(
        resources.model,
        [_combined_text(request.title, request.synopsis)],
    )
    predicted_tags = predict_tags_with_threshold(
        probabilities,
        resources.tag_classes,
        threshold=threshold,
        top_k=request.top_k_tags,
    )[0]

    return {
        "title": request.title,
        "predicted_tags": predicted_tags,
        "metadata": {
            "model_version": MODEL_VERSION,
            "threshold": threshold,
            "input_fields": ["title", "synopsis"],
        },
    }


@app.post("/search-similar", response_model=SimilarSearchResponse)
def search_similar(request: SimilarSearchRequest) -> dict:
    """Return semantically similar MPST movies."""

    try:
        resources = get_search_resources()
    except RuntimeError as error:
        raise _resource_error(error) from error

    results = search_similar_content(
        _combined_text(request.title, request.synopsis),
        resources.model,
        resources.index,
        resources.embeddings,
        resources.metadata_df,
        top_k=request.top_k,
    )
    return {
        "query": {
            "title": request.title,
            "input_fields": ["title", "synopsis"],
        },
        "similar_content": format_search_results(results),
        "metadata": {
            "retrieval_model": EMBEDDING_MODEL_NAME,
            "index_type": "sklearn_nearest_neighbors_cosine",
            "top_k": request.top_k,
        },
    }


@app.post("/intelligence-report", response_model=IntelligenceReportResponse)
def intelligence_report(request: IntelligenceReportRequest) -> dict:
    """Return tag predictions and similar-content results."""

    try:
        tagging_resources = get_tagging_resources()
        search_resources = get_search_resources()
    except RuntimeError as error:
        raise _resource_error(error) from error

    threshold = request.threshold if request.threshold is not None else get_default_threshold()
    combined_text = _combined_text(request.title, request.synopsis)

    probabilities = predict_tag_probabilities(tagging_resources.model, [combined_text])
    predicted_tags = predict_tags_with_threshold(
        probabilities,
        tagging_resources.tag_classes,
        threshold=threshold,
        top_k=request.top_k_tags,
    )[0]
    similar_results = search_similar_content(
        combined_text,
        search_resources.model,
        search_resources.index,
        search_resources.embeddings,
        search_resources.metadata_df,
        top_k=request.top_k_similar,
    )

    return {
        "input": {
            "title": request.title,
            "synopsis": request.synopsis,
        },
        "predicted_tags": predicted_tags,
        "similar_content": format_search_results(similar_results),
        "metadata": {
            "tagging_model": MODEL_VERSION,
            "retrieval_model": EMBEDDING_MODEL_NAME,
            "threshold": threshold,
            "top_k_similar": request.top_k_similar,
        },
    }
