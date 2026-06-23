"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


MAX_TOP_K = 20


class NonEmptyTextMixin(BaseModel):
    title: str = Field(..., min_length=1)
    synopsis: str = Field(..., min_length=1)

    @field_validator("title", "synopsis")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class TagPredictionRequest(NonEmptyTextMixin):
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k_tags: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class SimilarSearchRequest(NonEmptyTextMixin):
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class IntelligenceReportRequest(NonEmptyTextMixin):
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k_tags: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    top_k_similar: int = Field(default=5, ge=1, le=MAX_TOP_K)


class PredictedTag(BaseModel):
    tag: str
    confidence: float


class SimilarContentItem(BaseModel):
    rank: int
    imdb_id: str
    title: str
    similarity_score: float
    tags: list[str]
    synopsis_source: str
    short_synopsis_preview: str


class TagPredictionResponse(BaseModel):
    title: str
    predicted_tags: list[PredictedTag]
    metadata: dict[str, Any]


class SimilarSearchQuery(BaseModel):
    title: str
    input_fields: list[str]


class SimilarSearchResponse(BaseModel):
    query: SimilarSearchQuery
    similar_content: list[SimilarContentItem]
    metadata: dict[str, Any]


class IntelligenceReportInput(BaseModel):
    title: str
    synopsis: str


class IntelligenceReportResponse(BaseModel):
    input: IntelligenceReportInput
    predicted_tags: list[PredictedTag]
    similar_content: list[SimilarContentItem]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    tagging_model_available: bool
    search_index_available: bool


class ErrorResponse(BaseModel):
    detail: str
