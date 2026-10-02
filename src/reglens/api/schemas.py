"""API schemas.

The disclaimer is a field on every answer payload, not just a UI string, because the
requirement is "must display a notice" and the API is a surface too.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = "Informational only. Not legal or investment advice."

Route = Literal["text", "numeric", "hybrid", "unknown"]
RefusalReason = Literal[
    "not_implemented",
    "no_evidence",
    "below_support_threshold",
    "out_of_scope",
    "temporal_window_empty",
]


class QueryFilters(BaseModel):
    """Metadata filters applied to retrieval (issuer / document type / period)."""

    issuer: str | None = Field(default=None, description="e.g. RBI, SEBI, SBI, HDFCBANK")
    doc_type: str | None = Field(default=None, description="e.g. master_direction, transcript")
    fiscal_period: str | None = Field(default=None, description="e.g. FY2025-Q3")
    corpus_version: str | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    as_of_date: date | None = Field(
        default=None,
        description="Answer as of this date: never cite documents outside their validity window.",
    )
    filters: QueryFilters | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    experiment: str | None = Field(
        default=None, description="Override the active experiment config"
    )


class Citation(BaseModel):
    """A citation that must map to a real chunk (FR1)."""

    doc_id: str
    doc_title: str
    page: int | None = None
    clause: str | None = None
    chunk_id: str | None = None
    quote: str = ""
    verified: bool | None = Field(
        default=None, description="Set by the Phase 4 citation verifier; null until then"
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    route: Route = "unknown"
    config_hash: str
    corpus_version: str
    trace_id: str
    latency_ms: int
    cost_usd: float
    model: str | None = None
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    env: str
    corpus_version: str
    experiment: str
    config_hash: str
    database: str
    disclaimer: str = DISCLAIMER


class ReadyResponse(BaseModel):
    ready: bool
    database: dict[str, object]
    migrations: dict[str, object]


class ConfigResponse(BaseModel):
    experiment: str
    config_hash: str
    corpus_version: str
    summary: list[str]
    embedding: str
    generator: str
    pricing: dict[str, object]
    paths: dict[str, str]


class NotImplementedResponse(BaseModel):
    """Honest scaffolding: the endpoint exists, the capability does not yet."""

    detail: str
    phase: int
    status: Literal["not_implemented"] = "not_implemented"


class TraceSummary(BaseModel):
    trace_id: str
    name: str
    route: str | None = None
    latency_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    started_at: str | None = None


class TracesResponse(BaseModel):
    traces: list[TraceSummary]
    percentiles: dict[str, float | None]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    label: Literal["helpful", "wrong", "missing_source", "other"]
    comment: str | None = Field(default=None, max_length=2000)
