"""Retrieval interfaces (Phase 1-3).

``SearchFilters`` carries the temporal and metadata constraints (FR2) rather than letting
each retriever invent its own: a filter that only some code paths honour is how a
superseded rule ends up cited as current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchFilters:
    """Metadata and validity constraints applied during retrieval."""

    issuer: str | None = None
    doc_type: str | None = None
    fiscal_period: str | None = None
    corpus_version: str | None = None
    # FR2: never return a chunk whose document is not valid on this date.
    as_of_date: date | None = None
    doc_ids: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk with its document context and the scores that produced it."""

    chunk_id: str
    doc_id: str
    text: str
    doc_title: str
    issuer: str
    doc_type: str
    page_start: int | None
    page_end: int | None
    clause_path: str | None
    section_title: str | None
    issue_date: date | None
    effective_date: date | None
    score: float
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    stage: str = "retrieve"

    def citation_label(self) -> str:
        """The ``[doc title, page, clause]`` form the answer prompt must emit (FR1)."""
        page = self.page_start if self.page_start is not None else "?"
        clause = self.clause_path or self.section_title or "n/a"
        return f"[{self.doc_title}, p.{page}, {clause}]"


@runtime_checkable
class Retriever(Protocol):
    """One retrieval strategy (dense, keyword, hybrid, reranked)."""

    name: str

    def retrieve(
        self, query: str, *, top_k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranking of a candidate list (Phase 2)."""

    name: str

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]: ...
