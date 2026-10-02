"""Indexing interfaces (Phase 1-2).

Two embedding backends are needed for the PRD ablation (one hosted, one open-source), so
the interface is defined once here and both implementations must satisfy it. The same
applies to the keyword index: Postgres full-text in this implementation, OpenSearch behind
the same protocol if it is ever swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddedBatch:
    """Vectors plus the accounting needed to price and trace an embedding call."""

    vectors: list[list[float]]
    model: str
    input_tokens: int = 0
    dimensions: int = 0


@runtime_checkable
class EmbeddingModel(Protocol):
    """One embedding model behind a stable interface."""

    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        """Embed chunks for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Kept separate: some models use different prefixes here."""
        ...


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    score: float


@runtime_checkable
class KeywordIndex(Protocol):
    """Lexical retrieval: exact terms, clause numbers, acronyms."""

    name: str

    def index(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or update documents; returns the number written."""
        ...

    def search(
        self, query: str, *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[KeywordHit]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Dense retrieval over a vector index (pgvector in this project)."""

    name: str

    def upsert(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> int: ...

    def search(
        self, vector: list[float], *, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[KeywordHit]: ...


class IndexingNotImplementedError(NotImplementedError):
    """Raised by placeholders so a misconfigured run fails loudly, not quietly."""
