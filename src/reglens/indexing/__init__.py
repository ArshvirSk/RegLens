"""Indexing: embeddings, vector store (pgvector), keyword index (Postgres FTS).

Phase 0 defines contracts only. Phase 1 implements the hosted embedding model plus
pgvector writes; Phase 2 adds the open-source model and the keyword arm.
"""

from reglens.indexing.base import (
    EmbeddedBatch,
    EmbeddingModel,
    IndexingNotImplementedError,
    KeywordHit,
    KeywordIndex,
    VectorStore,
)

__all__ = [
    "EmbeddedBatch",
    "EmbeddingModel",
    "IndexingNotImplementedError",
    "KeywordHit",
    "KeywordIndex",
    "VectorStore",
]
