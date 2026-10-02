"""Dense retrieval: embed the query, rank in pgvector, load the winners.

Two protocol-shaped methods exist deliberately. ``retrieve`` satisfies the Phase 0
:class:`~reglens.retrieval.base.Retriever` protocol; ``retrieve_with_usage`` additionally
returns the query-embedding token count, so cost can be attributed to the retrieval
stage without shared mutable state on the embedding client (which would race when the
API serves concurrent requests).
"""

from __future__ import annotations

from reglens.indexing.base import EmbeddingModel, VectorStore
from reglens.retrieval.base import RetrievedChunk, SearchFilters


class DenseRetriever:
    """Cosine similarity over pgvector, one query embedding per request."""

    name = "dense"

    def __init__(self, *, embedder: EmbeddingModel, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve_with_usage(
        self, query: str, *, top_k: int = 10, filters: SearchFilters | None = None
    ) -> tuple[list[RetrievedChunk], int, str | None]:
        """Return ranked chunks plus the embedding usage (tokens, model) for the trace."""
        usage_embedder = getattr(self.embedder, "embed_query_with_usage", None)
        if callable(usage_embedder):
            vector, tokens = usage_embedder(query)
        else:
            vector, tokens = self.embedder.embed_query(query), 0
        hits = self.store.search(vector, top_k=top_k, filters=filters)
        if not hits:
            return [], tokens, getattr(self.embedder, "name", None)
        rows = self.store.fetch([hit.chunk_id for hit in hits])
        by_id = {row["chunk_id"]: row for row in rows}
        chunks: list[RetrievedChunk] = []
        for hit in hits:
            row = by_id.get(hit.chunk_id)
            if row is None:  # pragma: no cover - fetch keeps order; defensive
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    doc_title=row["doc_title"],
                    issuer=row["issuer"],
                    doc_type=row["doc_type"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    clause_path=row["clause_path"],
                    section_title=row["section_title"],
                    issue_date=row["issue_date"],
                    effective_date=row["effective_date"],
                    score=hit.score,
                    dense_score=hit.score,
                    stage="retrieve",
                )
            )
        return chunks, tokens, getattr(self.embedder, "name", None)

    def retrieve(
        self, query: str, *, top_k: int = 10, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        chunks, _, _ = self.retrieve_with_usage(query, top_k=top_k, filters=filters)
        return chunks
