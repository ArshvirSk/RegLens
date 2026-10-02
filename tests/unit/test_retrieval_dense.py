"""Dense retriever: protocol compliance, row mapping, and usage accounting."""

from __future__ import annotations

from datetime import date

from reglens.indexing.base import KeywordHit
from reglens.retrieval.base import Retriever, SearchFilters
from reglens.retrieval.dense import DenseRetriever


class FakeEmbedderWithUsage:
    name = "gemini-embedding-001"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query_with_usage(self, text: str) -> tuple[list[float], int]:
        self.queries.append(text)
        return [0.25, 0.75], 9

    def embed_documents(self, texts: list[str]):  # type: ignore[no-untyped-def]
        raise AssertionError("not used by retrieval")


class FakeEmbedderWithoutUsage:
    """A future provider that only satisfies the base protocol."""

    name = "other"
    dimensions = 2

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]):  # type: ignore[no-untyped-def]
        raise AssertionError("not used by retrieval")


class FakeStore:
    def __init__(self) -> None:
        self.searches: list[dict[str, object]] = []

    def search(self, vector: list[float], *, top_k: int, filters=None):  # type: ignore[no-untyped-def]
        self.searches.append({"vector": vector, "top_k": top_k, "filters": filters})
        # Deliberately out of order: fetch order must follow these hits, not row order.
        return [
            KeywordHit(chunk_id="c2", score=0.8),
            KeywordHit(chunk_id="c1", score=0.6),
            KeywordHit(chunk_id="gone", score=0.5),
        ]

    def fetch(self, chunk_ids: list[str]):  # type: ignore[no-untyped-def]
        rows = {
            "c1": {
                "chunk_id": "c1",
                "doc_id": "doc_a",
                "text": "first chunk",
                "doc_title": "Title A",
                "issuer": "RBI",
                "doc_type": "master_direction",
                "page_start": 3,
                "page_end": 4,
                "clause_path": "2.1",
                "section_title": None,
                "issue_date": date(2020, 9, 4),
                "effective_date": None,
            },
            "c2": {
                "chunk_id": "c2",
                "doc_id": "doc_b",
                "text": "second chunk",
                "doc_title": "Title B",
                "issuer": "SEBI",
                "doc_type": "regulation",
                "page_start": None,
                "page_end": None,
                "clause_path": None,
                "section_title": "Definitions",
                "issue_date": None,
                "effective_date": None,
            },
        }
        return [rows[chunk_id] for chunk_id in chunk_ids if chunk_id in rows]


def test_retriever_satisfies_the_protocol() -> None:
    retriever = DenseRetriever(embedder=FakeEmbedderWithUsage(), store=FakeStore())  # type: ignore[arg-type]
    assert isinstance(retriever, Retriever)
    assert retriever.name == "dense"


def test_hits_are_mapped_to_chunks_in_rank_order() -> None:
    store = FakeStore()
    retriever = DenseRetriever(embedder=FakeEmbedderWithUsage(), store=store)  # type: ignore[arg-type]

    chunks, tokens, model = retriever.retrieve_with_usage("What is the LCR?", top_k=3)

    # c2 outranks c1 in the hits, and the missing row is skipped, not fabricated.
    assert [chunk.chunk_id for chunk in chunks] == ["c2", "c1"]
    assert tokens == 9
    assert model == "gemini-embedding-001"
    first = chunks[0]
    assert first.doc_title == "Title B"
    assert first.issuer == "SEBI"
    assert first.page_start is None
    assert first.section_title == "Definitions"
    assert first.score == 0.8
    assert first.dense_score == 0.8
    assert chunks[1].issue_date == date(2020, 9, 4)
    assert chunks[1].page_start == 3

    search = store.searches[0]
    assert search["vector"] == [0.25, 0.75]
    assert search["top_k"] == 3


def test_filters_reach_the_store_unchanged() -> None:
    store = FakeStore()
    retriever = DenseRetriever(embedder=FakeEmbedderWithUsage(), store=store)  # type: ignore[arg-type]
    filters = SearchFilters(issuer="RBI")
    retriever.retrieve("q", top_k=10, filters=filters)
    assert store.searches[0]["filters"] is filters


def test_embedders_without_usage_still_work_with_zero_tokens() -> None:
    retriever = DenseRetriever(embedder=FakeEmbedderWithoutUsage(), store=FakeStore())  # type: ignore[arg-type]
    chunks, tokens, model = retriever.retrieve_with_usage("q")
    assert len(chunks) == 2
    assert tokens == 0
    assert model == "other"


def test_empty_search_returns_no_chunks() -> None:
    class EmptyStore(FakeStore):
        def search(self, vector: list[float], *, top_k: int, filters=None):  # type: ignore[no-untyped-def]
            return []

    retriever = DenseRetriever(embedder=FakeEmbedderWithUsage(), store=EmptyStore())  # type: ignore[arg-type]
    chunks, tokens, _ = retriever.retrieve_with_usage("q")
    assert chunks == []
    assert tokens == 9
