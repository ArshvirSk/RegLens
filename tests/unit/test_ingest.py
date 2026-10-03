"""Ingest pipeline: parse → chunk → embed → store, with fakes where money and the
database would otherwise be involved.

The fake embedder returns a deliberately un-normalised vector so the test can prove
``normalize_embeddings`` actually ran, and prices every token through the real cost
table so the report's dollar figure is checked against the published rate.
"""

from __future__ import annotations

import pymupdf
import pytest

from reglens.chunking.fixed import FixedSizeChunker
from reglens.cli import EXIT_ERROR, main
from reglens.config import get_settings, load_experiment
from reglens.indexing.base import EmbeddedBatch
from reglens.ingestion.ingest import (
    EmbeddingDimensionError,
    build_chunker,
    chunk_rows,
    ingest_corpus,
)
from reglens.ingestion.manifest import raw_relative_path, sha256_bytes, write_manifest
from reglens.ingestion.parse import parse_pdf
from tests.conftest import make_record


class CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class FakeEmbedder:
    name = "gemini-embedding-001"

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> EmbeddedBatch:
        self.batch_sizes.append(len(texts))
        return EmbeddedBatch(
            vectors=[[3.0, 0.0, 0.0, 0.0][: self.dim] for _ in texts],
            model=self.name,
            input_tokens=7 * len(texts),
            dimensions=self.dim,
        )

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0][: self.dim]


class FakeStore:
    name = "fake"

    def __init__(self, *, already_indexed: bool = False) -> None:
        self.already_indexed = already_indexed
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.vectors: list[list[float]] = []

    def has_document(self, file_hash: str, *, chunk_strategy: str) -> bool:
        return self.already_indexed

    def upsert_documents(self, records: list[dict]) -> int:
        self.documents.extend(records)
        return len(records)

    def upsert_chunks(self, records: list[dict], vectors: list[list[float]]) -> int:
        assert len(records) == len(vectors)
        self.chunks.extend(records)
        self.vectors.extend(vectors)
        return len(records)


class DimensionStore(FakeStore):
    """Fake that reports a column width, the way PgVectorStore reads it from Postgres."""

    def __init__(self, *, dimension: int | None) -> None:
        super().__init__()
        self.dimension = dimension

    def embedding_dimension(self) -> int | None:
        return self.dimension


def make_fetchable_pdf() -> bytes:
    document = pymupdf.open()
    for index in range(2):
        page = document.new_page()
        page.insert_text((72, 72), "Ingest fixture text for the Phase 1 pipeline. " * 8)
        page.insert_text((72, 120), f"page marker {index + 1}")
    payload = document.tobytes()
    document.close()
    return payload


def write_fetched_row(payload: bytes):  # type: ignore[no-untyped-def]
    settings = get_settings()
    digest = sha256_bytes(payload)
    record = make_record(
        status="fetched",
        file_hash=digest,
        issue_date="2020-09-04",
        effective_date="2020-09-04",
    )
    record = record.model_copy(update={"local_path": raw_relative_path(record, digest)})
    target = settings.raw_dir / record.local_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    write_manifest([record], settings.manifest_path)
    return record


def small_chunker() -> FixedSizeChunker:
    return FixedSizeChunker(size=128, overlap=16, min_tokens=8, tokenizer=CharTokenizer())


def test_ingest_parses_chunks_embeds_and_stores() -> None:
    settings = get_settings()
    record = write_fetched_row(make_fetchable_pdf())
    embedder = FakeEmbedder()
    store = FakeStore()

    report = ingest_corpus(
        settings=settings,
        experiment=load_experiment("baseline_naive"),
        embedder=embedder,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        chunker=small_chunker(),
    )

    assert report.processed == [record.doc_id]
    assert report.failed == []
    assert report.chunk_count == len(store.chunks) > 0
    assert report.token_count == sum(chunk["token_count"] for chunk in store.chunks)
    # Every chunk embedded exactly once, in batches bounded by the experiment config.
    assert sum(embedder.batch_sizes) == report.chunk_count
    assert all(size <= 64 for size in embedder.batch_sizes)
    # Token accounting priced at the published gemini-embedding-001 rate ($0.15/1M).
    assert report.embed_input_tokens == 7 * report.chunk_count
    assert report.embed_cost_usd == pytest.approx(report.embed_input_tokens * 0.15 / 1_000_000)

    # normalize_embeddings=true in the baseline: [3,0,0,0] must arrive as [1,0,0,0].
    assert store.vectors[0] == [1.0, 0.0, 0.0, 0.0]
    # Parsed output cached for reuse (data/parsed is gitignored).
    assert list(settings.parsed_dir.glob(f"{record.doc_id}__*.json"))

    document = store.documents[0]
    assert document["doc_id"] == record.doc_id
    assert document["issue_date"].isoformat() == "2020-09-04"
    assert document["parser"] == "pymupdf"
    assert document["page_count"] == 2


def test_ingest_skips_documents_already_indexed() -> None:
    settings = get_settings()
    write_fetched_row(make_fetchable_pdf())
    embedder = FakeEmbedder()
    report = ingest_corpus(
        settings=settings,
        experiment=load_experiment("baseline_naive"),
        embedder=embedder,  # type: ignore[arg-type]
        store=FakeStore(already_indexed=True),  # type: ignore[arg-type]
        chunker=small_chunker(),
    )

    assert report.skipped == ["rbi_md_test"]
    assert report.processed == []
    assert embedder.batch_sizes == []  # no paid call happened


def test_ingest_refuses_before_paying_when_store_dimension_mismatches() -> None:
    settings = get_settings()
    write_fetched_row(make_fetchable_pdf())
    embedder = FakeEmbedder()
    wrong = settings.embedding_dim + 1

    with pytest.raises(EmbeddingDimensionError, match=f"vector\\({wrong}\\)"):
        ingest_corpus(
            settings=settings,
            experiment=load_experiment("baseline_naive"),
            embedder=embedder,  # type: ignore[arg-type]
            store=DimensionStore(dimension=wrong),  # type: ignore[arg-type]
            chunker=small_chunker(),
        )

    assert embedder.batch_sizes == []  # the preflight ran before any paid call


def test_ingest_proceeds_when_store_dimension_matches_settings() -> None:
    settings = get_settings()
    record = write_fetched_row(make_fetchable_pdf())

    report = ingest_corpus(
        settings=settings,
        experiment=load_experiment("baseline_naive"),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        store=DimensionStore(dimension=settings.embedding_dim),  # type: ignore[arg-type]
        chunker=small_chunker(),
    )

    assert report.processed == [record.doc_id]
    assert report.failed == []


def test_failed_documents_are_reported_not_raised() -> None:
    settings = get_settings()
    record = write_fetched_row(make_fetchable_pdf())
    # Delete the raw bytes behind the manifest's back.
    (settings.raw_dir / record.local_path).unlink()

    report = ingest_corpus(
        settings=settings,
        experiment=load_experiment("baseline_naive"),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        store=FakeStore(),  # type: ignore[arg-type]
        chunker=small_chunker(),
    )

    assert report.processed == []
    assert report.failed == [(record.doc_id, "raw file missing")]


def test_chunk_rows_carry_every_column_the_schema_expects() -> None:
    settings = get_settings()
    record = write_fetched_row(make_fetchable_pdf())
    parsed = parse_pdf(settings.raw_dir / record.local_path, record.doc_id)
    chunks = small_chunker().chunk(parsed)
    rows = chunk_rows(chunks, "0.1.0")
    assert rows
    for row, chunk in zip(rows, chunks, strict=True):
        assert row["chunk_id"] == chunk.chunk_id
        assert row["corpus_version"] == "0.1.0"
        assert row["chunk_strategy"] == "fixed"
        assert {"page_start", "page_end", "text", "token_count"} <= set(row)


def test_build_chunker_refuses_phase_2_strategies() -> None:
    experiment = load_experiment("baseline_naive")
    assert build_chunker(experiment).name == "fixed"
    modified = experiment.model_copy(deep=True)
    modified.chunking.strategy = "clause_aware"
    with pytest.raises(NotImplementedError, match="Phase 2"):
        build_chunker(modified)


def test_cli_ingest_reports_dimension_mismatch_as_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_fetched_row(make_fetchable_pdf())

    def mismatched(**_kwargs: object) -> None:
        raise EmbeddingDimensionError(
            "chunks.embedding is vector(1) but settings expect vector(2)"
        )

    monkeypatch.setattr("reglens.ingestion.ingest.ingest_corpus", mismatched)
    assert main(["ingest", "--yes"]) == EXIT_ERROR
    assert "cannot ingest" in capsys.readouterr().err
