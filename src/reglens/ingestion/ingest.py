"""Ingest: parse → chunk → embed → store, idempotent by file hash (FR5).

One document per transaction boundary: if a later document fails, everything already
committed stays committed and the report says which document failed — a 20-document
corpus must not die at document 19 with nothing to show.

Cost is measured, not guessed: every embed call's token count is priced from the same
table the eval reports use, and the totals land in the run report.

Parsed output is also cached under ``data/parsed/`` (gitignored) so re-chunking
experiments do not re-parse PDFs, and so a human can read exactly what the chunker saw.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from reglens.chunking.base import Chunk, Chunker, ParsedDocument
from reglens.chunking.fixed import FixedSizeChunker
from reglens.config import Settings, get_settings, load_experiment
from reglens.config.experiment import ExperimentConfig
from reglens.indexing.base import EmbeddingModel, VectorStore
from reglens.ingestion.manifest import DocumentRecord, load_manifest, raw_path_for
from reglens.ingestion.parse import parse_pdf
from reglens.observability.cost import try_estimate_cost
from reglens.observability.logging import get_logger

logger = get_logger("reglens.ingest")


@dataclass
class IngestReport:
    """What one ingest run actually did, for the CLI and for the phase report."""

    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunk_count: int = 0
    token_count: int = 0
    embed_input_tokens: int = 0
    embed_cost_usd: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "documents": len(self.processed),
            "skipped_count": len(self.skipped),
            "failed_count": len(self.failed),
            "chunks": self.chunk_count,
            "tokens": self.token_count,
            "embed_input_tokens": self.embed_input_tokens,
            "embed_cost_usd": round(self.embed_cost_usd, 6),
            "seconds": round(self.seconds, 3),
        }


def build_chunker(experiment: ExperimentConfig, *, tokenizer: Any | None = None) -> Chunker:
    """Chunker for the experiment config (Phase 1: fixed windows only)."""
    if experiment.chunking.strategy != "fixed":
        raise NotImplementedError(
            f"chunk strategy {experiment.chunking.strategy!r} ships in Phase 2; "
            "the Phase 1 baseline measures fixed windows"
        )
    return FixedSizeChunker(
        size=experiment.chunking.chunk_size_tokens,
        overlap=experiment.chunking.overlap_tokens,
        min_tokens=experiment.chunking.min_chunk_tokens,
        tokenizer=tokenizer,
    )


def normalize(vector: list[float]) -> list[float]:
    """L2-normalise when the experiment asks for it (cosine is scale-invariant, but
    normalised vectors make the stored numbers meaningful on their own)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("zero vector from the embedding model; refusing to store it")
    return [value / norm for value in vector]


def _as_date(value: str) -> date | None:
    """Manifest dates are ``YYYY-MM-DD`` strings; asyncpg needs real ``date`` objects."""
    return date.fromisoformat(value) if value else None


def document_row(
    record: DocumentRecord, parsed: ParsedDocument, corpus_version: str
) -> dict[str, Any]:
    return {
        "doc_id": record.doc_id,
        "corpus_version": corpus_version,
        "source": record.source,
        "issuer": record.issuer,
        "doc_type": record.doc_type,
        "title": record.title,
        "issue_date": _as_date(record.issue_date),
        "effective_date": _as_date(record.effective_date),
        "fiscal_period": record.fiscal_period,
        "url": record.url,
        "file_hash": record.file_hash,
        "parse_quality": parsed.parse_quality,
        "parser": parsed.parser,
        "page_count": parsed.page_count,
        "metadata": {"scan_pages": parsed.metadata.get("scan_pages", [])},
    }


def chunk_rows(chunks: list[Chunk], corpus_version: str) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "corpus_version": corpus_version,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "clause_path": chunk.clause_path,
            "section_title": chunk.section_title,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "chunk_strategy": chunk.strategy,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def cache_parsed(parsed: ParsedDocument, file_hash: str, settings: Settings) -> Path:
    """Write the parser's view of the document to data/parsed/ (gitignored)."""
    target = settings.parsed_dir / f"{parsed.doc_id}__{file_hash[:12]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "doc_id": parsed.doc_id,
                "parser": parsed.parser,
                "page_count": parsed.page_count,
                "parse_quality": parsed.parse_quality,
                "scan_pages": parsed.metadata.get("scan_pages", []),
                "pages": [
                    {"page_number": page.page_number, "text": page.text} for page in parsed.pages
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return target


def embed_chunks(
    chunks: list[Chunk],
    *,
    embedder: EmbeddingModel,
    batch_size: int,
    report: IngestReport,
    normalize_embeddings: bool,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        result = embedder.embed_documents([chunk.text for chunk in batch])
        report.embed_input_tokens += result.input_tokens
        report.embed_cost_usd += try_estimate_cost(result.model, result.input_tokens) or 0.0
        for vector in result.vectors:
            vectors.append(normalize(vector) if normalize_embeddings else vector)
    return vectors


def ingest_corpus(
    *,
    settings: Settings | None = None,
    experiment: ExperimentConfig | None = None,
    embedder: EmbeddingModel | None = None,
    store: VectorStore | None = None,
    chunker: Chunker | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
) -> IngestReport:
    """Run the Phase 1 pipeline over every fetched, direct-discovery manifest row."""
    from reglens.indexing.embeddings import build_embedding_model
    from reglens.indexing.vector_store import PgVectorStore

    active = settings or get_settings()
    config = experiment or load_experiment(active.experiment_config)
    model = embedder or build_embedding_model(active)
    target = store or PgVectorStore(settings=active)
    windows = chunker or build_chunker(config)

    records = [
        record
        for record in load_manifest()
        if record.status == "fetched"
        and record.discovery == "direct"
        and record.file_hash
        and (only is None or record.doc_id in only)
    ]
    if limit is not None:
        records = records[:limit]

    report = IngestReport()
    started = time.perf_counter()
    for record in records:
        path = raw_path_for(record)
        if path is None or not path.is_file():
            report.failed.append((record.doc_id, "raw file missing"))
            continue
        try:
            if target.has_document(record.file_hash, chunk_strategy=windows.name):
                report.skipped.append(record.doc_id)
                continue
            parsed = parse_pdf(
                path,
                record.doc_id,
                parser=config.parsing.parser,
                min_chars_per_page=config.parsing.min_chars_per_page,
            )
            cache_parsed(parsed, record.file_hash, active)
            chunks = windows.chunk(parsed)
            if not chunks:
                report.failed.append((record.doc_id, "chunker produced no chunks"))
                continue
            vectors = embed_chunks(
                chunks,
                embedder=model,
                batch_size=config.indexing.batch_size,
                report=report,
                normalize_embeddings=config.indexing.normalize_embeddings,
            )
            target.upsert_documents([document_row(record, parsed, active.corpus_version)])
            target.upsert_chunks(chunk_rows(chunks, active.corpus_version), vectors)
        except Exception as exc:  # one bad document must not kill the whole run
            logger.warning("ingest failed", extra={"doc_id": record.doc_id, "error": str(exc)})
            report.failed.append((record.doc_id, str(exc)))
            continue
        report.processed.append(record.doc_id)
        report.chunk_count += len(chunks)
        report.token_count += sum(chunk.token_count for chunk in chunks)

    report.seconds = time.perf_counter() - started
    return report
