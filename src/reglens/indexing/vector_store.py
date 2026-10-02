"""pgvector store: the :class:`VectorStore` protocol over asyncpg.

Design notes that matter:

* **Sync facade, async core.** The Phase 0 protocol is synchronous (the eval runner and
  the CLI are sync), while asyncpg is loop-bound. Each public call therefore opens one
  connection inside its own ``asyncio.run``. FastAPI must call the retriever through
  ``asyncio.to_thread`` — inside a worker thread there is no running loop, so this works
  there and would raise a clear error if called directly on the event loop.
* **Vectors travel as text.** ``'[0.1,0.2,...]'::vector`` avoids depending on codec
  registration per connection and keeps the SQL readable; ``repr(float)`` round-trips
  exactly, so the stored vector is bit-identical to the one the model returned.
* **Order-preserving fetch.** Retrieval ranks in SQL but the protocol returns rich
  chunks, so ``search`` gets ids+scores and ``fetch`` loads rows back in the caller's
  order — two cheap queries instead of one lossy join.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

import asyncpg

from reglens.config import Settings, get_settings
from reglens.db.pool import connect
from reglens.indexing.base import KeywordHit
from reglens.retrieval.base import SearchFilters

T = TypeVar("T")

#: Columns ``fetch`` returns; also the field list of the dicts it produces.
FETCH_SQL = """
SELECT c.chunk_id, c.doc_id, c.text, c.page_start, c.page_end, c.clause_path,
       c.section_title, c.chunk_strategy, c.token_count, c.metadata,
       d.title AS doc_title, d.issuer, d.doc_type, d.issue_date, d.effective_date
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE c.chunk_id = ANY($1::text[])
"""

SEARCH_SQL_TEMPLATE = """
SELECT c.chunk_id, 1 - (c.embedding <=> $1::vector) AS score
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE {where}
ORDER BY c.embedding <=> $1::vector
LIMIT $3
"""


def vector_literal(values: list[float]) -> str:
    """pgvector's text input, with round-trip-exact floats."""
    if not values:
        raise ValueError("cannot serialise an empty vector")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("vector contains NaN or Infinity; refusing to store it")
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def build_filter_clause(
    filters: SearchFilters | None, *, param_start: int = 4
) -> tuple[str, list[Any]]:
    """WHERE fragment for the chunk search.

    Parameter layout for the search statement: ``$1`` vector, ``$2`` corpus_version,
    ``$3`` top_k, then any filter params from ``$4`` on (``param_start``).

    Pure function so the phase where ``as_of_date`` starts filtering (FR2) is a unit
    test change, not a database experiment: a filter that only some code paths honour
    is how a superseded rule ends up cited as current.
    """
    clauses = ["c.corpus_version = $2"]
    params: list[Any] = []
    if filters is None:
        return " AND ".join(clauses), params

    def add(column: str, value: Any) -> None:
        if value is None:
            return
        index = param_start + len(params)
        params.append(value)
        clauses.append(f"{column} = ${index}")

    add("d.issuer", filters.issuer)
    add("d.doc_type", filters.doc_type)
    add("d.fiscal_period", filters.fiscal_period)
    add("d.corpus_version", filters.corpus_version)
    if filters.doc_ids:
        index = param_start + len(params)
        params.append(list(filters.doc_ids))
        clauses.append(f"d.doc_id = ANY(${index}::text[])")
    if filters.as_of_date is not None:
        index = param_start + len(params)
        params.append(filters.as_of_date)
        clauses.append(f"(d.issue_date <= ${index} OR d.issue_date IS NULL)")
    return " AND ".join(clauses), params


def hits_from_rows(rows: list[Mapping[str, Any]]) -> list[KeywordHit]:
    return [KeywordHit(chunk_id=row["chunk_id"], score=float(row["score"])) for row in rows]


def chunks_from_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


class PgVectorStore:
    """Dense retrieval and writes over the ``chunks``/``documents`` tables."""

    name = "pgvector"

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ plumbing
    def _run(self, work: Callable[[asyncpg.Connection], Awaitable[T]]) -> T:
        async def wrapped() -> T:
            connection = await connect(self.settings)
            try:
                return await work(connection)
            finally:
                await connection.close()

        try:
            return asyncio.run(wrapped())
        except RuntimeError as exc:  # pragma: no cover - only when misused from a loop
            if "asyncio.run()" in str(exc) or "event loop" in str(exc):
                raise RuntimeError(
                    "PgVectorStore is a sync facade; call it from a thread "
                    "(asyncio.to_thread in FastAPI routes), not on the event loop"
                ) from exc
            raise

    # ------------------------------------------------------------------ writes
    def upsert_documents(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0

        async def work(connection: asyncpg.Connection) -> int:
            await connection.executemany(
                """
                INSERT INTO documents (
                    doc_id, corpus_version, source, issuer, doc_type, title, issue_date,
                    effective_date, fiscal_period, url, file_hash, parse_quality, parser,
                    page_count, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (doc_id) DO UPDATE SET
                    corpus_version = EXCLUDED.corpus_version,
                    title = EXCLUDED.title,
                    issue_date = EXCLUDED.issue_date,
                    parse_quality = EXCLUDED.parse_quality,
                    parser = EXCLUDED.parser,
                    page_count = EXCLUDED.page_count,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                [
                    (
                        record["doc_id"],
                        record["corpus_version"],
                        record["source"],
                        record["issuer"],
                        record["doc_type"],
                        record["title"],
                        record.get("issue_date"),
                        record.get("effective_date"),
                        record.get("fiscal_period"),
                        record["url"],
                        record["file_hash"],
                        record.get("parse_quality"),
                        record.get("parser"),
                        record.get("page_count"),
                        json.dumps(record.get("metadata") or {}),
                    )
                    for record in records
                ],
            )
            return len(records)

        return self._run(work)

    def upsert_chunks(self, records: list[dict[str, Any]], vectors: list[list[float]]) -> int:
        if len(records) != len(vectors):
            raise ValueError(
                f"{len(records)} chunks but {len(vectors)} vectors — refusing a partial index"
            )
        if not records:
            return 0

        async def work(connection: asyncpg.Connection) -> int:
            await connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, corpus_version, chunk_index, page_start, page_end,
                    clause_path, section_title, text, token_count, chunk_strategy, metadata,
                    embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    token_count = EXCLUDED.token_count,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                [
                    (
                        record["chunk_id"],
                        record["doc_id"],
                        record["corpus_version"],
                        record["chunk_index"],
                        record.get("page_start"),
                        record.get("page_end"),
                        record.get("clause_path"),
                        record.get("section_title"),
                        record["text"],
                        record["token_count"],
                        record["chunk_strategy"],
                        json.dumps(record.get("metadata") or {}),
                        vector_literal(vector),
                    )
                    for record, vector in zip(records, vectors, strict=True)
                ],
            )
            return len(records)

        return self._run(work)

    # ------------------------------------------------------------------ reads
    def search(
        self, vector: list[float], *, top_k: int, filters: SearchFilters | None = None
    ) -> list[KeywordHit]:
        where, params = build_filter_clause(filters)
        sql = SEARCH_SQL_TEMPLATE.format(where=where)

        async def work(connection: asyncpg.Connection) -> list[Mapping[str, Any]]:
            return await connection.fetch(
                sql, vector_literal(vector), self.settings.corpus_version, top_k, *params
            )

        return hits_from_rows(self._run(work))

    def fetch(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []

        async def work(connection: asyncpg.Connection) -> list[asyncpg.Record]:
            return await connection.fetch(FETCH_SQL, chunk_ids)

        rows = chunks_from_rows(self._run(work))
        by_id = {row["chunk_id"]: row for row in rows}
        # The database returns arbitrary order; retrieval order lives in the ids.
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def count_chunks(self) -> int:
        """Chunks under the active corpus version (eval's empty-index preflight)."""

        async def work(connection: asyncpg.Connection) -> int:
            row = await connection.fetchrow(
                "SELECT count(*) FROM chunks WHERE corpus_version = $1",
                self.settings.corpus_version,
            )
            return int(row[0])

        return self._run(work)

    # ------------------------------------------------------------------ idempotency
    def has_document(self, file_hash: str, *, chunk_strategy: str) -> bool:
        """True when this exact byte-set is already parsed, chunked and embedded."""

        async def work(connection: asyncpg.Connection) -> bool:
            row = await connection.fetchrow(
                """
                SELECT EXISTS (
                    SELECT 1 FROM documents d
                    WHERE d.file_hash = $1 AND d.corpus_version = $2
                      AND EXISTS (
                          SELECT 1 FROM chunks c
                          WHERE c.doc_id = d.doc_id AND c.chunk_strategy = $3
                      )
                )
                """,
                file_hash,
                self.settings.corpus_version,
                chunk_strategy,
            )
            return bool(row[0])

        return self._run(work)
