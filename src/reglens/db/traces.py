"""Trace persistence (FR6): one row per request plus one row per pipeline stage."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:  # pragma: no cover - avoids a circular import at runtime
    from reglens.observability.tracing import TraceRecord

INSERT_TRACE_SQL = """
INSERT INTO request_traces (
    trace_id, query_id, name, route, model, corpus_version, config_hash,
    started_at, finished_at, latency_ms, input_tokens, output_tokens, cost_usd, error, metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
)
ON CONFLICT (trace_id) DO NOTHING
"""

INSERT_STAGE_SQL = """
INSERT INTO request_stages (
    trace_id, name, seq, latency_ms, model, input_tokens, output_tokens, cost_usd, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""


def _json(value: Any) -> str:
    return json.dumps(value or {}, default=str)


async def insert_trace(pool: asyncpg.Pool, record: TraceRecord) -> None:
    """Write a trace and its stages. Never raises: tracing must not break a request."""
    try:
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                INSERT_TRACE_SQL,
                record.trace_id,
                record.query_id,
                record.name,
                record.route,
                record.model,
                record.corpus_version,
                record.config_hash,
                record.started_at,
                record.finished_at,
                record.latency_ms,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.error,
                _json(record.metadata),
            )
            for stage in record.stages:
                await connection.execute(
                    INSERT_STAGE_SQL,
                    record.trace_id,
                    stage.name,
                    stage.seq,
                    stage.latency_ms,
                    stage.model,
                    stage.input_tokens,
                    stage.output_tokens,
                    stage.cost_usd,
                    _json(stage.metadata),
                )
    except (asyncpg.PostgresError, OSError):
        return


async def fetch_recent_traces(pool: asyncpg.Pool, *, limit: int = 50) -> list[dict[str, Any]]:
    """Most recent traces, newest first, with the columns the Phase 4 dashboard needs."""
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT trace_id, name, route, latency_ms, input_tokens, output_tokens,
                   cost_usd, error, started_at
            FROM request_traces
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [dict(row) for row in rows]


async def latency_percentiles(pool: asyncpg.Pool) -> dict[str, float | None]:
    """p50/p95 latency and mean cost, the two numbers the Phase 4 dashboard leads with."""
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT
                percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)  AS p50_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
                avg(cost_usd)                                            AS mean_cost_usd,
                count(*)                                                 AS requests
            FROM request_traces
            WHERE error IS NULL
            """
        )
    return {
        key: (float(value) if value is not None else None) for key, value in dict(row or {}).items()
    }
