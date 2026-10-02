"""asyncpg pool management.

Raw SQL over asyncpg rather than an ORM: the interesting queries here are vector
similarity, ``tsvector`` ranking and window functions over ``metrics``, which an ORM
would only obscure. The trade-off is that SQL lives in strings and must be reviewed and
tested like code (Phase 3 adds a schema allow-list and a read-only role for that reason).
"""

from __future__ import annotations

import asyncpg

from reglens.config import Settings, get_settings


async def create_pool(settings: Settings | None = None) -> asyncpg.Pool:
    """Create the application pool. Callers own closing it."""
    active = settings or get_settings()
    return await asyncpg.create_pool(
        dsn=active.database_url,
        min_size=active.db_pool_min_size,
        max_size=active.db_pool_max_size,
        command_timeout=active.db_command_timeout_seconds,
        server_settings={"application_name": "reglens"},
    )


async def close_pool(pool: asyncpg.Pool | None) -> None:
    if pool is not None and not pool._closed:
        await pool.close()


async def connect(
    settings: Settings | None = None, *, readonly: bool = False
) -> asyncpg.Connection:
    """Single connection for scripts and migrations."""
    active = settings or get_settings()
    dsn = active.readonly_database_url if readonly else active.database_url
    return await asyncpg.connect(dsn=dsn, server_settings={"application_name": "reglens-conn"})


async def health(pool: asyncpg.Pool) -> dict[str, object]:
    """Cheap readiness probe: can we query, and is pgvector present in the version we expect?"""
    try:
        async with pool.acquire() as connection:
            version = await connection.fetchval("SHOW server_version")
            vector_ext = await connection.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            document_count = await connection.fetchval(
                "SELECT count(*) FROM documents"
                if await connection.fetchval("SELECT to_regclass('public.documents') IS NOT NULL")
                else "SELECT 0"
            )
            chunk_count = await connection.fetchval(
                "SELECT count(*) FROM chunks"
                if await connection.fetchval("SELECT to_regclass('public.chunks') IS NOT NULL")
                else "SELECT 0"
            )
    except (asyncpg.PostgresError, OSError) as exc:
        return {"status": "unavailable", "detail": str(exc)}
    return {
        "status": "ok",
        "server_version": version,
        "pgvector_version": vector_ext or "missing",
        "documents": int(document_count or 0),
        "chunks": int(chunk_count or 0),
    }
