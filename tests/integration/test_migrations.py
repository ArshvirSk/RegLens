"""Integration tests that need a real Postgres (``make test-integration``).

Skipped automatically when the database is unreachable, so ``make test`` stays runnable on
a fresh clone. What is asserted here is what a mock cannot prove: pgvector exists, the
generated ``tsvector`` column works, idempotency constraints really reject duplicates, and
migrations are recorded with checksums.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from reglens.config import get_settings
from reglens.db.pool import connect, health
from reglens.db.runner import apply_migrations, migration_status

pytestmark = pytest.mark.integration


@pytest.fixture
async def connection():
    try:
        conn = await connect()
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"postgres unavailable: {exc}")
    try:
        yield conn
    finally:
        await conn.close()


async def test_migrations_apply_and_are_idempotent(connection: asyncpg.Connection) -> None:
    await apply_migrations(connection)
    status = await migration_status(connection)
    assert status["pending"] == []
    assert status["changed"] == []
    assert "0001_core.sql" in status["applied"]

    # Running again must apply nothing.
    assert await apply_migrations(connection) == []


async def test_pgvector_and_full_text_search_are_available(connection: asyncpg.Connection) -> None:
    version = await connection.fetchval(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    assert version, "pgvector extension is not installed"
    ranked = await connection.fetchval(
        "SELECT to_tsvector('english', 'liquidity coverage ratio') @@ plainto_tsquery('english', 'liquidity')"
    )
    assert ranked is True


async def test_document_hash_constraint_prevents_duplicates(connection: asyncpg.Connection) -> None:
    await connection.execute("DELETE FROM documents WHERE doc_id LIKE 'it_%'")
    insert = """
        INSERT INTO documents (doc_id, corpus_version, source, issuer, doc_type, title, url, file_hash)
        VALUES ($1, 'test', 'rbi', 'RBI', 'master_direction', 'IT doc', 'https://rbi.org.in/x', $2)
    """
    await connection.execute(insert, "it_doc_a", "f" * 64)
    with pytest.raises(asyncpg.UniqueViolationError):
        await connection.execute(insert, "it_doc_b", "f" * 64)
    await connection.execute("DELETE FROM documents WHERE doc_id LIKE 'it_%'")


async def test_chunk_identity_constraint_prevents_duplicates(
    connection: asyncpg.Connection,
) -> None:
    await connection.execute("DELETE FROM documents WHERE doc_id = 'it_doc_chunks'")
    await connection.execute(
        """
        INSERT INTO documents (doc_id, corpus_version, source, issuer, doc_type, title, url, file_hash)
        VALUES ('it_doc_chunks', 'test', 'rbi', 'RBI', 'master_direction', 'IT chunks',
                'https://rbi.org.in/x', $1)
        """,
        "e" * 64,
    )
    insert = """
        INSERT INTO chunks (chunk_id, doc_id, corpus_version, chunk_index, text, token_count, chunk_strategy)
        VALUES ($1, 'it_doc_chunks', 'test', 0, 'clause text', 2, 'fixed')
    """
    await connection.execute(insert, "it_chunk_1")
    with pytest.raises(asyncpg.UniqueViolationError):
        await connection.execute(insert, "it_chunk_2")

    # The generated tsvector column is populated without an explicit write.
    tsv = await connection.fetchval("SELECT tsv::text FROM chunks WHERE chunk_id = 'it_chunk_1'")
    assert "claus" in tsv

    await connection.execute("DELETE FROM documents WHERE doc_id = 'it_doc_chunks'")


async def test_vector_column_accepts_configured_dimensions(connection: asyncpg.Connection) -> None:
    dim = get_settings().embedding_dim
    value = "[" + ",".join(["0.1"] * dim) + "]"
    distance = await connection.fetchval("SELECT $1::vector <=> $1::vector", value)
    assert distance == pytest.approx(0.0)


async def test_readonly_role_is_configured() -> None:
    """The Phase 3 text-to-SQL tool depends on this role existing and being read-only."""
    settings = get_settings()
    try:
        conn = await asyncpg.connect(dsn=settings.readonly_database_url)
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"read-only role unavailable: {exc}")
    try:
        with pytest.raises(asyncpg.PostgresError):
            await conn.execute("CREATE TABLE should_not_exist (x int)")
        assert await conn.fetchval("SELECT 1") == 1
    finally:
        await conn.close()


async def test_health_reports_counts(connection: asyncpg.Connection) -> None:
    pool = await asyncpg.create_pool(dsn=get_settings().database_url, min_size=1, max_size=2)
    try:
        report = await health(pool)
    finally:
        await pool.close()
    assert report["status"] == "ok"
    assert report["pgvector_version"] != "missing"
    assert isinstance(report["documents"], int)


async def test_verbose_migration_logging_does_not_crash(
    connection: asyncpg.Connection, tmp_path: Path
) -> None:
    """Regression: ``logger.info(extra={"filename": ...})`` raises KeyError from logging,
    which crashed the migrator *after* the schema change had already committed. A half-run
    migration with a traceback is the worst possible failure mode, so the verbose path is
    exercised for real."""
    (tmp_path / "9000_tmp_regression.sql").write_text(
        "CREATE TABLE it_verbose_logging (x int);", encoding="utf-8"
    )
    applied = await apply_migrations(connection, migrations_dir=tmp_path, verbose=True)
    assert applied == ["9000_tmp_regression.sql"]
    assert await apply_migrations(connection, migrations_dir=tmp_path, verbose=True) == []

    await connection.execute("DROP TABLE it_verbose_logging")
    await connection.execute(
        "DELETE FROM schema_migrations WHERE filename = $1", "9000_tmp_regression.sql"
    )


async def test_migrations_are_immutable_once_applied(
    connection: asyncpg.Connection, tmp_path: Path
) -> None:
    """Editing an applied migration must abort, not silently diverge from the schema."""
    from reglens.db.runner import MigrationError, apply_migrations

    path = tmp_path / "9001_immutable.sql"
    path.write_text("SELECT 1;", encoding="utf-8")
    await apply_migrations(connection, migrations_dir=tmp_path)

    path.write_text("SELECT 2;", encoding="utf-8")
    with pytest.raises(MigrationError, match="changed after it was applied"):
        await apply_migrations(connection, migrations_dir=tmp_path)

    await connection.execute(
        "DELETE FROM schema_migrations WHERE filename = $1", "9001_immutable.sql"
    )
