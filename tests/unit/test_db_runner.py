"""Migration-runner tests (no database needed).

Migrations are immutable once applied, so the checksum and placeholder behaviour is tested
directly: a changed file must be detected, and an unknown ``${PLACEHOLDER}`` must fail
rather than reach Postgres as literal text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.config import get_settings
from reglens.db.runner import MigrationError, discover_migrations, render_sql


def write_migration(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_migrations_are_discovered_in_filename_order(tmp_path: Path) -> None:
    write_migration(tmp_path, "0002_b.sql", "SELECT 2;")
    write_migration(tmp_path, "0001_a.sql", "SELECT 1;")
    names = [migration.filename for migration in discover_migrations(tmp_path)]
    assert names == ["0001_a.sql", "0002_b.sql"]


def test_embedding_dim_placeholder_is_substituted(tmp_path: Path) -> None:
    write_migration(tmp_path, "0001_v.sql", "CREATE TABLE t (v vector(${EMBEDDING_DIM}));")
    (migration,) = discover_migrations(tmp_path)
    # Assert against the configured dimension, not a copy of it: the embedding model is
    # a config choice and hard-coding the number here is how the two drift apart.
    expected_dim = get_settings().embedding_dim
    assert f"vector({expected_dim})" in migration.sql
    assert "${" not in migration.sql


def test_unknown_placeholder_fails_loudly() -> None:
    with pytest.raises(MigrationError, match="unknown placeholder"):
        render_sql("SELECT ${NOT_A_SETTING};", {"EMBEDDING_DIM": "1536"})


def test_checksum_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    path = write_migration(tmp_path, "0001_a.sql", "SELECT 1;")
    first = discover_migrations(tmp_path)[0].checksum
    assert discover_migrations(tmp_path)[0].checksum == first
    path.write_text("SELECT 2;", encoding="utf-8")
    assert discover_migrations(tmp_path)[0].checksum != first


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="not found"):
        discover_migrations(tmp_path / "nope")


def test_core_migration_declares_the_required_tables() -> None:
    """Guards against a migration that silently loses a table the product depends on."""
    migrations = discover_migrations()
    assert migrations, "no migrations found"
    sql = "\n".join(migration.sql for migration in migrations).lower()
    for table in (
        "documents",
        "document_links",
        "chunks",
        "metrics",
        "queries",
        "feedback",
        "request_traces",
        "request_stages",
    ):
        assert f"create table if not exists {table}" in sql, table


def test_core_migration_keeps_the_idempotency_constraints() -> None:
    sql = "\n".join(migration.sql for migration in discover_migrations()).lower()
    assert "documents_file_hash_key" in sql
    assert "chunks_identity_key" in sql
    assert "create extension if not exists vector" in sql
    assert "generated always as (to_tsvector" in sql


def test_chunks_table_has_no_embedding_model_column() -> None:
    """Embedding model identity lives in the experiment config and corpus version, not in
    a per-row column, so an ablation cannot be compared against mixed vectors by accident."""
    sql = "\n".join(migration.sql for migration in discover_migrations())
    assert "embedding_model" not in sql.split("CREATE TABLE IF NOT EXISTS chunks")[1].split(");")[0]
