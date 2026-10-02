"""Minimal SQL migration runner.

Chosen over Alembic deliberately: the schema is the interesting part of this project
(vectors, tsvector, supersession links), and plain numbered ``.sql`` files keep it
readable and reviewable. What the runner must guarantee, and what a naive
``psql -f`` loop does not:

1. Each migration runs in a transaction, exactly once, tracked in ``schema_migrations``.
2. Applied migrations are immutable: a changed checksum aborts startup instead of
   silently diverging from whatever the database actually has.
3. ``${EMBEDDING_DIM}`` placeholders are substituted from typed settings, because
   pgvector needs a concrete dimension.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from reglens.config import get_settings
from reglens.observability.logging import get_logger, safe_extra

logger = get_logger("reglens.migrate")

PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Migration:
    filename: str
    path: Path
    checksum: str
    sql: str


class MigrationError(RuntimeError):
    """Raised when a migration cannot be applied safely."""


def discover_migrations(migrations_dir: Path | None = None) -> list[Migration]:
    """All migrations, ordered by filename, with rendered placeholders and checksums."""
    directory = migrations_dir or get_settings().migrations_dir
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")
    settings = get_settings()
    values = {
        "EMBEDDING_DIM": str(settings.embedding_dim),
        "CORPUS_VERSION": settings.corpus_version,
    }
    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        raw = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                filename=path.name,
                path=path,
                checksum=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                sql=render_sql(raw, values),
            )
        )
    return migrations


def render_sql(sql: str, values: dict[str, str]) -> str:
    """Substitute ``${NAME}`` placeholders, refusing to leave unknown ones behind."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise MigrationError(f"unknown placeholder ${{{key}}} in migration SQL")
        return values[key]

    return PLACEHOLDER_RE.sub(replace, sql)


async def applied_migrations(connection: asyncpg.Connection) -> dict[str, str]:
    rows = await connection.fetch("SELECT filename, checksum FROM schema_migrations")
    return {row["filename"]: row["checksum"] for row in rows}


async def migration_status(
    connection: asyncpg.Connection, migrations_dir: Path | None = None
) -> dict[str, object]:
    """Pending vs applied migrations, used by ``reglens status`` and CI logs."""
    await ensure_bootstrap(connection)
    applied = await applied_migrations(connection)
    migrations = discover_migrations(migrations_dir)
    pending = [m.filename for m in migrations if m.filename not in applied]
    changed = [
        m.filename
        for m in migrations
        if m.filename in applied and applied[m.filename] != m.checksum
    ]
    return {"applied": sorted(applied), "pending": pending, "changed": changed}


async def ensure_bootstrap(connection: asyncpg.Connection) -> None:
    await connection.execute(BOOTSTRAP_SQL)


async def apply_migrations(
    connection: asyncpg.Connection,
    *,
    migrations_dir: Path | None = None,
    verbose: bool = False,
) -> list[str]:
    """Apply pending migrations; return the filenames that were applied."""
    await ensure_bootstrap(connection)
    applied = await applied_migrations(connection)
    newly_applied: list[str] = []
    for migration in discover_migrations(migrations_dir):
        recorded = applied.get(migration.filename)
        if recorded is not None:
            if recorded != migration.checksum:
                raise MigrationError(
                    f"{migration.filename} changed after it was applied "
                    f"(recorded {recorded[:12]}…, file {migration.checksum[:12]}…). "
                    "Migrations are immutable: add a new file instead."
                )
            continue
        async with connection.transaction():
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (filename, checksum) VALUES ($1, $2)",
                migration.filename,
                migration.checksum,
            )
        newly_applied.append(migration.filename)
        if verbose:
            logger.info(
                "applied migration",
                extra=safe_extra(event="migration_applied", filename=migration.filename),
            )
    return newly_applied
