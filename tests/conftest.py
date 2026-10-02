"""Shared pytest fixtures.

Tests never touch the real corpus, the network, or the developer's ``.env``: every fixture
builds a temporary world. That keeps ``make test`` runnable on a fresh clone with no
Docker, no keys, and no downloaded PDFs.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import pytest

from reglens.config import get_settings
from reglens.ingestion.manifest import MANIFEST_COLUMNS, DocumentRecord, write_manifest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point data paths at a temp dir and drop any inherited secrets."""
    monkeypatch.setenv("REGLENS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REGLENS_RAW_DIR", str(tmp_path / "data" / "raw"))
    monkeypatch.setenv("REGLENS_PARSED_DIR", str(tmp_path / "data" / "parsed"))
    monkeypatch.setenv("REGLENS_DERIVED_DIR", str(tmp_path / "data" / "derived"))
    # Point the env-file loader at a path that does not exist: the project's real .env
    # (providers, database URL, keys) must never shape an assertion.
    monkeypatch.setenv("REGLENS_ENV_FILE", str(tmp_path / "no.env"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """A temp manifest that tests may safely overwrite.

    This must live under the temp data directory: a fixture pointing at the project's real
    ``data/manifest.csv`` once let a test clobber the reviewed corpus plan's output.
    """
    return tmp_path / "data" / "manifest.csv"


def make_record(**overrides: object) -> DocumentRecord:
    """A valid manifest row; override any field to test a specific rule."""
    payload: dict[str, object] = {
        "doc_id": "rbi_md_test",
        "source": "rbi",
        "issuer": "RBI",
        "doc_type": "master_direction",
        "title": "Master Direction - Test",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/164MD.PDF",
        "discovery": "direct",
        "status": "planned",
    }
    payload.update(overrides)
    return DocumentRecord.model_validate(payload)


@pytest.fixture
def records(manifest_path: Path) -> list[DocumentRecord]:
    """Two valid rows written to a temp manifest, one fetched with real bytes on disk."""
    raw = get_settings().raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    payload = b"%PDF-1.6 fake pdf for hashing"
    from reglens.ingestion.manifest import raw_relative_path, sha256_bytes

    digest = sha256_bytes(payload)
    first = make_record(
        doc_id="rbi_md_test",
        status="fetched",
        file_hash=digest,
        issue_date="2020-09-04",
        effective_date="2020-09-04",
    )
    first = first.model_copy(update={"local_path": raw_relative_path(first, digest)})
    (raw / first.local_path).parent.mkdir(parents=True, exist_ok=True)
    (raw / first.local_path).write_bytes(payload)
    second = make_record(
        doc_id="sebi_mc_test",
        source="sebi",
        issuer="SEBI",
        doc_type="master_circular",
        title="Master Circular - Test",
        url="https://www.sebi.gov.in/legal/master-circulars/jul-2025/example_95221.html",
    )
    rows = [first, second]
    write_manifest(rows, manifest_path)
    return rows


@pytest.fixture
def csv_text(records: list[DocumentRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_row())
    return buffer.getvalue()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove REGLENS_* variables so a test can assert true defaults."""
    for key in list(os.environ):
        if key.startswith("REGLENS_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
