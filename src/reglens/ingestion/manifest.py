"""The corpus manifest: one row per source document, with hashes.

Why a CSV manifest instead of a database table? Because the manifest is the *input* to
ingestion, must be reviewable in a diff and editable by hand, and has to exist before
there is a database to put it in. It is versioned in git; the PDFs it points at are not
(they are large, immutable and stored content-addressed under ``data/raw/``).

Layout of one row: see ``MANIFEST_COLUMNS``. Multi-valued fields (``supersedes``,
``amends``) are semicolon-separated, because the file is comma-separated.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict

from reglens.config import get_settings

Source = Literal["rbi", "sebi", "bank_ir", "exchange", "other"]
DocType = Literal[
    "master_direction",
    "master_circular",
    "circular",
    "notification",
    "regulation",
    "annual_report",
    "transcript",
    "quarterly_results",
    "press_release",
    "other",
]
Status = Literal["planned", "fetched", "parsed", "failed", "excluded"]
Discovery = Literal["direct", "listing"]
Level = Literal["error", "warning"]

MANIFEST_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "source",
    "issuer",
    "doc_type",
    "title",
    "url",
    "issue_date",
    "effective_date",
    "fiscal_period",
    "discovery",
    "status",
    "file_hash",
    "local_path",
    "parse_quality",
    "http_status",
    "fetched_at",
    "supersedes",
    "amends",
    "notes",
)

SOURCES: tuple[str, ...] = ("rbi", "sebi", "bank_ir", "exchange", "other")
DOC_TYPES: tuple[str, ...] = (
    "master_direction",
    "master_circular",
    "circular",
    "notification",
    "regulation",
    "annual_report",
    "transcript",
    "quarterly_results",
    "press_release",
    "other",
)
STATUSES: tuple[str, ...] = ("planned", "fetched", "parsed", "failed", "excluded")
DISCOVERIES: tuple[str, ...] = ("direct", "listing")
#: Statuses that imply bytes on disk. ``planned``, ``failed`` and ``excluded`` do not:
#: an entry-point or deliberately skipped row has nothing to hash.
FETCHED_STATUSES: frozenset[str] = frozenset({"fetched", "parsed"})

DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,79}$")
FISCAL_PERIOD_RE = re.compile(r"^FY\d{4}(-(Q[1-4]|H[12]))?$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Hosts we deliberately collect from. Checked as a suffix so subdomains
# (e.g. rbidocs.rbi.org.in) pass. A new host is a warning, not an error, because the
# hard gate is the fetcher: robots.txt plus an explicit operator confirmation.
ALLOWED_HOST_SUFFIXES: tuple[str, ...] = (
    "rbi.org.in",
    "sebi.gov.in",
    "bseindia.com",
    "nseindia.com",
    "sbi.co.in",
    "hdfcbank.com",
    "icicibank.com",
    "axisbank.com",
    "kotak.com",
    "indusind.com",
    "federalbank.co.in",
    "bajajfinserv.in",
    "bajajfinserv.com",
    "cholamandalam.com",
    "cholainvest.com",
    "shriramfinance.in",
    "recindia.nic.in",
    "pfcindia.com",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "doc_id",
    "source",
    "issuer",
    "doc_type",
    "title",
    "url",
    "discovery",
    "status",
)


class DocumentRecord(BaseModel):
    """One manifest row. Fields are permissive on purpose: validation is a separate,
    aggregate step (:func:`validate_records`) so an operator sees every problem at once
    instead of one pydantic error at a time."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    doc_id: str
    source: str
    issuer: str
    doc_type: str
    title: str
    url: str
    issue_date: str = ""
    effective_date: str = ""
    fiscal_period: str = ""
    discovery: str = "direct"
    status: str = "planned"
    file_hash: str = ""
    local_path: str = ""
    parse_quality: str = ""
    http_status: str = ""
    fetched_at: str = ""
    supersedes: str = ""
    amends: str = ""
    notes: str = ""

    # ---------------------------------------------------------------- helpers
    @property
    def multi_supersedes(self) -> list[str]:
        return [item for item in (self.supersedes or "").split(";") if item.strip()]

    @property
    def multi_amends(self) -> list[str]:
        return [item for item in (self.amends or "").split(";") if item.strip()]

    @property
    def is_fetched(self) -> bool:
        return bool(self.file_hash) and self.status in {"fetched", "parsed"}

    def to_row(self) -> dict[str, str]:
        data = self.model_dump()
        return {column: str(data.get(column, "")) for column in MANIFEST_COLUMNS}


@dataclass(frozen=True)
class Issue:
    """A single validation finding."""

    level: Level
    doc_id: str
    column: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "doc_id": self.doc_id,
            "column": self.column,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated validation result: errors block, warnings inform."""

    issues: list[Issue]
    record_count: int

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "record_count": self.record_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.as_dict() for issue in self.issues],
        }


# ------------------------------------------------------------------ hashing / I-O
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path | None = None) -> list[DocumentRecord]:
    """Read the manifest CSV. Raises ValueError if required columns are missing."""
    target = path or get_settings().manifest_path
    if not target.is_file():
        raise FileNotFoundError(f"manifest not found: {target}")
    text = target.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    header = tuple(reader.fieldnames or ())
    missing = [column for column in REQUIRED_FIELDS if column not in header]
    if missing:
        raise ValueError(f"{target} is missing required columns: {', '.join(missing)}")
    unknown = [column for column in header if column not in MANIFEST_COLUMNS]
    if unknown:
        raise ValueError(f"{target} has unknown columns: {', '.join(unknown)}")
    return [DocumentRecord.model_validate(row) for row in reader]


def write_manifest(records: Iterable[DocumentRecord], path: Path | None = None) -> Path:
    """Write the manifest CSV atomically (temp file + replace) so a crash cannot halve it."""
    target = path or get_settings().manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_row())
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    tmp.replace(target)
    return target


# ------------------------------------------------------------------ validation
def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _host_allowed(url: str) -> bool:
    host = url.split("//", 1)[-1].split("/", 1)[0].split("?", 1)[0].lower()
    host = host.split("@")[-1].split(":", 1)[0]
    return any(host == suffix or host.endswith("." + suffix) for suffix in ALLOWED_HOST_SUFFIXES)


def validate_records(records: list[DocumentRecord]) -> ValidationReport:
    """Check the manifest: identity, enums, dates, hashes, links, and on-disk hashes."""
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    settings = get_settings()

    for index, record in enumerate(records, start=2):  # row 2 = first data row
        doc_id = record.doc_id or f"<row {index}>"

        if not record.doc_id:
            issues.append(Issue("error", doc_id, "doc_id", "doc_id is required"))
        elif not DOC_ID_RE.match(record.doc_id):
            issues.append(
                Issue(
                    "error",
                    doc_id,
                    "doc_id",
                    "doc_id must be lowercase [a-z0-9_.-], 3-80 chars, starting alphanumeric",
                )
            )
        if record.doc_id in seen:
            issues.append(
                Issue(
                    "error",
                    doc_id,
                    "doc_id",
                    f"duplicate doc_id (first seen on row {seen[record.doc_id]})",
                )
            )
        seen[record.doc_id] = index

        for column, allowed in (
            ("source", SOURCES),
            ("doc_type", DOC_TYPES),
            ("status", STATUSES),
            ("discovery", DISCOVERIES),
        ):
            value = getattr(record, column)
            if value not in allowed:
                issues.append(
                    Issue("error", doc_id, column, f"{column}={value!r} not in {list(allowed)}")
                )

        if not record.title:
            issues.append(Issue("error", doc_id, "title", "title is required"))
        if not record.issuer:
            issues.append(Issue("error", doc_id, "issuer", "issuer is required"))

        if not record.url:
            issues.append(Issue("error", doc_id, "url", "url is required"))
        elif not _is_http_url(record.url):
            issues.append(Issue("error", doc_id, "url", f"url must be http(s): {record.url!r}"))
        elif not _host_allowed(record.url):
            issues.append(
                Issue(
                    "warning",
                    doc_id,
                    "url",
                    f"host not in the approved source list; confirm terms of use before fetching: {record.url}",
                )
            )

        for column in ("issue_date", "effective_date"):
            value = getattr(record, column)
            if value and not DATE_RE.match(value):
                issues.append(
                    Issue("error", doc_id, column, f"{column} must be YYYY-MM-DD, got {value!r}")
                )
        if record.fiscal_period and not FISCAL_PERIOD_RE.match(record.fiscal_period):
            issues.append(
                Issue(
                    "error",
                    doc_id,
                    "fiscal_period",
                    f"fiscal_period must look like FY2025 or FY2025-Q3, got {record.fiscal_period!r}",
                )
            )

        # Regulations always have a real issue date; filings carry a fiscal period instead.
        if (
            record.status in FETCHED_STATUSES
            and record.source in {"rbi", "sebi"}
            and not record.issue_date
        ):
            issues.append(
                Issue(
                    "error",
                    doc_id,
                    "issue_date",
                    "regulatory documents require issue_date once fetched",
                )
            )

        if record.file_hash:
            if not HASH_RE.match(record.file_hash):
                issues.append(
                    Issue("error", doc_id, "file_hash", "file_hash must be 64 lowercase hex chars")
                )
        elif record.status in FETCHED_STATUSES:
            issues.append(
                Issue("error", doc_id, "file_hash", f"status={record.status} requires a file_hash")
            )

        if record.status in {"fetched", "parsed"} and not record.local_path:
            issues.append(
                Issue("error", doc_id, "local_path", "fetched/parsed rows require local_path")
            )

        if record.parse_quality:
            try:
                quality = float(record.parse_quality)
            except ValueError:
                issues.append(
                    Issue(
                        "error",
                        doc_id,
                        "parse_quality",
                        f"parse_quality must be a float, got {record.parse_quality!r}",
                    )
                )
            else:
                if not 0.0 <= quality <= 1.0:
                    issues.append(
                        Issue(
                            "error",
                            doc_id,
                            "parse_quality",
                            f"parse_quality must be in [0, 1], got {quality}",
                        )
                    )
        elif record.status == "parsed":
            issues.append(
                Issue(
                    "warning", doc_id, "parse_quality", "parsed document has no parse_quality score"
                )
            )

        if (
            record.issue_date
            and record.effective_date
            and record.effective_date < record.issue_date
        ):
            issues.append(
                Issue(
                    "warning",
                    doc_id,
                    "effective_date",
                    f"effective_date {record.effective_date} precedes issue_date {record.issue_date}",
                )
            )

        # On-disk hash verification is the immutability guarantee.
        if record.local_path:
            raw_path = settings.raw_dir / record.local_path
            if not raw_path.is_file():
                issues.append(
                    Issue("error", doc_id, "local_path", f"file missing on disk: {raw_path}")
                )
            elif record.file_hash and sha256_file(raw_path) != record.file_hash:
                issues.append(
                    Issue(
                        "error",
                        doc_id,
                        "file_hash",
                        "file on disk does not match the recorded hash (raw files are immutable)",
                    )
                )
            else:
                # Same check the fetcher applies on download: an HTML interstitial
                # recorded as a PDF would otherwise pass every hash validation.
                ok, detail = _payload_matches_url(record.url, raw_path.read_bytes())
                if record.url and not ok:
                    issues.append(
                        Issue(
                            "error",
                            doc_id,
                            "local_path",
                            f"stored payload is not the document: {detail}",
                        )
                    )

    known_ids = {record.doc_id for record in records}
    for record in records:
        for column, refs in (
            ("supersedes", record.multi_supersedes),
            ("amends", record.multi_amends),
        ):
            for ref in refs:
                if ref not in known_ids:
                    issues.append(
                        Issue(
                            "error",
                            record.doc_id,
                            column,
                            f"{column} references unknown doc_id {ref!r}",
                        )
                    )
                if ref == record.doc_id:
                    issues.append(
                        Issue("error", record.doc_id, column, "a document cannot reference itself")
                    )

    return ValidationReport(issues=issues, record_count=len(records))


def _payload_matches_url(url: str, payload: bytes) -> tuple[bool, str]:
    """Check that the bytes are what the URL claims to be.

    Found by running the real fetch: rbidocs answered a PDF URL with a ~45 KB
    ``<!DOCTYPE`` interstitial, and nothing in the pipeline objected - 18 HTML pages were
    hashed and recorded as fetched documents. An interstitial recorded as evidence is
    worse than a failure, because idempotency would then skip those rows forever.
    """
    path = urlparse(url).path.lower()
    head = payload[:80].lstrip()
    if path.endswith(".pdf"):
        if payload[:5] == b"%PDF-":
            return True, "pdf"
        snippet = head[:40].decode("utf-8", errors="replace").replace("\n", " ")
        return False, f"expected %PDF- header, got {snippet!r}"
    if path.endswith((".xlsx", ".xlsm")):
        if payload[:2] == b"PK":
            return True, "zip-based excel"
        return False, "expected a zip-based Excel workbook"
    if path.endswith(".xls"):
        if payload[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return True, "ole excel"
        return False, "expected an OLE Excel workbook"
    return True, "unverified type"


def payload_matches_url(url: str, payload: bytes) -> tuple[bool, str]:
    """Public wrapper so the fetcher and the validator share one implementation."""
    return _payload_matches_url(url, payload)


def verify_raw_store(records: list[DocumentRecord]) -> ValidationReport:
    """Re-hash every fetched file and check it is still the right kind of file.

    Used by CI and ``reglens status``. The payload check exists because a download that
    recorded an HTML interstitial must be caught on the next run, not by a human who
    happens to open the file.
    """
    issues: list[Issue] = []
    settings = get_settings()
    for record in records:
        if not record.local_path:
            continue
        path = settings.raw_dir / record.local_path
        if not path.is_file():
            issues.append(Issue("error", record.doc_id, "local_path", f"missing raw file {path}"))
            continue
        payload = path.read_bytes()
        actual = sha256_bytes(payload)
        if actual != record.file_hash:
            issues.append(
                Issue("error", record.doc_id, "file_hash", f"hash mismatch: {actual[:12]}…")
            )
        if record.url:
            ok, detail = _payload_matches_url(record.url, payload)
            if not ok:
                issues.append(
                    Issue(
                        "error",
                        record.doc_id,
                        "local_path",
                        f"stored payload is not the document: {detail}",
                    )
                )
    return ValidationReport(issues=issues, record_count=len(records))


# ------------------------------------------------------------------ raw store
def raw_relative_path(record: DocumentRecord, file_hash: str, suffix: str = ".pdf") -> str:
    """Content-addressed raw path, relative to ``data/raw``.

    Including the hash in the filename is what makes the store immutable and ingestion
    idempotent: a re-download that produces different bytes lands beside the old file
    (and is reported) instead of silently overwriting evidence.
    """
    return f"{record.source}/{record.doc_id}/{record.doc_id}__{file_hash[:12]}{suffix}"


def raw_path_for(record: DocumentRecord) -> Path | None:
    if not record.local_path:
        return None
    return get_settings().raw_dir / record.local_path


# ------------------------------------------------------------------ json schema
def manifest_json_schema() -> dict[str, Any]:
    """JSON Schema for one manifest row, plus the enum dictionaries tools need."""
    schema = DocumentRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "RegLens corpus manifest row"
    schema["x-manifest-columns"] = list(MANIFEST_COLUMNS)
    schema["x-required-fields"] = list(REQUIRED_FIELDS)
    schema["x-enums"] = {
        "source": list(SOURCES),
        "doc_type": list(DOC_TYPES),
        "status": list(STATUSES),
        "discovery": list(DISCOVERIES),
    }
    schema["x-allowed-host-suffixes"] = list(ALLOWED_HOST_SUFFIXES)
    return schema


def write_manifest_schema(path: Path | None = None) -> Path:
    target = path or get_settings().manifest_schema_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest_json_schema(), indent=2) + "\n", encoding="utf-8")
    return target


# ------------------------------------------------------------------ corpus plan
def load_corpus_plan(path: Path | None = None) -> dict[str, Any]:
    """Load ``data/corpus_plan.yaml``: the reviewed, human-editable source of the manifest."""
    target = path or get_settings().corpus_plan_path
    if not target.is_file():
        raise FileNotFoundError(f"corpus plan not found: {target}")
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{target} must contain a YAML mapping")
    return raw


def records_from_plan(
    plan: dict[str, Any], existing: list[DocumentRecord] | None = None
) -> list[DocumentRecord]:
    """Merge the plan into manifest rows, preserving fetch state for known doc_ids.

    Re-running this after editing the plan must never lose a hash, so fetched fields are
    carried over for documents that were already downloaded.
    """
    preserved = {record.doc_id: record for record in (existing or [])}
    rows: list[DocumentRecord] = []

    for entry in plan.get("documents", []):
        if not isinstance(entry, dict):
            raise ValueError("every entry under documents: must be a mapping")
        # Only pass through the keys the plan actually sets: filling every column with ""
        # would wipe the model defaults (status=planned, discovery=direct).
        payload = {
            key: entry[key] for key in MANIFEST_COLUMNS if key in entry and entry[key] is not None
        }
        record = DocumentRecord.model_validate(payload)
        previous = preserved.get(record.doc_id)
        if previous and previous.file_hash:
            record = record.model_copy(
                update={
                    "status": previous.status,
                    "file_hash": previous.file_hash,
                    "local_path": previous.local_path,
                    "parse_quality": previous.parse_quality,
                    "http_status": previous.http_status,
                    "fetched_at": previous.fetched_at,
                }
            )
        rows.append(record)

    known = {record.doc_id for record in rows}
    missing = sorted(set(preserved) - known)
    for doc_id in missing:
        # Documents already fetched but no longer in the plan stay in the manifest so
        # their evidence is not lost; they are marked excluded instead of deleted.
        dropped = preserved[doc_id].model_copy(update={"status": "excluded"})
        if not dropped.notes:
            dropped = dropped.model_copy(
                update={"notes": "removed from corpus_plan.yaml; kept for traceability"}
            )
        rows.append(dropped)
    return rows


def plan_summary(records: list[DocumentRecord]) -> dict[str, Any]:
    """Counts used by ``reglens status`` and the Phase 0 acceptance report."""
    by_source: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_doc_type: dict[str, int] = {}
    for record in records:
        by_source[record.source] = by_source.get(record.source, 0) + 1
        by_status[record.status] = by_status.get(record.status, 0) + 1
        by_doc_type[record.doc_type] = by_doc_type.get(record.doc_type, 0) + 1
    return {
        "total": len(records),
        "by_source": dict(sorted(by_source.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_doc_type": dict(sorted(by_doc_type.items())),
    }
