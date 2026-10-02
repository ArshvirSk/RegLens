"""Ingestion: manifest handling, fetching, parsing, OCR, structure and table extraction.

Implemented in Phase 0: manifest model/validation and the polite fetcher.
Implemented in Phase 1: PDF parsing and structure extraction.
Implemented in Phase 3: table extraction into the ``metrics`` table.
"""

from reglens.ingestion.fetch import (
    DownloadAction,
    DownloadReport,
    FetchPolicy,
    FetchResult,
    RobotsCache,
    download_manifest,
    extract_links,
    fetch_bytes,
    filter_document_links,
    plan_downloads,
    resolve_listing,
)
from reglens.ingestion.manifest import (
    MANIFEST_COLUMNS,
    DocumentRecord,
    Issue,
    ValidationReport,
    load_corpus_plan,
    load_manifest,
    manifest_json_schema,
    plan_summary,
    raw_path_for,
    raw_relative_path,
    records_from_plan,
    sha256_bytes,
    sha256_file,
    validate_records,
    verify_raw_store,
    write_manifest,
    write_manifest_schema,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "DocumentRecord",
    "DownloadAction",
    "DownloadReport",
    "FetchPolicy",
    "FetchResult",
    "Issue",
    "RobotsCache",
    "ValidationReport",
    "download_manifest",
    "extract_links",
    "fetch_bytes",
    "filter_document_links",
    "load_corpus_plan",
    "load_manifest",
    "manifest_json_schema",
    "plan_downloads",
    "plan_summary",
    "raw_path_for",
    "raw_relative_path",
    "records_from_plan",
    "resolve_listing",
    "sha256_bytes",
    "sha256_file",
    "validate_records",
    "verify_raw_store",
    "write_manifest",
    "write_manifest_schema",
]
