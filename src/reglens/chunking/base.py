"""Chunking interfaces (Phase 1-2).

Only the contract lives here in Phase 0, so every strategy is swappable behind one
signature from the moment it exists. The PRD's ablation list maps one-to-one to
implementations of :class:`Chunker`: ``fixed`` (Phase 1 baseline), ``clause_aware`` and
``speaker_aware`` (Phase 2), ``parent_child`` (Phase 2).

Why the ``ParsedDocument``/``ParsedPage`` types and not a PDF library type? Because the
chunker must stay testable without PDFs on disk: parser output is plain data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ParsedPage:
    """One page of parsed content, 1-indexed to match how citations are read."""

    page_number: int
    text: str
    blocks: list[str] = field(default_factory=list)
    ocr_used: bool = False


@dataclass(frozen=True)
class ParsedDocument:
    """Parser output: pages plus enough structure to chunk clause-aware."""

    doc_id: str
    pages: list[ParsedPage]
    parser: str
    page_count: int
    detected_sections: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    parse_quality: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. ``chunk_id`` must be deterministic for idempotent re-ingest."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    strategy: str
    page_start: int | None = None
    page_end: int | None = None
    clause_path: str | None = None
    section_title: str | None = None
    speaker: str | None = None
    token_count: int = 0
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    """Anything that turns a parsed document into chunks."""

    name: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Return chunks in reading order. Must be deterministic for the same input."""
        ...
