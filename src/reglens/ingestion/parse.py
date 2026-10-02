"""Parsing digital PDFs: two parsers behind one measured comparison.

Phase 1 parses the fetched corpus with PyMuPDF (``pymupdf``) and pdfplumber and records
which one to standardise on — speed, text yield, and page-by-page agreement — as a
versioned report under ``eval/results/``. The winner becomes the experiment config field
``parsing.parser``; nothing in the pipeline hard-codes a parser import, so swapping it is
a YAML edit plus a re-run of the comparison.

OCR is deliberately not wired yet: every fetched page in the corpus so far has a text
layer. Pages that come out nearly empty are *flagged* (``scan_pages``) so a scanned page
stays visible instead of silently becoming an empty chunk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from reglens.chunking.base import ParsedDocument, ParsedPage

ParserName = Literal["pymupdf", "pdfplumber"]
PARSERS: tuple[ParserName, ...] = ("pymupdf", "pdfplumber")


def parse_pdf(
    path: Path,
    doc_id: str,
    *,
    parser: ParserName = "pymupdf",
    min_chars_per_page: int = 40,
) -> ParsedDocument:
    """Extract every page of ``path`` as plain data, with a quality score.

    ``ParsedDocument`` (not a library type) is the boundary: chunkers and the eval
    harness must be testable without a PDF library installed.
    """
    if parser == "pymupdf":
        pages, extra = _parse_pymupdf(path)
    elif parser == "pdfplumber":
        pages, extra = _parse_pdfplumber(path)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown parser {parser!r}; expected one of {PARSERS}")
    return ParsedDocument(
        doc_id=doc_id,
        pages=pages,
        parser=parser,
        page_count=len(pages),
        parse_quality=quality_score(pages, min_chars=min_chars_per_page),
        metadata={
            **extra,
            "source_file": path.name,
            "scan_pages": low_text_pages(pages, min_chars=min_chars_per_page),
        },
    )


def _parse_pymupdf(path: Path) -> tuple[list[ParsedPage], dict[str, Any]]:
    import pymupdf

    pages: list[ParsedPage] = []
    document = pymupdf.open(path)
    try:
        for index, page in enumerate(document):
            blocks = [str(block[4]) for block in page.get_text("blocks") if str(block[4]).strip()]
            pages.append(
                ParsedPage(
                    page_number=index + 1,
                    text=page.get_text("text"),
                    blocks=blocks,
                    ocr_used=False,
                )
            )
    finally:
        document.close()
    return pages, {"pymupdf_version": getattr(pymupdf, "__version__", "unknown")}


def _parse_pdfplumber(path: Path) -> tuple[list[ParsedPage], dict[str, Any]]:
    import pdfplumber

    pages: list[ParsedPage] = []
    with pdfplumber.open(path) as document:
        for index, page in enumerate(document.pages):
            # No reliable block grouping here; clause-aware chunking (Phase 2) reads
            # blocks from whichever parser produced them and must handle [].
            pages.append(
                ParsedPage(
                    page_number=index + 1,
                    text=page.extract_text() or "",
                    blocks=[],
                    ocr_used=False,
                )
            )
    return pages, {"pdfplumber_version": getattr(pdfplumber, "__version__", "unknown")}


def quality_score(pages: list[ParsedPage], *, min_chars: int = 40) -> float:
    """Fraction of pages that yielded real text. 1.0 = every page is machine-readable."""
    if not pages:
        return 0.0
    readable = sum(1 for page in pages if len(page.text.strip()) >= min_chars)
    return round(readable / len(pages), 4)


def low_text_pages(pages: list[ParsedPage], *, min_chars: int = 40) -> list[int]:
    """1-indexed pages that yielded (almost) no text: scan/OCR candidates."""
    return [page.page_number for page in pages if len(page.text.strip()) < min_chars]


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def text_agreement(a: str, b: str) -> float:
    """Token-set Jaccard between two extractions of the same page/document.

    Order-insensitive on purpose: the two parsers interleave whitespace and line breaks
    differently, and the question is whether they agree on the *words*.
    """
    tokens_a = set(normalized(a).split())
    tokens_b = set(normalized(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 4)


def compare_parsers(
    documents: list[tuple[str, Path]],
    *,
    min_chars_per_page: int = 40,
) -> dict[str, Any]:
    """Run every parser over every document and return a JSON-ready report body.

    Timings are wall-clock single runs (no repetition): they are indicative, and the
    report says so, rather than pretending to be microbenchmarks.
    """
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {
        parser: {"seconds": 0.0, "chars": 0, "pages_with_text": 0, "quality_sum": 0.0}
        for parser in PARSERS
    }
    faster_count: dict[str, int] = dict.fromkeys(PARSERS, 0)

    for doc_id, path in documents:
        per_parser: dict[str, dict[str, Any]] = {}
        for parser in PARSERS:
            started = time.perf_counter()
            parsed = parse_pdf(path, doc_id, parser=parser, min_chars_per_page=min_chars_per_page)
            elapsed = time.perf_counter() - started
            chars = sum(len(page.text) for page in parsed.pages)
            per_parser[parser] = {
                "seconds": round(elapsed, 3),
                "chars": chars,
                "pages": parsed.page_count,
                "pages_with_text": parsed.page_count - len(parsed.metadata.get("scan_pages", [])),
                "quality": parsed.parse_quality,
                "scan_pages": parsed.metadata.get("scan_pages", []),
                "_full_text": parsed.full_text,
            }
            bucket = totals[parser]
            bucket["seconds"] += elapsed
            bucket["chars"] += chars
            bucket["pages_with_text"] += per_parser[parser]["pages_with_text"]
            bucket["quality_sum"] += parsed.parse_quality or 0.0

        a, b = per_parser["pymupdf"], per_parser["pdfplumber"]
        if a["seconds"] < b["seconds"]:
            faster_count["pymupdf"] += 1
        elif b["seconds"] < a["seconds"]:
            faster_count["pdfplumber"] += 1

        rows.append(
            {
                "doc_id": doc_id,
                "file": path.name,
                "pages": a["pages"],
                "agreement_token_jaccard": text_agreement(a.pop("_full_text"), b.pop("_full_text")),
                "pymupdf": a,
                "pdfplumber": b,
            }
        )

    document_count = len(rows)
    aggregate = {
        "documents": document_count,
        "faster_count": faster_count,
        "mean_agreement": round(
            sum(row["agreement_token_jaccard"] for row in rows) / document_count, 4
        )
        if document_count
        else None,
        "per_parser": {
            parser: {
                "total_seconds": round(totals[parser]["seconds"], 3),
                "total_chars": int(totals[parser]["chars"]),
                "total_pages_with_text": int(totals[parser]["pages_with_text"]),
                "mean_quality": round(totals[parser]["quality_sum"] / document_count, 4)
                if document_count
                else None,
            }
            for parser in PARSERS
        },
        "scan_pages_detected": sorted(
            {page for row in rows for parser in PARSERS for page in row[parser]["scan_pages"]}
        ),
    }
    return {"parsers": list(PARSERS), "documents": rows, "aggregate": aggregate}
