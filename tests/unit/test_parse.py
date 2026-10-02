"""Parser comparison: both parsers read the same PDF; blanks are flagged, not hidden."""

from __future__ import annotations

import json

import pymupdf
import pytest

from reglens.ingestion.parse import compare_parsers, parse_pdf, quality_score, text_agreement

LOREM = (
    "The Reserve Bank of India regulates priority sector lending targets for scheduled "
    "commercial banks. Parser comparison page text must exceed forty characters. "
    "Regulatory documents wrap their lines within the page margins, so the fixture "
    "must wrap too: a single overlong line would be clipped by one parser and not the "
    "other, measuring page geometry instead of extraction quality."
)


@pytest.fixture
def sample_pdf(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    for index in range(2):
        page = document.new_page()
        # insert_textbox wraps inside the rect; insert_text would overflow and get
        # clipped by pymupdf but not by pdfplumber.
        rect = pymupdf.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        remaining = page.insert_textbox(rect, f"{LOREM}page {index + 1}")
        assert remaining >= 0, "fixture text must fit the textbox"
    document.new_page()  # a blank page: the scan/OCR candidate case
    document.save(path)
    document.close()
    return path


def test_pymupdf_extracts_every_page_and_flags_the_blank_one(sample_pdf) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_pdf(sample_pdf, "doc_sample", parser="pymupdf")
    assert parsed.page_count == 3
    assert "priority sector lending" in parsed.pages[0].text.lower()
    assert parsed.metadata["scan_pages"] == [3]
    assert parsed.parse_quality == pytest.approx(2 / 3, abs=0.001)
    assert parsed.pages[0].blocks, "pymupdf should expose block structure"


def test_pdfplumber_agrees_with_pymupdf(sample_pdf) -> None:  # type: ignore[no-untyped-def]
    first = parse_pdf(sample_pdf, "doc_sample", parser="pymupdf")
    second = parse_pdf(sample_pdf, "doc_sample", parser="pdfplumber")
    assert first.page_count == second.page_count == 3
    assert second.metadata["scan_pages"] == [3]
    assert "priority sector lending" in second.pages[0].text.lower()
    assert text_agreement(first.full_text, second.full_text) >= 0.8


def test_quality_score_edges() -> None:
    from reglens.chunking.base import ParsedPage

    assert quality_score([]) == 0.0
    assert quality_score([ParsedPage(page_number=1, text="short")]) == 0.0
    long_page = "a page with plenty of extracted text for the quality score"
    assert quality_score([ParsedPage(page_number=1, text=long_page)]) == 1.0


def test_text_agreement_is_symmetric_and_bounded() -> None:
    assert text_agreement("a b c", "a b c") == 1.0
    assert text_agreement("", "") == 1.0
    assert text_agreement("a b", "") == 0.0
    assert text_agreement("a b c", "b c d") == text_agreement("b c d", "a b c")
    assert 0.0 <= text_agreement("a b c", "c d e") <= 1.0


def test_compare_parsers_produces_a_json_ready_report(sample_pdf) -> None:  # type: ignore[no-untyped-def]
    report = compare_parsers([("doc_sample", sample_pdf)])
    assert report["parsers"] == ["pymupdf", "pdfplumber"]
    row = report["documents"][0]
    assert row["pymupdf"]["seconds"] >= 0
    assert row["pdfplumber"]["seconds"] >= 0
    assert 0.0 <= row["agreement_token_jaccard"] <= 1.0
    aggregate = report["aggregate"]
    assert aggregate["documents"] == 1
    assert sum(aggregate["faster_count"].values()) == 1
    assert aggregate["scan_pages_detected"] == [3]
    json.dumps(report)  # the report must serialise for eval/results/


def test_unknown_parser_is_rejected(sample_pdf) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="unknown parser"):
        parse_pdf(sample_pdf, "doc_sample", parser="bogus")  # type: ignore[arg-type]
