"""Fixed-size chunker: determinism, overlap, page mapping, tail absorption."""

from __future__ import annotations

import pytest

from reglens.chunking.base import Chunker, ParsedDocument, ParsedPage
from reglens.chunking.fixed import FixedSizeChunker


class CharTokenizer:
    """Offline tokenizer: one token per character, so tests never touch the network."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def make_document() -> ParsedDocument:
    # Unique tokens per position: with homogeneous "xxxx" text a later window is a
    # substring of an earlier one by accident, which would mask real duplication bugs.
    page_one = " ".join(f"a{i:04d}" for i in range(60))
    page_two = " ".join(f"b{i:04d}" for i in range(60))
    page_three = " ".join(f"c{i:04d}" for i in range(20))
    pages = [
        ParsedPage(page_number=1, text=page_one),
        ParsedPage(page_number=2, text=page_two),
        ParsedPage(page_number=3, text=page_three),
    ]
    return ParsedDocument(doc_id="doc_test", pages=pages, parser="pymupdf", page_count=3)


def make_chunker(**overrides: object) -> FixedSizeChunker:
    return FixedSizeChunker(
        size=100,
        overlap=20,
        min_tokens=10,
        tokenizer=CharTokenizer(),
        **overrides,  # type: ignore[arg-type]
    )


def test_chunker_satisfies_the_protocol() -> None:
    assert isinstance(make_chunker(), Chunker)


def test_chunks_are_deterministic_with_stable_ids() -> None:
    first = make_chunker().chunk(make_document())
    second = make_chunker().chunk(make_document())
    assert first and second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert first[0].chunk_id == "doc_test__fixed__00000"
    assert first[1].chunk_id == "doc_test__fixed__00001"
    assert all(chunk.strategy == "fixed" for chunk in first)


def test_windows_overlap_and_respect_size_bounds() -> None:
    chunks = make_chunker().chunk(make_document())
    assert len(chunks) >= 3
    for chunk in chunks:
        assert 10 <= chunk.token_count <= 100 + 10
    # The second window starts inside the first: its decoded prefix must appear in it.
    assert chunks[1].text[:15] in chunks[0].text


def test_pages_are_mapped_and_monotonic() -> None:
    chunks = make_chunker().chunk(make_document())
    pages = [(chunk.page_start, chunk.page_end) for chunk in chunks]
    assert all(start is not None and end is not None and start <= end for start, end in pages)
    assert [start for start, _ in pages] == sorted(start for start, _ in pages)
    assert any(start < end for start, end in pages), "expected a chunk crossing a page break"
    assert pages[0][0] == 1
    assert chunks[-1].page_end == 3


def test_tail_is_absorbed_never_duplicated() -> None:
    """Regression: absorbing the tail must not also emit it again as its own chunk."""
    chunks = make_chunker().chunk(make_document())
    texts = [chunk.text for chunk in chunks]
    for index, text in enumerate(texts):
        for earlier in texts[:index]:
            assert text not in earlier, "a chunk was fully contained in an earlier chunk"
    assert texts[-1].endswith("c0019"), "the document tail must be covered"


def test_short_document_yields_one_usable_chunk() -> None:
    document = ParsedDocument(
        doc_id="doc_short",
        pages=[ParsedPage(page_number=1, text="tiny")],
        parser="pymupdf",
        page_count=1,
    )
    chunks = make_chunker().chunk(document)
    assert len(chunks) == 1
    assert "tiny" in chunks[0].text


def test_empty_document_yields_no_chunks() -> None:
    document = ParsedDocument(
        doc_id="doc_empty",
        pages=[ParsedPage(page_number=1, text="")],
        parser="pymupdf",
        page_count=1,
    )
    assert make_chunker().chunk(document) == []


@pytest.mark.parametrize(
    ("size", "overlap", "min_tokens"),
    [(0, 0, 10), (100, 100, 10), (100, -1, 10), (100, 20, 0)],
)
def test_invalid_parameters_are_rejected(size: int, overlap: int, min_tokens: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=size, overlap=overlap, min_tokens=min_tokens)
