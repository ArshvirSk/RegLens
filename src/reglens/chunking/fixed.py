"""Fixed-size token windows: the deliberately naive Phase 1 baseline.

Why windows of *tokens* and not characters or sentences: every later ablation
(clause-aware, speaker-aware, parent-child) is measured against exactly this, so it has
to be simple, deterministic and cheap. Two details that matter more than they look:

* **Deterministic chunk ids.** ``<doc_id>__fixed__<index>`` — re-ingesting the same
  document re-creates the same ids, which is what makes ingest idempotent (FR5).
* **Page mapping.** Tokens carry the page they came from, so a citation can say
  ``p.17`` without re-opening the PDF at answer time.

Tokenizer note: Gemini ships no official BPE tokenizer, so chunk sizes are counted with
``tiktoken`` ``cl100k_base`` as a pinned, reproducible proxy. Counts are therefore
stable across runs and machines (a stated PRD requirement) while not being literal
Gemini token counts — recorded in docs/learning/phase-1.md rather than papered over.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from reglens.chunking.base import Chunk, ParsedDocument

ENCODING_NAME = "cl100k_base"


@runtime_checkable
class Tokenizer(Protocol):
    """Encode/decode boundary so tests can inject a trivial tokenizer."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    """Pinned ``tiktoken`` encoding (loaded lazily; downloads once, then cached)."""

    def __init__(self, encoding_name: str = ENCODING_NAME) -> None:
        self._encoding_name = encoding_name
        self._encoding = None

    @property
    def encoding(self):  # type: ignore[no-untyped-def]
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)


class FixedSizeChunker:
    """Fixed windows over the token stream with a fixed overlap.

    ``name`` satisfies the :class:`~reglens.chunking.base.Chunker` protocol; it is kept
    as a real class (not a module function) so Phase 2 can hold several strategies
    side by side behind one constructor argument in the ingest pipeline.
    """

    name = "fixed"

    def __init__(
        self,
        *,
        size: int = 512,
        overlap: int = 64,
        min_tokens: int = 48,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        if min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        self.size = size
        self.overlap = overlap
        self.min_tokens = min_tokens
        self.tokenizer = tokenizer or TiktokenTokenizer()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        tokens: list[int] = []
        page_of: list[int] = []
        # Empty pages contribute nothing — not even the separator, which would otherwise
        # turn a fully blank document into a two-token chunk.
        non_empty = [
            (page, page_tokens)
            for page in document.pages
            if (page_tokens := self.tokenizer.encode(page.text))
        ]
        for position, (page, page_tokens) in enumerate(non_empty):
            tokens.extend(page_tokens)
            page_of.extend([page.page_number] * len(page_tokens))
            if position < len(non_empty) - 1:
                separator = self.tokenizer.encode("\n\n")
                tokens.extend(separator)
                page_of.extend([page.page_number] * len(separator))

        if not tokens:
            return []

        stride = self.size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + self.size, len(tokens))
            # Absorb a too-small tail into the current window instead of emitting a
            # token fragment that can never answer anything — then stop, so the
            # absorbed tail is not re-emitted as a fully-overlapping extra chunk.
            tail_absorbed = False
            if 0 < len(tokens) - end < self.min_tokens:
                end = len(tokens)
                tail_absorbed = True
            window = tokens[start:end]
            if not window:
                break
            page_start = page_of[start]
            page_end = page_of[end - 1]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}__{self.name}__{index:05d}",
                    doc_id=document.doc_id,
                    text=self.tokenizer.decode(window),
                    chunk_index=index,
                    strategy=self.name,
                    page_start=page_start,
                    page_end=page_end,
                    clause_path=None,
                    section_title=None,
                    token_count=len(window),
                    metadata={"parser": document.parser},
                )
            )
            index += 1
            if tail_absorbed:
                break
            start += stride
        return chunks
