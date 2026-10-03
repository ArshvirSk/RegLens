"""Grounded answerer: prompt shape, citation parsing, and refusal behaviour.

The refusal tests matter most: ``INSUFFICIENT EVIDENCE`` must survive untouched from
prompt to parsed answer, because the Phase 1 eval measures correct refusals against
exactly this string.
"""

from __future__ import annotations

from datetime import date

from reglens.generation.answerer import (
    REFUSAL_MARKER,
    GroundedAnswerer,
    build_context_block,
    parse_citations,
)
from reglens.generation.base import Answerer, LLMResponse, Message
from reglens.retrieval.base import RetrievedChunk


def make_chunk(**overrides: object) -> RetrievedChunk:
    fields: dict[str, object] = {
        "chunk_id": "doc__fixed__00000",
        "doc_id": "doc",
        "text": "The LCR shall be not less than one hundred per cent.",
        "doc_title": "Master Direction - Liquidity",
        "issuer": "RBI",
        "doc_type": "master_direction",
        "page_start": 12,
        "page_end": 13,
        "clause_path": "4.2",
        "section_title": None,
        "issue_date": date(2025, 11, 28),
        "effective_date": None,
        "score": 0.9,
    }
    fields.update(overrides)
    return RetrievedChunk(**fields)  # type: ignore[arg-type]


class FakeLLM:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 900,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        return LLMResponse(
            text=self.text,
            model=model,
            input_tokens=100,
            output_tokens=25,
            finish_reason="STOP",
            latency_ms=42,
        )


def answerer_with(text: str) -> tuple[GroundedAnswerer, FakeLLM]:
    llm = FakeLLM(text)
    answerer = GroundedAnswerer(llm=llm, model="gemini-3.5-flash-lite")  # type: ignore[arg-type]
    return answerer, llm


def test_answerer_satisfies_the_protocol() -> None:
    answerer, _ = answerer_with("x")
    assert isinstance(answerer, Answerer)
    assert isinstance(answerer, object)


def test_context_block_carries_pages_and_clauses() -> None:
    block = build_context_block([make_chunk(), make_chunk(page_start=20, page_end=20)])
    assert "[1] Master Direction - Liquidity | p.12-13 | clause 4.2" in block
    assert "[2] Master Direction - Liquidity | p.20 | clause 4.2" in block
    assert "not less than one hundred per cent" in block
    no_clause = build_context_block([make_chunk(clause_path=None, section_title=None)])
    assert "| clause n/a" in no_clause


def test_citations_are_parsed_and_deduplicated() -> None:
    text = (
        "Answer one. [Direction A, p.3, 2.1] "
        "Answer two. [Direction A, p.3, 2.1] "
        "Answer three. [Direction B, p.9, n/a]"
    )
    assert parse_citations(text) == [
        "[Direction A, p.3, 2.1]",
        "[Direction B, p.9, n/a]",
    ]
    assert parse_citations("no citations here [not a citation]") == []


def test_citations_tolerate_page_ranges_and_spacing() -> None:
    """The context block itself prints ``p.40-41`` for multi-page chunks, so a model
    citing that verbatim is a *correct* citation, not a mangled format."""
    text = (
        "Answer. [Reserve Bank of India (Commercial Banks - Know Your Customer) "
        "Directions, 2025, p.40-41, clause n/a] Also [Direction B, p. 7, clause 3.2]."
    )
    assert parse_citations(text) == [
        "[Reserve Bank of India (Commercial Banks - Know Your Customer) "
        "Directions, 2025, p.40-41, clause n/a]",
        "[Direction B, p.7, clause 3.2]",
    ]


def test_answer_passes_context_and_usage_through() -> None:
    answerer, llm = answerer_with("The LCR is 100%. [Master Direction - Liquidity, p.12, 4.2]")
    answer = answerer.answer("What is the LCR?", [make_chunk()])

    assert isinstance(answer, object)
    assert answer.refused is False
    assert answer.citations == ["[Master Direction - Liquidity, p.12, 4.2]"]
    assert answer.input_tokens == 100
    assert answer.output_tokens == 25
    assert answer.model == "gemini-3.5-flash-lite"

    call = llm.calls[0]
    messages = call["messages"]
    assert messages[0].role == "system"
    assert REFUSAL_MARKER in messages[0].content
    assert "only facts stated in the context" in messages[0].content
    assert "Question: What is the LCR?" in messages[1].content
    assert "[1] Master Direction - Liquidity" in messages[1].content
    assert call["temperature"] == 0.0


def test_refusal_marker_is_detected_verbatim() -> None:
    answerer, llm = answerer_with(f"  {REFUSAL_MARKER}  ")
    answer = answerer.answer("Unanswerable?", [make_chunk()])
    assert answer.refused is True
    assert answer.refusal_reason == "no_evidence"
    assert answer.citations == []
    assert len(llm.calls) == 1


def test_empty_model_output_is_a_refusal_not_a_crash() -> None:
    answerer, _llm = answerer_with("")
    answer = answerer.answer("Q?", [make_chunk()])
    assert answer.refused is True
    assert answer.refusal_reason == "no_evidence"


def test_no_context_short_circuits_without_a_model_call() -> None:
    answerer, llm = answerer_with("should never be used")
    answer = answerer.answer("Q?", [])
    assert answer.refused is True
    assert answer.refusal_reason == "no_evidence"
    assert answer.text == ""
    assert llm.calls == []
