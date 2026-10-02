"""Grounded answering: context-only prompts, explicit citations, explicit refusal.

Phase 1 is deliberately naive — one prompt, no verification loop, no support threshold
(``min_support_score`` stays 0.0 and ``verify_citations`` off in the baseline config).
What is *not* negotiable even at baseline:

* The prompt forbids outside knowledge and demands ``[title, p.N, clause]`` after every
  factual claim (FR1's shape; Phase 4 verifies the links for real).
* ``INSUFFICIENT EVIDENCE`` is a first-class answer, not an error: a model that says
  "not in context" is the behaviour the whole refusal feature builds on, so it must
  exist — and be measurable — from the first eval run.

Citations are parsed back out of the model's text with a strict regex, so an answer
that mangles the format can be *seen* in the eval report instead of silently counting
as cited.
"""

from __future__ import annotations

import re
import time
from typing import Any

from reglens.generation.base import Answer, LLMClient, LLMResponse, Message
from reglens.retrieval.base import RetrievedChunk

REFUSAL_MARKER = "INSUFFICIENT EVIDENCE"
CITATION_RE = re.compile(r"\[([^\[\]\n]{3,160}?),\s*p\.(\d+),\s*([^\[\]\n]*?)\]")

SYSTEM_PROMPT = f"""\
You answer questions about Indian banking regulation ONLY from the numbered context \
passages provided below.

Rules:
1. Use only facts stated in the context. Never use outside knowledge, even if you are \
certain of the answer.
2. After every factual claim, add a citation in exactly this format: \
[document title, p.N, clause] — use clause "n/a" when the passage has no clause number.
3. If the context does not contain the answer, reply with exactly: {REFUSAL_MARKER}
   and nothing else.
4. Quote short phrases from the passages where they carry the answer."""


def build_context_block(context: list[RetrievedChunk]) -> str:
    parts = []
    for index, chunk in enumerate(context, start=1):
        pages = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end or chunk.page_end is None
            else f"p.{chunk.page_start}-{chunk.page_end}"
        )
        clause = chunk.clause_path or chunk.section_title or "n/a"
        parts.append(f"[{index}] {chunk.doc_title} | {pages} | clause {clause}\n{chunk.text}")
    return "\n\n".join(parts)


def parse_citations(text: str) -> list[str]:
    """Every ``[title, p.N, clause]`` label in answer order (unique, first wins)."""
    seen: dict[str, None] = {}
    for match in CITATION_RE.finditer(text):
        title, page, clause = match.groups()
        label = f"[{title.strip()}, p.{page}, {clause.strip()}]"
        seen.setdefault(label, None)
    return list(seen)


class GroundedAnswerer:
    """One prompt, one model call, citations parsed from the answer text."""

    name = "grounded_v0"

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 900,
    ) -> None:
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def answer(
        self, question: str, context: list[Any], **_kwargs: Any
    ) -> Answer:  # **kwargs mirrors the Answerer protocol; Phase 1 takes no options yet
        chunks = [chunk for chunk in context if isinstance(chunk, RetrievedChunk)]
        if not chunks:
            return Answer(
                text="",
                refused=True,
                refusal_reason="no_evidence",
                model=self.model,
            )
        user_prompt = f"Context:\n{build_context_block(chunks)}\n\nQuestion: {question}"
        started = time.perf_counter()
        response: LLMResponse = self.llm.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = response.text.strip()
        refused = (not text) or text.upper().startswith(REFUSAL_MARKER)
        return Answer(
            text=text,
            citations=parse_citations(text) if not refused else [],
            refused=refused,
            refusal_reason="no_evidence" if refused else None,
            support_score=0.0,  # Phase 4 replaces this with real verification
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency_ms or response.latency_ms,
        )
