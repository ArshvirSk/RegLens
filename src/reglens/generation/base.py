"""Generation interfaces (Phase 1 for the answerer, Phase 4 for the verifier).

The provider-agnostic client exists so the small-vs-large model ablation (PRD Phase 2) is
a config change rather than a code change, and so the answerer never knows which vendor
is behind it. Token usage is part of the return value, not a side channel: cost per query
is a stated target and must be attributable to a stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface every provider wrapper must satisfy."""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 900,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class CitationCheck:
    """Result of verifying one citation against the retrieved context (FR1, Phase 4)."""

    citation_index: int
    chunk_id: str | None
    supported: bool
    support_score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Answer:
    """The generation result, before it is serialised into the API response."""

    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    support_score: float = 0.0
    citation_checks: list[CitationCheck] = field(default_factory=list)
    model: str | None = None
    #: Token accounting travels with the answer (FR6) rather than through shared client
    #: state, which would race under concurrent requests.
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@runtime_checkable
class Answerer(Protocol):
    """Turns a question plus retrieved context into a cited answer or a refusal."""

    name: str

    def answer(self, question: str, context: list[Any], **kwargs: Any) -> Answer: ...
