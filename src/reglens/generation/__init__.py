"""Generation: prompts, provider-agnostic LLM client, citation verification, refusals.

Phase 0 defines contracts only; Phase 1 implements grounded answering, Phase 4 the
citation verifier and calibrated refusal threshold.
"""

from reglens.generation.base import (
    Answer,
    Answerer,
    CitationCheck,
    LLMClient,
    LLMResponse,
    Message,
)

__all__ = ["Answer", "Answerer", "CitationCheck", "LLMClient", "LLMResponse", "Message"]
