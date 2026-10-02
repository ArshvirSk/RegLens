"""Gemini chat behind the provider-agnostic :class:`~reglens.generation.base.LLMClient`.

The protocol takes a ``model`` per call rather than per client: the small-vs-large
ablation (Phase 2) and the answerer-vs-judge split (Phase 1) are both config changes,
never a second client instance. Token usage is parsed out of the response — including
``thoughts_token_count``, because Gemini bills thinking tokens as output and a cost
report that ignores them would understate spend.
"""

from __future__ import annotations

import time
from typing import Any

from google.genai import types

from reglens.config import Settings, get_settings
from reglens.generation.base import LLMClient, LLMResponse, Message


class GeminiLLM:
    """Chat completion via ``google-genai``. ``client`` is injectable for tests."""

    name = "gemini"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        self._api_key = api_key
        self._client = client

    def _sdk(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key) if self._api_key else genai.Client()
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 900,
    ) -> LLMResponse:
        system_instruction = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        contents = [
            types.Content(
                role="user" if message.role == "user" else "model",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
            if message.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        response = self._sdk().models.generate_content(
            model=model, contents=contents, config=config
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidates_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or 0)

        try:
            text = response.text or ""
        except (ValueError, AttributeError):
            # Blocked/empty candidates: an empty string keeps the caller honest — the
            # answerer decides whether an empty answer is a refusal, this does not.
            text = ""

        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = getattr(reason, "name", None) or (str(reason) if reason else None)

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            # Thinking tokens are billed as output tokens; count them as such.
            output_tokens=candidates_tokens + thoughts_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw={
                "prompt_token_count": input_tokens,
                "candidates_token_count": candidates_tokens,
                "thoughts_token_count": thoughts_tokens,
                "total_token_count": total_tokens,
            },
        )


def build_llm(settings: Settings | None = None) -> LLMClient:
    """Chat backend for the configured provider (Phase 1: gemini only)."""
    active = settings or get_settings()
    if active.llm_provider != "gemini":
        raise NotImplementedError(
            f"LLM provider {active.llm_provider!r} ships with the provider ablation; "
            "Phase 1 measures gemini only"
        )
    return GeminiLLM(api_key=active.require_gemini_key())
