"""Token pricing and cost estimation.

Cost per query is a stated non-functional target (under ~$0.02), so it has to be
computed from real token counts rather than estimated by vibes. Prices live in one table
because every provider changes them; an unknown model raises instead of silently
reporting $0.00, and the tracing layer turns that into a loud warning.

Provenance
----------
Values below were read from the official OpenAI pricing page on 2026-10-02
(https://developers.openai.com/api/docs/pricing) and cross-checked against published
per-token rates for the legacy GPT-4o family. They are *not* re-verified automatically:
``prices_verified_on`` is the date a human last checked, and any eval report that quotes
a cost is only as good as that date. Open-source models running locally cost $0 in API
terms but still consume GPU time; the table records them at zero with a reason, so the
reported "cost per query" never hides a local model behind a paid one.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_VERIFIED_ON = "2026-10-02"
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"


class UnknownModelError(KeyError):
    """Raised when a model has no price entry, so cost is never silently zero."""


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    notes: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


MODEL_PRICES: dict[str, ModelPrice] = {
    # --- generation (hosted) ---
    "gpt-4o-mini": ModelPrice(0.15, 0.60, "small generator / judge default"),
    "gpt-4o": ModelPrice(2.50, 10.00, "larger generator for the model-size ablation"),
    "gpt-6-luna": ModelPrice(0.10, 0.50, "current small flagship, short context"),
    # --- embeddings (hosted) ---
    "text-embedding-3-small": ModelPrice(0.02, 0.0, "1536 dims, baseline embedding model"),
    "text-embedding-3-large": ModelPrice(0.13, 0.0, "3072 dims, second hosted option"),
    # --- local ---
    "BAAI/bge-small-en-v1.5": ModelPrice(0.0, 0.0, "local CPU/GPU; no API spend"),
    "BAAI/bge-base-en-v1.5": ModelPrice(0.0, 0.0, "local CPU/GPU; no API spend"),
    "BAAI/bge-reranker-base": ModelPrice(0.0, 0.0, "local cross-encoder reranker"),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Cost in USD for one call. Raises :class:`UnknownModelError` for unpriced models."""
    price = MODEL_PRICES.get(model)
    if price is None:
        raise UnknownModelError(model)
    return price.cost(input_tokens, output_tokens)


def try_estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float | None:
    """Cost in USD, or ``None`` when the model has no price entry."""
    try:
        return estimate_cost(model, input_tokens, output_tokens)
    except UnknownModelError:
        return None


def describe_pricing() -> dict[str, object]:
    """Price table for docs/reports, so a reported cost can be traced to a rate."""
    return {
        "source": PRICE_SOURCE,
        "verified_on": PRICES_VERIFIED_ON,
        "unit": "USD per 1M tokens",
        "models": {
            name: {
                "input": price.input_per_mtok,
                "output": price.output_per_mtok,
                "notes": price.notes,
            }
            for name, price in sorted(MODEL_PRICES.items())
        },
    }
