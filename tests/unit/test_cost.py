"""Cost accounting tests.

Cost per query is a stated target, so a silent zero would be a lie in a report. An
unpriced model must raise here and be reported as a warning by the tracing layer.
"""

from __future__ import annotations

import pytest

from reglens.observability.cost import (
    MODEL_PRICES,
    PRICES_VERIFIED_ON,
    UnknownModelError,
    describe_pricing,
    estimate_cost,
    try_estimate_cost,
)


def test_generation_cost_uses_input_and_output_rates() -> None:
    # gpt-4o-mini: $0.15/M in, $0.60/M out -> 1M in + 1M out = $0.75
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_embedding_cost_is_input_only() -> None:
    assert estimate_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)


def test_local_models_cost_nothing_but_are_listed() -> None:
    assert estimate_cost("BAAI/bge-reranker-base", 10_000, 10_000) == 0.0
    assert "BAAI/bge-reranker-base" in MODEL_PRICES


def test_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        estimate_cost("gpt-9-imaginary", 10, 10)


def test_try_estimate_cost_returns_none_for_unknown_models() -> None:
    assert try_estimate_cost("gpt-9-imaginary", 10, 10) is None
    assert try_estimate_cost("gpt-4o-mini", 1000, 0) is not None


def test_typical_query_cost_is_well_under_the_target() -> None:
    """A ~6k-token context with a 400-token answer on the small model."""
    cost = estimate_cost("gpt-4o-mini", 6_000, 400)
    assert cost < 0.02


def test_pricing_provenance_is_recorded() -> None:
    table = describe_pricing()
    assert table["verified_on"] == PRICES_VERIFIED_ON
    assert table["source"].startswith("https://")
    assert set(table["models"]) == set(MODEL_PRICES)
