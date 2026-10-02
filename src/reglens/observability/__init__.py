"""Observability: structured logs, per-stage tracing, and cost accounting."""

from reglens.observability.cost import (
    MODEL_PRICES,
    PRICES_VERIFIED_ON,
    ModelPrice,
    UnknownModelError,
    describe_pricing,
    estimate_cost,
    try_estimate_cost,
)
from reglens.observability.logging import get_logger, log_event, setup_logging
from reglens.observability.tracing import Span, Trace, TraceRecord

__all__ = [
    "MODEL_PRICES",
    "PRICES_VERIFIED_ON",
    "ModelPrice",
    "Span",
    "Trace",
    "TraceRecord",
    "UnknownModelError",
    "describe_pricing",
    "estimate_cost",
    "get_logger",
    "log_event",
    "setup_logging",
    "try_estimate_cost",
]
