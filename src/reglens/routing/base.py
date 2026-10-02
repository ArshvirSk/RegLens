"""Routing interfaces (Phase 3).

Kept as a protocol because the naive baseline must not route at all: Phase 1 sends every
question down the text path. Routing is added in Phase 3 with its own ablation row, and
the router's accuracy is measured per question type rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

RouteKind = Literal["text", "numeric", "hybrid"]


@dataclass(frozen=True)
class RouteDecision:
    """The router's output: a route plus the reasoning used to audit it."""

    route: RouteKind
    confidence: float = 0.0
    reason: str = ""
    sub_queries: list[str] = field(default_factory=list)
    extract_hints: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Router(Protocol):
    name: str

    def route(self, question: str, *, as_of_date: Any = None) -> RouteDecision: ...


@runtime_checkable
class Tool(Protocol):
    """A capability the router can call (vector search, SQL over metrics, graph lookup)."""

    name: str

    def run(self, **kwargs: Any) -> Any: ...
