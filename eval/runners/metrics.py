"""Retrieval and generation metrics — pure functions, no model calls.

Every metric here computes against ``gold_passages`` and the retrieved chunks, never
against a model's opinion (eval/README.md). Two conventions matter for reading reports:

* ``None`` means *undefined for this question* (e.g. recall on an unanswerable question,
  or citation precision when the answer cited nothing). Aggregates report ``defined``
  counts next to means, and a metric with no defined values is reported as ``null`` —
  never as 0.
* A gold passage is "hit" by a chunk when the doc_ids match and the page ranges overlap
  (page 1-indexed, chunk pages inclusive). A question is recalled at k when *any* of its
  gold passages is hit inside the top-k retrieved chunks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from reglens.retrieval.base import RetrievedChunk


def passage_hit(gold_doc_id: str, gold_page: int | None, chunk: RetrievedChunk) -> bool:
    """Does one gold passage point into this chunk? Page ranges are inclusive."""
    if gold_doc_id != chunk.doc_id:
        return False
    if gold_page is None or chunk.page_start is None:
        return True  # clause-only gold: same document is the best we can check
    page_end = chunk.page_end if chunk.page_end is not None else chunk.page_start
    return chunk.page_start <= gold_page <= page_end


def recall_at_k(
    golds: Sequence[tuple[str, int | None]],
    retrieved: Sequence[RetrievedChunk],
    k: int,
) -> float | None:
    """Fraction of gold passages found in the top-k chunks; None when there are no golds."""
    if not golds:
        return None
    top = retrieved[:k]
    hits = sum(
        1
        for gold_doc, gold_page in golds
        if any(passage_hit(gold_doc, gold_page, chunk) for chunk in top)
    )
    return hits / len(golds)


def reciprocal_rank(
    golds: Sequence[tuple[str, int | None]],
    retrieved: Sequence[RetrievedChunk],
) -> float | None:
    """1/rank of the first chunk that hits any gold passage; None when there are no golds."""
    if not golds:
        return None
    for rank, chunk in enumerate(retrieved, start=1):
        if any(passage_hit(gold_doc, gold_page, chunk) for gold_doc, gold_page in golds):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    golds: Sequence[tuple[str, int | None]],
    retrieved: Sequence[RetrievedChunk],
    k: int,
) -> float | None:
    """Binary-relevance nDCG@k: a chunk is relevant iff it hits any gold passage."""
    if not golds:
        return None
    gains = [
        1.0
        if any(passage_hit(gold_doc, gold_page, chunk) for gold_doc, gold_page in golds)
        else 0.0
        for chunk in retrieved[:k]
    ]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class CitationScore:
    """One answer's citations: how many, and how many resolve into the context."""

    total: int
    resolved: int

    @property
    def precision(self) -> float | None:
        return self.resolved / self.total if self.total else None


def score_citations(labels: Sequence[str], retrieved: Sequence[RetrievedChunk]) -> CitationScore:
    """Citation precision against the retrieved context (not against gold).

    A label resolves when a chunk shares its title *and* its page falls in that chunk's
    page range — the same mapping the API performs, minus the "fall back to any chunk
    with this title" concession, because an off-page citation is exactly the defect this
    metric exists to count.
    """
    from reglens.generation.answerer import CITATION_RE

    resolved = 0
    for label in labels:
        match = CITATION_RE.search(label)
        if not match:
            continue
        title, page, _clause = match.groups()
        # ``page`` may be a printed range (``40-41``); the range's start is the page the
        # label points at, and it must still fall inside the chunk's page range.
        page_number = int(page.split("-", 1)[0])
        if any(
            chunk.doc_title.strip() == title.strip()
            and chunk.page_start is not None
            and chunk.page_end is not None
            and chunk.page_start <= page_number <= chunk.page_end
            for chunk in retrieved
        ):
            resolved += 1
    return CitationScore(total=len(labels), resolved=resolved)


def mean(values: Iterable[float | None]) -> float | None:
    """Mean of the *defined* values only; None when nothing is defined (missing ≠ 0)."""
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def summarize(records: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    """Aggregate per-question metric records into a report section.

    Each record needs: ``gold_count``, ``recall``, ``rr``, ``ndcg``, ``refused``,
    ``citation_precision``, ``has_citations``. ``None`` fields stay missing.
    """
    answerable = [record for record in records if record.get("gold_count", 0) > 0]
    unanswerable = [record for record in records if record.get("gold_count", 0) == 0]
    answered = [record for record in records if not record.get("refused")]
    return {
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval": {
            "k": k,
            "recall_at_k": mean(record.get("recall") for record in answerable),
            "recall_at_k_defined": sum(
                1 for record in answerable if record.get("recall") is not None
            ),
            "mrr": mean(record.get("rr") for record in answerable),
            "mrr_defined": sum(1 for record in answerable if record.get("rr") is not None),
            "ndcg_at_k": mean(record.get("ndcg") for record in answerable),
            "ndcg_at_k_defined": sum(1 for record in answerable if record.get("ndcg") is not None),
        },
        "generation": {
            "refusal_rate": mean(1.0 if record.get("refused") else 0.0 for record in records)
            if records
            else None,
            "correct_refusals": sum(
                1
                for record in unanswerable
                if record.get("refused") and record.get("gold_count") == 0
            ),
            "answered_rate": mean(1.0 if not record.get("refused") else 0.0 for record in records)
            if records
            else None,
            "citation_precision": mean(record.get("citation_precision") for record in answered),
            "citation_precision_defined": sum(
                1 for record in answered if record.get("citation_precision") is not None
            ),
            "answers_with_citations": sum(1 for record in answered if record.get("has_citations")),
        },
    }
