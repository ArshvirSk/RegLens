"""Eval metrics tests.

The metrics are the measuring instrument's arithmetic: hand-computed fixtures with
known answers, plus the missing-vs-zero rule (``None`` stays out of means).
"""

from __future__ import annotations

from eval.runners.metrics import (
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_citations,
    summarize,
)

from reglens.retrieval.base import RetrievedChunk


def chunk(
    chunk_id: str,
    doc_id: str,
    page_start: int | None,
    page_end: int | None = None,
    title: str = "Master Direction",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text of {chunk_id}",
        doc_title=title,
        issuer="RBI",
        doc_type="master_direction",
        page_start=page_start,
        page_end=page_end if page_end is not None else page_start,
        clause_path=None,
        section_title=None,
        issue_date=None,
        effective_date=None,
        score=1.0,
    )


def test_recall_counts_golds_found_in_top_k() -> None:
    retrieved = [chunk("a", "doc1", 5), chunk("b", "doc2", 9)]
    golds = [("doc1", 5), ("doc2", 9), ("doc3", 1)]
    assert recall_at_k(golds, retrieved, k=2) == 2 / 3
    assert recall_at_k(golds, retrieved, k=1) == 1 / 3


def test_recall_is_none_without_golds() -> None:
    assert recall_at_k([], [chunk("a", "doc1", 1)], k=5) is None


def test_recall_page_range_is_inclusive() -> None:
    retrieved = [chunk("a", "doc1", 4, 8)]
    assert recall_at_k([("doc1", 8)], retrieved, k=1) == 1.0
    assert recall_at_k([("doc1", 9)], retrieved, k=1) == 0.0


def test_clause_only_gold_matches_same_document() -> None:
    retrieved = [chunk("a", "doc1", None)]
    assert recall_at_k([("doc1", None)], retrieved, k=1) == 1.0
    assert recall_at_k([("other", None)], retrieved, k=1) == 0.0


def test_reciprocal_rank_first_hit_position() -> None:
    retrieved = [chunk("a", "docX", 1), chunk("b", "doc1", 5), chunk("c", "doc1", 6)]
    assert reciprocal_rank([("doc1", 5)], retrieved) == 0.5
    assert reciprocal_rank([("doc1", 6)], retrieved) == 1 / 3


def test_reciprocal_rank_zero_when_nothing_hits() -> None:
    retrieved = [chunk("a", "docX", 1)]
    assert reciprocal_rank([("doc1", 5)], retrieved) == 0.0
    assert reciprocal_rank([], retrieved) is None


def test_ndcg_perfect_order_is_one() -> None:
    retrieved = [chunk("a", "doc1", 5), chunk("b", "docX", 1)]
    assert ndcg_at_k([("doc1", 5)], retrieved, k=2) == 1.0


def test_ndcg_relevant_second_is_worse() -> None:
    golds = [("doc1", 5)]
    good = [chunk("a", "doc1", 5), chunk("b", "docX", 1)]
    bad = [chunk("a", "docX", 1), chunk("b", "doc1", 5)]
    good_score = ndcg_at_k(golds, good, k=2)
    bad_score = ndcg_at_k(golds, bad, k=2)
    assert good_score is not None and bad_score is not None
    assert good_score > bad_score
    assert bad_score > 0.0


def test_ndcg_none_without_golds_and_zero_when_no_relevant() -> None:
    assert ndcg_at_k([], [chunk("a", "doc1", 1)], k=5) is None
    score = ndcg_at_k([("doc1", 5)], [chunk("a", "docX", 1)], k=1)
    assert score == 0.0


def test_citation_precision_resolves_by_title_and_page() -> None:
    retrieved = [chunk("a", "doc1", 5, 9, title="KYC Direction")]
    labels = ["[KYC Direction, p.7, 4.2]", "[KYC Direction, p.12, 5.0]", "[Unknown Doc, p.7, x]"]
    score = score_citations(labels, retrieved)
    assert score.total == 3
    assert score.resolved == 1
    assert score.precision == 1 / 3


def test_citation_precision_none_when_no_citations() -> None:
    score = score_citations([], [])
    assert score.total == 0
    assert score.precision is None


def test_mean_ignores_none_and_returns_none_when_empty() -> None:
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([]) is None
    assert mean([None, None]) is None


def test_summarize_separates_answerable_and_reports_missing_as_none() -> None:
    records = [
        {
            "gold_count": 2,
            "recall": 0.5,
            "rr": 1.0,
            "ndcg": 1.0,
            "refused": False,
            "citation_precision": 1.0,
            "has_citations": True,
        },
        {
            "gold_count": 1,
            "recall": None,  # failed retrieval scoring must not become 0
            "rr": None,
            "ndcg": None,
            "refused": True,
            "citation_precision": None,
            "has_citations": False,
        },
        {
            "gold_count": 0,
            "recall": None,
            "rr": None,
            "ndcg": None,
            "refused": True,
            "citation_precision": None,
            "has_citations": False,
        },
    ]
    result = summarize(records, k=10)
    assert result["questions"] == 3
    assert result["answerable"] == 2
    assert result["unanswerable"] == 1
    # mean over defined values only: 0.5, not 0.25
    assert result["retrieval"]["recall_at_k"] == 0.5
    assert result["retrieval"]["recall_at_k_defined"] == 1
    assert result["generation"]["correct_refusals"] == 1
    assert result["generation"]["citation_precision"] == 1.0
    assert result["generation"]["refusal_rate"] == 2 / 3


def test_summarize_reports_none_when_nothing_is_defined() -> None:
    result = summarize([], k=10)
    assert result["questions"] == 0
    assert result["retrieval"]["recall_at_k"] is None
    assert result["generation"]["citation_precision"] is None
    assert result["generation"]["refusal_rate"] is None
