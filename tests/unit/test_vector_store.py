"""Vector-store helpers: the SQL-adjacent logic, tested without a database.

The database-facing methods are exercised by the integration suite; what is tested
here is everything that can be wrong while still being *runnable*: parameter numbering,
vector serialisation and result shaping.
"""

from __future__ import annotations

from datetime import date

import pytest

from reglens.indexing.vector_store import (
    build_filter_clause,
    chunks_from_rows,
    hits_from_rows,
    vector_literal,
)
from reglens.retrieval.base import SearchFilters


def test_vector_literal_round_trips_exactly() -> None:
    values = [0.1, -0.25, 3.0, 1e-07]
    literal = vector_literal(values)
    assert literal == "[0.1,-0.25,3.0,1e-07]"
    assert [float(part) for part in literal[1:-1].split(",")] == values


def test_vector_literal_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="empty"):
        vector_literal([])
    with pytest.raises(ValueError, match="NaN"):
        vector_literal([1.0, float("nan")])
    with pytest.raises(ValueError, match="NaN"):
        vector_literal([float("inf")])


def test_no_filters_keeps_the_corpus_predicate() -> None:
    clause, params = build_filter_clause(None)
    assert clause == "c.corpus_version = $2"
    assert params == []


def test_metadata_filters_bind_from_position_four() -> None:
    clause, params = build_filter_clause(SearchFilters(issuer="RBI", doc_type="master_direction"))
    assert params == ["RBI", "master_direction"]
    # $1 vector, $2 corpus_version, $3 top_k — filters must start at $4.
    assert "d.issuer = $4" in clause
    assert "d.doc_type = $5" in clause
    assert "c.corpus_version = $2" in clause


def test_doc_ids_and_as_of_date_are_bound_safely() -> None:
    filters = SearchFilters(
        doc_ids=("rbi_md_a", "rbi_md_b"),
        as_of_date=date(2026, 1, 1),
        fiscal_period="FY2025",
    )
    clause, params = build_filter_clause(filters)
    # Binding order: fiscal_period, then doc_ids, then as_of_date — starting at $4.
    assert "d.fiscal_period = $4" in clause
    assert "d.doc_id = ANY($5::text[])" in clause
    assert "(d.issue_date <= $6 OR d.issue_date IS NULL)" in clause
    assert params == ["FY2025", ["rbi_md_a", "rbi_md_b"], date(2026, 1, 1)]


def test_empty_doc_ids_filter_is_ignored() -> None:
    clause, params = build_filter_clause(SearchFilters(doc_ids=()))
    assert "ANY" not in clause
    assert params == []


def test_results_are_shaped_into_plain_data() -> None:
    hits = hits_from_rows([{"chunk_id": "c1", "score": "0.75"}, {"chunk_id": "c2", "score": 0.5}])
    assert [hit.chunk_id for hit in hits] == ["c1", "c2"]
    assert hits[0].score == 0.75

    rows = chunks_from_rows([{"chunk_id": "c1", "text": "hello"}])
    assert rows == [{"chunk_id": "c1", "text": "hello"}]
