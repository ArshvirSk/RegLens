"""Golden-set tests.

The eval set is the measuring instrument, so its rules are tested like code: ids are
stable, gold passages must point at real manifest documents, unanswerable questions carry
no gold passages, and held-out questions are tracked so they cannot be tuned on by accident.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from eval.runners.golden import (
    QUESTION_TYPE_QUOTAS,
    GoldenQuestion,
    load_golden,
    validate_golden,
    write_golden,
)


def question(**overrides: object) -> GoldenQuestion:
    payload: dict[str, object] = {
        "id": "q0001",
        "type": "lookup",
        "question": "What is the minimum liquidity coverage ratio for scheduled commercial banks?",
        "reference_answer": "At least 100% of net cash outflows over a 30-day stress period.",
        "gold_passages": [{"doc_id": "rbi_md_alm_2025", "page": 12, "clause": "4.2"}],
        "notes": "Exact threshold and the 30-day window are both required.",
    }
    payload.update(overrides)
    return GoldenQuestion.model_validate(payload)


def test_valid_question_passes() -> None:
    report = validate_golden([question()])
    assert report.ok
    assert report.counts_by_type == {"lookup": 1}


def test_id_format_is_enforced() -> None:
    with pytest.raises(ValueError, match="q0001"):
        question(id="question-1")


def test_duplicate_ids_are_errors() -> None:
    report = validate_golden([question(), question()])
    assert not report.ok
    assert any("duplicate id" in issue.message for issue in report.errors)


def test_answerable_question_requires_a_gold_passage() -> None:
    report = validate_golden([question(gold_passages=[])])
    assert not report.ok
    assert any("gold passage" in issue.message for issue in report.errors)


def test_unanswerable_question_must_not_have_gold_passages() -> None:
    report = validate_golden(
        [
            question(
                id="q0002",
                type="unanswerable",
                question="What will SBI's gross NPA ratio be at the end of FY2031?",
                reference_answer="The corpus contains no such figure; a refusal is correct.",
                gold_passages=[{"doc_id": "rbi_md_alm_2025", "page": 1}],
            )
        ]
    )
    assert report.ok
    assert any("lists gold passages" in issue.message for issue in report.warnings)


def test_gold_passage_needs_page_or_clause() -> None:
    report = validate_golden([question(gold_passages=[{"doc_id": "rbi_md_alm_2025"}])])
    assert not report.ok


def test_gold_passage_doc_id_must_exist_in_the_manifest() -> None:
    report = validate_golden(
        [question(gold_passages=[{"doc_id": "not_in_manifest", "page": 3}])],
        manifest_ids={"rbi_md_alm_2025"},
    )
    assert not report.ok
    assert any("not in the manifest" in issue.message for issue in report.errors)


def test_temporal_question_should_declare_an_as_of_date() -> None:
    report = validate_golden(
        [question(type="temporal", as_of_date=None, notes="asks about the rule in force in 2023")]
    )
    assert any("as_of_date" in issue.message for issue in report.warnings)


def test_temporal_question_with_as_of_date_is_clean() -> None:
    report = validate_golden(
        [
            question(
                id="q0003",
                type="temporal",
                as_of_date=date(2023, 6, 30),
                notes="Rule as it stood on 30 June 2023.",
            )
        ]
    )
    assert not any("as_of_date" in issue.message for issue in report.warnings)


def test_numeric_questions_should_record_grading_notes() -> None:
    report = validate_golden([question(type="numeric", notes="")])
    assert any("grading notes" in issue.message for issue in report.warnings)


def test_draft_and_unchecked_questions_are_flagged() -> None:
    report = validate_golden([question()])
    messages = " ".join(issue.message for issue in report.warnings)
    assert "awaiting owner review" in messages
    assert "not yet checked against the source PDF" in messages


def test_minimum_question_count_is_warned_about() -> None:
    report = validate_golden([question()])
    assert any("required for this phase" in issue.message for issue in report.warnings)


def test_full_quota_check_reports_gaps() -> None:
    report = validate_golden([question()], require_full_minimum=True)
    assert any("quota" in issue.message for issue in report.warnings)


def test_quota_targets_match_the_prd() -> None:
    assert QUESTION_TYPE_QUOTAS == {
        "lookup": 25,
        "numeric": 20,
        "temporal": 20,
        "comparison": 15,
        "multi_hop": 10,
        "unanswerable": 10,
    }
    assert sum(QUESTION_TYPE_QUOTAS.values()) == 100


def test_review_and_held_out_counts() -> None:
    questions = [
        question(),
        question(id="q0004", review="reviewed", held_out=True),
    ]
    report = validate_golden(questions)
    assert report.held_out_count == 1
    assert report.reviewed_count == 1
    assert report.total == 2


def test_roundtrip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    write_golden([question(), question(id="q0005", type="unanswerable", gold_passages=[])], path)
    loaded = load_golden(path)
    assert [q.id for q in loaded] == ["q0001", "q0005"]
    assert loaded[1].type == "unanswerable"


def test_load_golden_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": "q0001"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"questions\.jsonl:1"):
        load_golden(path)


def test_empty_golden_file_loads_as_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert load_golden(path) == []
