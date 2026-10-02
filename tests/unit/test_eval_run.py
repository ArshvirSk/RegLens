"""Run-eval tests: the runner's contract, with a fake pipeline and fake judge.

Covers what makes a report trustworthy: per-question failures don't sink the run,
failed questions are excluded (missing ≠ 0), and both report files carry the same
versioned id.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from eval.runners.golden import GoldenQuestion, write_golden
from eval.runners.judge import JudgeScore
from eval.runners.run_eval import (
    build_report,
    evaluate,
    render_markdown,
    write_report,
)

from reglens.config import get_settings, load_experiment
from reglens.generation.base import Answer
from reglens.retrieval.base import RetrievedChunk


def question(**overrides: object) -> GoldenQuestion:
    payload: dict[str, object] = {
        "id": "q0001",
        "type": "lookup",
        "question": "What is the minimum KYC periodic-updating interval for low-risk accounts?",
        "reference_answer": "Every 10 years for accounts classified as low risk.",
        "gold_passages": [{"doc_id": "rbi_md_kyc_2016", "page": 30, "clause": "4.4"}],
        "notes": "10 years exactly.",
    }
    payload.update(overrides)
    return GoldenQuestion.model_validate(payload)


def make_chunk(
    chunk_id: str,
    doc_id: str,
    page_start: int,
    page_end: int | None = None,
    title: str = "KYC Direction",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="context text",
        doc_title=title,
        issuer="RBI",
        doc_type="master_direction",
        page_start=page_start,
        page_end=page_end or page_start,
        clause_path="4.4",
        section_title=None,
        issue_date=None,
        effective_date=None,
        score=0.9,
    )


def make_ask(
    chunks: list[RetrievedChunk], *, refused: bool = False, citations: list[str] | None = None
):
    """A pipeline.ask-shaped callable taking (question, top_k=, filters=)."""

    def ask(q_text: str, *, top_k: int | None = None, filters=None) -> SimpleNamespace:
        answer_text = (
            "INSUFFICIENT EVIDENCE"
            if refused
            else "Low-risk accounts are updated every 10 years [KYC Direction, p.30, 4.4]."
        )
        return SimpleNamespace(
            answer=Answer(
                text=answer_text,
                citations=[] if refused else (citations or ["[KYC Direction, p.30, 4.4]"]),
                refused=refused,
                model="fake-model",
                input_tokens=80,
                output_tokens=40,
                latency_ms=12,
            ),
            chunks=[] if refused else chunks,
            embed_input_tokens=33,
            embed_model="fake-embed",
        )

    return ask


def test_evaluate_scores_retrieval_and_citations() -> None:
    questions = [question()]
    chunks = [make_chunk("c1", "rbi_md_kyc_2016", 29, 31)]
    records, failures = evaluate(
        questions,
        lambda q: make_ask(chunks)(q.question),
        k=10,
        experiment=load_experiment("baseline_naive"),
    )
    assert failures == []
    assert len(records) == 1
    record = records[0]
    assert record["recall"] == 1.0
    assert record["rr"] == 1.0
    assert record["ndcg"] == 1.0
    assert record["citation_precision"] == 1.0
    assert record["embed_input_tokens"] == 33
    assert record["judge"] is None


def test_evaluate_records_failure_and_continues() -> None:
    good = question(id="q0001")
    bad = question(id="q0002")

    def ask(q: GoldenQuestion):
        if q.id == "q0002":
            raise RuntimeError("database gone")
        return make_ask([make_chunk("c1", "rbi_md_kyc_2016", 30)])(q.question)

    records, failures = evaluate(
        [good, bad], ask, k=10, experiment=load_experiment("baseline_naive")
    )
    assert [record["id"] for record in records] == ["q0001"]
    assert len(failures) == 1
    assert failures[0]["id"] == "q0002"
    assert "database gone" in failures[0]["error"]


def test_evaluate_attaches_judge_scores_when_given() -> None:
    class FakeJudge:
        def judge(self, **kwargs) -> JudgeScore:
            return JudgeScore(correctness=2, faithfulness=1, notes="ok", model="j-model")

    records, _ = evaluate(
        [question()],
        lambda q: make_ask([make_chunk("c1", "rbi_md_kyc_2016", 30)])(q.question),
        k=10,
        experiment=load_experiment("baseline_naive"),
        judge=FakeJudge(),
    )
    judged = records[0]["judge"]
    assert judged["correctness"] == 2 and judged["faithfulness"] == 1
    assert judged["prompt_version"] == "judge-v1"


def test_build_report_envelope_carries_repro_fields_and_excludes_failures() -> None:
    settings = get_settings()
    experiment = load_experiment("baseline_naive")
    questions = [question()]
    records, failures = evaluate(
        questions,
        lambda q: make_ask([make_chunk("c1", "rbi_md_kyc_2016", 30)])(q.question),
        k=10,
        experiment=experiment,
    )
    failures = [{"id": "q0002", "error": "boom"}]
    report = build_report(
        records=records,
        failures=failures,
        questions=questions,
        experiment=experiment,
        settings=settings,
        judge_enabled=False,
        generated_at=datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC),
    )
    assert report["kind"] == "golden-eval"
    assert report["report_id"].startswith("2026-10-03T120000Z_baseline_naive_")
    assert report["config_hash"] == experiment.fingerprint()
    assert report["corpus_version"] == settings.corpus_version
    assert report["git_commit"]
    assert report["golden"]["total"] == 1
    assert report["aggregates"]["questions"] == 1  # failure excluded, not counted as 0
    assert report["failures"] == failures
    assert any("drafts" in caveat for caveat in report["caveats"])
    assert any("failed" in caveat for caveat in report["caveats"])


def test_build_report_includes_agreement_when_grades_supplied() -> None:
    from eval.runners.judge import HumanGrade

    settings = get_settings()
    experiment = load_experiment("baseline_naive")
    questions = [question()]
    records, _ = evaluate(
        questions,
        lambda q: make_ask([make_chunk("c1", "rbi_md_kyc_2016", 30)])(q.question),
        k=10,
        experiment=experiment,
        judge=type(
            "J",
            (),
            {"judge": lambda self, **kw: JudgeScore(correctness=2, faithfulness=2)},
        )(),
    )
    report = build_report(
        records=records,
        failures=[],
        questions=questions,
        experiment=experiment,
        settings=settings,
        judge_enabled=True,
        human_grades=[HumanGrade("q0001", correctness=2, faithfulness=1)],
    )
    agreement_section = report["judge"]["agreement"]
    assert agreement_section is not None
    assert agreement_section["paired"] == 1
    assert agreement_section["exact_agreement"]["correctness"] == 1.0
    assert agreement_section["exact_agreement"]["faithfulness"] == 0.0


def test_run_eval_writes_matching_json_and_md(tmp_path) -> None:
    settings = get_settings()
    experiment = load_experiment("baseline_naive")
    questions = [question()]
    records, failures = evaluate(
        questions,
        lambda q: make_ask([make_chunk("c1", "rbi_md_kyc_2016", 30)])(q.question),
        k=10,
        experiment=experiment,
    )
    report = build_report(
        records=records,
        failures=failures,
        questions=questions,
        experiment=experiment,
        settings=settings,
        judge_enabled=False,
        generated_at=datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC),
    )
    # write into tmp: eval/results is the project's immutable public record
    json_path, md_path = write_report(report, tmp_path)
    assert json_path.name == md_path.name.replace(".md", ".json")
    assert json_path.name == f"{report['report_id']}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_id"] == report["report_id"]
    markdown = md_path.read_text(encoding="utf-8")
    assert report["report_id"] in markdown
    assert "| recall@10 | 1.0000 |" in markdown


def test_render_markdown_reports_missing_as_missing() -> None:
    settings = get_settings()
    experiment = load_experiment("baseline_naive")
    unanswerable = question(
        id="q0001", type="unanswerable", gold_passages=[], reference_answer="Refuse."
    )
    records, failures = evaluate(
        [unanswerable],
        lambda q: make_ask([], refused=True)(q.question),
        k=10,
        experiment=experiment,
    )
    report = build_report(
        records=records,
        failures=failures,
        questions=[unanswerable],
        experiment=experiment,
        settings=settings,
        judge_enabled=False,
        generated_at=datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC),
    )
    markdown = render_markdown(report)
    assert "| recall@10 | missing |" in markdown
    assert "| MRR | missing |" in markdown
    # correct refusal is counted, not averaged away
    assert "correct refusals (of 1 unanswerable)" in markdown


def test_golden_roundtrip_for_runner(tmp_path) -> None:
    questions = [question()]
    path = write_golden(questions, tmp_path / "questions.jsonl")
    assert path.read_text(encoding="utf-8").strip().startswith('{"id": "q0001"')
