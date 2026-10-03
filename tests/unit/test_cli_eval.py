"""CLI eval and reindex commands: preflight gates, not the pipeline itself.

Refusals that must hold before any tokens are spent: no questions, unreviewed drafts
without an explicit flag, a missing API key, and (reindex) a purge that never runs on a
dry run or without credentials. The pipeline and scoring are covered by
tests/unit/test_eval_run.py; this file only pins the gates.
"""

from __future__ import annotations

import json

import pytest
from eval.runners.golden import GoldenQuestion

from reglens.cli import EXIT_ERROR, EXIT_INVALID, EXIT_OK, main


def draft_question() -> GoldenQuestion:
    return GoldenQuestion.model_validate(
        {
            "id": "q0001",
            "type": "lookup",
            "question": "What is the minimum KYC interval for low-risk accounts?",
            "reference_answer": "Every 10 years.",
            "gold_passages": [{"doc_id": "rbi_md_kyc_2016", "page": 30, "clause": "4.4"}],
            "review": "draft",
        }
    )


def test_eval_refuses_when_golden_set_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eval.runners.golden.load_golden", lambda *_a, **_k: [])
    assert main(["eval"]) == EXIT_INVALID


def test_eval_refuses_drafts_without_allow_drafts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("eval.runners.golden.load_golden", lambda *_a, **_k: [draft_question()])
    assert main(["eval"]) == EXIT_INVALID
    assert "--allow-drafts" in capsys.readouterr().err


def test_eval_dry_run_spends_nothing_with_allow_drafts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("eval.runners.golden.load_golden", lambda *_a, **_k: [draft_question()])

    def must_not_build(*_a, **_k: object) -> None:
        raise AssertionError("dry run must not build a pipeline")

    monkeypatch.setattr("reglens.routing.pipeline.build_pipeline", must_not_build)
    assert main(["eval", "--allow-drafts"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "draft=1" in out


def test_eval_yes_without_api_key_fails_before_touching_the_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("eval.runners.golden.load_golden", lambda *_a, **_k: [draft_question()])

    def must_not_connect(*_a: object, **_k: object) -> None:
        raise AssertionError("missing key must be detected before any DB connection")

    monkeypatch.setattr("reglens.indexing.vector_store.PgVectorStore", must_not_connect)
    # conftest strips GEMINI_API_KEY/GOOGLE_API_KEY and points REGLENS_ENV_FILE nowhere
    code = main(["eval", "--yes", "--allow-drafts"])
    assert code == EXIT_ERROR
    assert "GEMINI_API_KEY" in capsys.readouterr().err


class FakeStore:
    """Count-only store; any purge attempt fails the test unless expected."""

    def __init__(self, chunks: int = 0, *, allow_purge: bool = False) -> None:
        self.chunks = chunks
        self.allow_purge = allow_purge
        self.purged = 0

    def count_chunks(self) -> int:
        return self.chunks

    def purge_chunks(self) -> int:
        if not self.allow_purge:
            raise AssertionError("purge must not run on a dry run or without credentials")
        self.purged = self.chunks
        return self.purged


def test_reindex_dry_run_never_purges(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FakeStore(42)
    monkeypatch.setattr("reglens.indexing.vector_store.PgVectorStore", lambda **_k: store)
    assert main(["reindex"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "42 chunk(s)" in out
    assert store.purged == 0


def test_reindex_yes_without_api_key_never_purges(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FakeStore(42)
    monkeypatch.setattr("reglens.indexing.vector_store.PgVectorStore", lambda **_k: store)
    # conftest strips both Gemini env keys and points REGLENS_ENV_FILE nowhere
    assert main(["reindex", "--yes"]) == EXIT_ERROR
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert store.purged == 0


def test_reindex_unreachable_database_is_a_preflight_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class DeadStore:
        def __init__(self, **_k: object) -> None:
            pass

        def count_chunks(self) -> int:
            raise RuntimeError("connection refused")

    monkeypatch.setattr("reglens.indexing.vector_store.PgVectorStore", DeadStore)
    assert main(["reindex"]) == EXIT_ERROR
    assert "unreachable" in capsys.readouterr().err


def test_reindex_dimension_mismatch_never_purges(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A purge followed by a failed ingest would lose the index for nothing."""

    class MismatchedStore(FakeStore):
        def embedding_dimension(self) -> int | None:
            return 1

    store = MismatchedStore(42, allow_purge=True)
    monkeypatch.setattr("reglens.indexing.vector_store.PgVectorStore", lambda **_k: store)
    assert main(["reindex", "--yes"]) == EXIT_ERROR
    assert "vector(1)" in capsys.readouterr().err
    assert store.purged == 0


# ------------------------------------------------------------------ eval --agree-with
def judged_report(report_id: str) -> dict:
    """A minimal-but-complete run report: enough envelope for render_markdown."""
    records = [
        {
            "id": "q0001",
            "type": "lookup",
            "recall": 1.0,
            "rr": 1.0,
            "ndcg": 1.0,
            "refused": False,
            "citation_precision": 1.0,
            "judge": {"correctness": 2, "faithfulness": 1, "notes": "", "error": None},
        },
        {
            "id": "q0002",
            "type": "lookup",
            "recall": 1.0,
            "rr": 1.0,
            "ndcg": 1.0,
            "refused": True,
            "citation_precision": None,
            "judge": {"correctness": 0, "faithfulness": 2, "notes": "", "error": None},
        },
    ]
    return {
        "kind": "golden-eval",
        "schema_version": 1,
        "generated_at": "2026-10-03T12:00:00+00:00",
        "report_id": report_id,
        "git_commit": "0123456789abcdef",
        "config_hash": "abcdef123456",
        "experiment": "baseline_naive",
        "corpus_version": "0.1.0",
        "models": {"embedding": "gemini-embedding-001", "generation": "g", "judge": "j"},
        "golden": {
            "total": 2,
            "counts_by_type": {"lookup": 2},
            "reviewed": 2,
            "draft": 0,
            "held_out": 0,
        },
        "judge": {
            "enabled": True,
            "prompt_version": "judge-v1",
            "model": "j",
            "agreement": None,
        },
        "aggregates": {
            "questions": 2,
            "answerable": 2,
            "unanswerable": 0,
            "retrieval": {
                "k": 10,
                "recall_at_k": 1.0,
                "recall_at_k_defined": 2,
                "mrr": 1.0,
                "mrr_defined": 2,
                "ndcg_at_k": 1.0,
                "ndcg_at_k_defined": 2,
            },
            "generation": {
                "citation_precision": 1.0,
                "citation_precision_defined": 1,
                "refusal_rate": 0.5,
                "answers_with_citations": 1,
                "correct_refusals": 0,
            },
        },
        "totals": {
            "embed_input_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
            "judge_input_tokens": 0,
            "judge_output_tokens": 0,
            "estimated_cost_usd": None,
        },
        "failures": [],
        "caveats": [],
        "questions": records,
    }


def test_eval_agree_requires_human_grades(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    report_path = tmp_path / "r.json"
    report_path.write_text("{}", encoding="utf-8")
    assert main(["eval", "--agree-with", str(report_path)]) == EXIT_INVALID
    assert "--human-grades" in capsys.readouterr().err


def test_eval_agree_attaches_agreement_without_spending(tmp_path, capsys) -> None:
    report_id = "2026-10-03T120000Z_baseline_naive_abcdef123456"
    report_path = tmp_path / f"{report_id}.json"
    report_path.write_text(json.dumps(judged_report(report_id)), encoding="utf-8")
    grades = tmp_path / "grades.jsonl"
    grades.write_text(
        '{"question_id": "q0001", "correctness": 2, "faithfulness": 2}\n'
        '{"question_id": "q0002", "correctness": 1, "faithfulness": 2}\n',
        encoding="utf-8",
    )

    # The post-hoc path must not require a golden set, an index, or an API key.
    code = main(
        ["eval", "--agree-with", str(report_path), "--human-grades", str(grades)]
    )
    assert code == EXIT_OK

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    section = payload["judge"]["agreement"]
    assert section["paired"] == 2
    # judge F: q0001 1-vs-2 misses, q0002 2-vs-2 hits; correctness: one hit, one miss
    assert section["exact_agreement"] == {"correctness": 0.5, "faithfulness": 0.5}
    markdown = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "human agreement (paired=2)" in markdown
    assert "agreement attached" in capsys.readouterr().out


def test_eval_agree_rejects_a_file_that_is_not_a_run_report(tmp_path, capsys) -> None:
    report_path = tmp_path / "notes.json"
    report_path.write_text(json.dumps({"report_id": "something_else"}), encoding="utf-8")
    grades = tmp_path / "grades.jsonl"
    grades.write_text(
        '{"question_id": "q0001", "correctness": 2, "faithfulness": 2}\n', encoding="utf-8"
    )
    code = main(
        ["eval", "--agree-with", str(report_path), "--human-grades", str(grades)]
    )
    assert code == EXIT_INVALID
    assert "report_id" in capsys.readouterr().err
