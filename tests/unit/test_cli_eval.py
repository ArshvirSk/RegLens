"""CLI eval and reindex commands: preflight gates, not the pipeline itself.

Refusals that must hold before any tokens are spent: no questions, unreviewed drafts
without an explicit flag, a missing API key, and (reindex) a purge that never runs on a
dry run or without credentials. The pipeline and scoring are covered by
tests/unit/test_eval_run.py; this file only pins the gates.
"""

from __future__ import annotations

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
