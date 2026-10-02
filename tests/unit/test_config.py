"""Settings and experiment-config tests.

Two things here matter more than they look:

* the config hash must be stable across processes and change when behaviour changes, or
  "which config produced this number?" is unanswerable;
* secrets must never appear in a settings dump, because that dump is logged and served.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reglens.config import Settings, config_hash, get_settings, list_experiments, load_experiment
from reglens.config.experiment import ExperimentConfig


def test_defaults_match_the_env_example(clean_env: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.env == "local"
    assert settings.seed == 1337
    assert settings.corpus_version == "0.1.0"
    assert settings.experiment_config == "baseline_naive"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.llm_model_small == "gemini-3.5-flash-lite"
    assert settings.llm_model_large == "gemini-3.6-flash"
    assert settings.embedding_dim == 3072
    assert settings.fetch_delay_seconds == 2.0
    assert settings.fetch_respect_robots is True


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGLENS_CORPUS_VERSION", "0.2.0")
    monkeypatch.setenv("REGLENS_SEED", "7")
    settings = Settings(_env_file=None)
    assert settings.corpus_version == "0.2.0"
    assert settings.seed == 7


def test_cors_origins_accept_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGLENS_CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]


def test_log_level_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGLENS_LOG_LEVEL", "chatty")
    with pytest.raises(ValueError, match="log_level"):
        Settings(_env_file=None)


def test_paths_are_absolute(clean_env: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.raw_dir.is_absolute()
    assert settings.manifest_path.is_absolute()
    assert settings.raw_dir.name == "raw"


def test_secrets_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret")
    settings = Settings(_env_file=None)
    dumped = settings.redacted()
    assert dumped["openai_api_key"] == "***"
    assert "sk-super-secret-value" not in str(dumped)
    assert "lf-secret" not in str(dumped)


def test_database_url_password_is_redacted(clean_env: None) -> None:
    settings = Settings(_env_file=None)
    dumped = settings.redacted()
    assert "reglens_local_dev" not in dumped["database_url"]
    assert dumped["database_url"].startswith("postgresql://reglens:***@")


def test_has_openai_key_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).has_openai_key is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert Settings(_env_file=None).has_openai_key is True


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


# ------------------------------------------------------------------ experiment config
def test_baseline_config_is_deliberately_naive() -> None:
    config = load_experiment("baseline_naive")
    assert config.chunking.strategy == "fixed"
    assert config.retrieval.mode == "dense"
    assert config.retrieval.rerank is False
    assert config.retrieval.fusion == "none"
    assert config.rewriting.enabled is False
    assert config.indexing.keyword_index == "none"


def test_phase2_config_turns_the_toggles_on() -> None:
    config = load_experiment("phase2_hybrid_rerank")
    assert config.retrieval.mode == "hybrid"
    assert config.retrieval.fusion == "rrf"
    assert config.retrieval.rerank is True
    assert config.chunking.contextual_headers is True


def test_experiment_hash_is_stable_and_behaviour_sensitive() -> None:
    config = load_experiment("baseline_naive")
    again = load_experiment("baseline_naive")
    assert config_hash(config) == config_hash(again)

    changed = config.model_copy(deep=True)
    changed.retrieval.top_k += 1
    assert config_hash(changed) != config_hash(config)


def test_prose_fields_do_not_change_the_hash() -> None:
    """Documentation edits must not invalidate recorded results."""
    config = load_experiment("baseline_naive")
    annotated = config.model_copy(deep=True)
    annotated.description = "a better description"
    annotated.notes = ["note"]
    assert config_hash(annotated) == config_hash(config)


def test_unknown_experiment_lists_available_names() -> None:
    with pytest.raises(FileNotFoundError, match="baseline_naive"):
        load_experiment("nope")


def test_load_experiment_by_path() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "src/reglens/config/experiments/baseline_naive.yaml"
    )
    assert load_experiment(path).name == "baseline_naive"


def test_summary_lines_cover_every_stage() -> None:
    lines = load_experiment("baseline_naive").summary_lines()
    assert len(lines) == 6
    assert any("parsing" in line for line in lines)
    assert any("chunking" in line for line in lines)
    assert any("embeddings" in line for line in lines)
    assert any("generation" in line for line in lines)


def test_shipped_experiments_are_valid() -> None:
    names = list_experiments()
    assert "baseline_naive" in names
    for name in names:
        ExperimentConfig.model_validate(load_experiment(name).model_dump())
