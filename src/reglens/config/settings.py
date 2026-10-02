"""Typed application settings.

Priority order (highest first): explicit init kwargs, environment variables, then the
``.env`` file at the project root. Variable names are documented in ``.env.example``;
nothing is ever hard-coded here that could be a secret.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from reglens.config import paths

EnvName = Literal["local", "test", "ci", "docker", "prod"]


class MissingApiKeyError(RuntimeError):
    """No API key for the configured provider: fail loudly before any pipeline runs,
    never mid-eval with half the questions answered."""


class Settings(BaseSettings):
    """Runtime configuration for every RegLens process (API, scripts, eval)."""

    model_config = SettingsConfigDict(
        # `_env_file` is injected in get_settings() so the .env file is found relative
        # to the project root, not to whatever directory the process started in.
        env_file=None,
        env_prefix="REGLENS_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- runtime ---
    env: EnvName = "local"
    log_level: str = "INFO"
    seed: int = 1337
    corpus_version: str = "0.1.0"
    experiment_config: str = "baseline_naive"

    # --- api ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # NoDecode: the comma-separated form in .env must not be JSON-decoded before the
    # validator below runs.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- storage ---
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    derived_dir: Path = Path("data/derived")

    # --- database ---
    database_url: str = "postgresql://reglens:reglens_local_dev@localhost:5433/reglens"
    readonly_database_url: str = (
        "postgresql://reglens_ro:reglens_ro_local_dev@localhost:5433/reglens"
    )
    # Neon requires TLS; a plain local Postgres needs REGLENS_DB_SSL=disable. An sslmode=
    # query parameter in the URL always wins (asyncpg parses it from the DSN directly).
    db_ssl: Literal["require", "disable"] = "require"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_command_timeout_seconds: float = 30.0

    # --- models (used from Phase 1 onwards; declared here so configs stay swappable) ---
    # Phase 1 baseline pins the cheapest stable Gemini picks (owner decision, 2026-10-02):
    # gemini-embedding-001 (GA, 3072-dim) + gemini-3.5-flash-lite + judge gemini-3.6-flash.
    embedding_provider: Literal["gemini", "openai", "sentence_transformers"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    llm_provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    llm_model_small: str = "gemini-3.5-flash-lite"
    llm_model_large: str = "gemini-3.6-flash"
    llm_temperature: float = 0.0
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- observability ---
    tracing_enabled: bool = True
    trace_db_write: bool = True
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    # --- secrets ---
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # --- fetcher politeness ---
    user_agent: str = "RegLens/0.1 (research; contact: you@example.com)"
    fetch_delay_seconds: float = 2.0
    fetch_timeout_seconds: float = 60.0
    fetch_max_bytes: int = 50 * 1024 * 1024
    fetch_respect_robots: bool = True

    # ------------------------------------------------------------------ validators
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``a,b`` from the environment as well as a JSON list or a real list.

        pydantic-settings tries ``json.loads`` on complex field types before validators
        run, so a bare ``http://a,http://b`` string raises before reaching this method.
        The input is therefore unwrapped from a JSON string first when it is not a list.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Make every directory absolute so behaviour does not depend on the CWD."""
        self.data_dir = paths.resolve(self.data_dir)
        self.raw_dir = paths.resolve(self.raw_dir)
        self.parsed_dir = paths.resolve(self.parsed_dir)
        self.derived_dir = paths.resolve(self.derived_dir)
        return self

    # ------------------------------------------------------------------ derived
    # Corpus paths hang off ``data_dir`` so that pointing REGLENS_DATA_DIR at a temp
    # directory relocates the whole corpus (manifest included). Tests rely on this: when
    # the manifest lived at a fixed project path, a test run overwrote the real one.
    @property
    def manifest_path(self) -> Path:
        return self.data_dir / Path(paths.MANIFEST_PATH).name

    @property
    def manifest_schema_path(self) -> Path:
        return self.data_dir / Path(paths.MANIFEST_SCHEMA_PATH).name

    @property
    def corpus_plan_path(self) -> Path:
        return self.data_dir / Path(paths.CORPUS_PLAN_PATH).name

    # Eval artefacts stay at fixed project paths: reports are the project's public record
    # and must not move with a local data-directory override.
    @property
    def golden_path(self) -> Path:
        return paths.resolve(paths.GOLDEN_PATH)

    @property
    def eval_results_dir(self) -> Path:
        return paths.resolve(paths.EVAL_RESULTS_DIR)

    @property
    def experiment_config_dir(self) -> Path:
        return paths.resolve(paths.EXPERIMENT_CONFIG_DIR)

    @property
    def migrations_dir(self) -> Path:
        return paths.resolve(paths.MIGRATIONS_DIR)

    @property
    def has_openai_key(self) -> bool:
        return self.openai_api_key is not None

    @property
    def has_gemini_key(self) -> bool:
        return self.gemini_api_key is not None

    def require_gemini_key(self) -> str:
        """A usable Gemini key (possibly ``""`` = let the SDK read the environment).

        Raises :class:`MissingApiKeyError` when neither ``.env`` nor the process
        environment has one, so a pipeline fails *before* its first paid call instead
        of half-way through an eval run.
        """
        if self.gemini_api_key is not None:
            value = self.gemini_api_key.get_secret_value().strip()
            if value:
                return value
        for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if os.environ.get(env_name, "").strip():
                return ""
        raise MissingApiKeyError(
            "no GEMINI_API_KEY in .env or the environment; create one at "
            "https://aistudio.google.com/apikey and add GEMINI_API_KEY=... to .env"
        )

    def redacted(self) -> dict[str, object]:
        """Settings safe to log or return from the API: secrets are replaced."""
        data = self.model_dump()
        for key, value in list(data.items()):
            if isinstance(value, SecretStr) or "key" in key or "password" in key:
                data[key] = "***" if value else None
        data["database_url"] = _redact_url(str(data.get("database_url", "")))
        data["readonly_database_url"] = _redact_url(str(data.get("readonly_database_url", "")))
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in data.items()}


def _redact_url(url: str) -> str:
    """Hide the password in ``postgresql://user:pass@host/db``."""
    if "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    creds, host = rest.split("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}//{user}:***@{host}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (call ``get_settings.cache_clear()`` in tests).

    ``REGLENS_ENV_FILE`` points at an alternative env file: tests set it to a temp path
    so a developer's real ``.env`` (providers, keys, database URL) can never leak into
    an assertion and make the suite depend on the machine it runs on.
    """
    override = os.environ.get("REGLENS_ENV_FILE")
    if override:
        candidate = Path(override)
        return Settings(_env_file=candidate if candidate.is_file() else None)
    env_file = paths.project_root() / ".env"
    return Settings(_env_file=env_file if env_file.is_file() else None)
