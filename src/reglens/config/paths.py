"""Path helpers.

Every path in the project is resolved against the project root (the directory that
contains ``pyproject.toml`` and ``src/``) rather than the current working directory.
Without this, ``make eval`` and ``pytest`` and the containerised API would each
resolve ``./data`` differently, which breaks reproducibility of eval reports.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root, falling back to the CWD if the markers are absent."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (project_root() / candidate).resolve()


# --- canonical locations -------------------------------------------------------
MANIFEST_PATH = "data/manifest.csv"
MANIFEST_SCHEMA_PATH = "data/manifest.schema.json"
CORPUS_PLAN_PATH = "data/corpus_plan.yaml"
GOLDEN_PATH = "eval/golden/questions.jsonl"
GOLDEN_SCHEMA_PATH = "eval/golden/schema.md"
EVAL_RESULTS_DIR = "eval/results"
EXPERIMENT_CONFIG_DIR = "src/reglens/config/experiments"
MIGRATIONS_DIR = "src/reglens/db/migrations"
