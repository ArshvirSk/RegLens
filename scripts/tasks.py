#!/usr/bin/env python
"""Task runner that mirrors the Makefile.

Why this exists: ``make`` is the documented interface, but it is not installed on every
host (in particular the Windows machine this project was built on). Rather than keep two
lists of commands that drift, the Makefile delegates here and this file is the single
source of truth for what each target runs.

    uv run python scripts/tasks.py            # list targets
    uv run python scripts/tasks.py test
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    commands: tuple[str, ...]


TASKS: tuple[Task, ...] = (
    Task("setup", "Create the venv and install pinned dependencies", ("uv sync --all-groups",)),
    Task("lock", "Resolve and pin dependency versions", ("uv lock",)),
    Task("up", "Start the stack (postgres + api + web)", ("docker compose up -d --build",)),
    Task("down", "Stop the stack", ("docker compose down",)),
    Task("logs", "Tail stack logs", ("docker compose logs -f --tail=100",)),
    Task("ps", "Show stack status", ("docker compose ps",)),
    Task("build", "Build container images", ("docker compose build",)),
    Task("migrate", "Apply SQL migrations", ("uv run python -m reglens.cli migrate",)),
    Task(
        "test",
        "Run unit tests (no Docker required)",
        ('uv run pytest -m "not integration and not eval"',),
    ),
    Task(
        "test-integration",
        "Run integration tests against Postgres",
        ("uv run pytest -m integration",),
    ),
    Task(
        "lint",
        "Lint without modifying files",
        ("uv run ruff check .", "uv run ruff format --check ."),
    ),
    Task("fmt", "Auto-format and auto-fix", ("uv run ruff format .", "uv run ruff check --fix .")),
    Task(
        "check",
        "What CI runs: lint, unit tests, manifest validation",
        (
            "uv run ruff check .",
            "uv run ruff format --check .",
            'uv run pytest -m "not integration and not eval"',
            "uv run python -m reglens.cli validate-manifest",
        ),
    ),
    Task(
        "status",
        "Configuration, manifest and eval-set status",
        ("uv run python -m reglens.cli status --with-db",),
    ),
    Task(
        "validate-manifest",
        "Validate data/manifest.csv and refresh its JSON schema",
        ("uv run python -m reglens.cli validate-manifest --write-schema",),
    ),
    Task(
        "corpus-plan",
        "Regenerate data/manifest.csv from data/corpus_plan.yaml",
        ("uv run python -m reglens.cli corpus-plan --write",),
    ),
    Task(
        "download",
        "Dry-run the manifest download plan (add -- --yes --accept-terms to fetch)",
        ("uv run python -m reglens.cli download --dry-run",),
    ),
    Task(
        "ingest",
        "Parse, chunk and index the corpus (Phase 1)",
        ("uv run python -m reglens.cli ingest",),
    ),
    Task("reindex", "Rebuild indexes (Phase 1)", ("uv run python -m reglens.cli reindex",)),
    Task("eval", "Run the golden eval set (Phase 1)", ("uv run python -m reglens.cli eval",)),
    Task("refresh", "Check for new circulars (Phase 4)", ("uv run python -m reglens.cli refresh",)),
    Task(
        "api", "Run the API locally", ("uv run uvicorn reglens.api.main:app --reload --port 8000",)
    ),
    Task("web", "Run the Next.js dev server", ("npm run dev",)),
    Task(
        "clean",
        "Remove caches and build artifacts",
        ("rm -rf .pytest_cache .ruff_cache htmlcov .coverage web/.next",),
    ),
    Task("reset-db", "Drop the local database volume (destructive)", ("docker compose down -v",)),
)

BY_NAME = {task.name: task for task in TASKS}


def run(task: Task, extra: list[str]) -> int:
    for index, command in enumerate(task.commands):
        final = index == len(task.commands) - 1
        full = f"{command} {' '.join(extra)}".strip() if (final and extra) else command
        cwd = ROOT / "web" if task.name == "web" else ROOT
        print(f"[tasks] {full}")
        result = subprocess.run(full, shell=True, cwd=cwd, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"help", "--help", "-h", "list"}:
        width = max(len(task.name) for task in TASKS)
        for task in TASKS:
            print(f"  {task.name.ljust(width)}  {task.description}")
        return 0
    name, *extra = argv
    task = BY_NAME.get(name)
    if task is None:
        print(f"unknown target: {name}\n", file=sys.stderr)
        return main(["help"])
    return run(task, extra)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
