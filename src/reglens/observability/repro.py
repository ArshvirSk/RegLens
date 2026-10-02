"""Reproducibility metadata recorded in every eval report.

The PRD requires each report to name the config, corpus version, git commit and model
versions it came from. This module produces the git half; when git is unavailable the
value is the string ``unknown`` — never a guessed sha.
"""

from __future__ import annotations

import subprocess


def git_commit(*, cwd: str | None = None) -> str:
    """Return HEAD's sha, or ``unknown`` if git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"
