"""FastAPI dependencies.

The database pool is optional on purpose: the API must start and serve ``/health`` even
when Postgres is not up yet (a fresh clone, a CI smoke test), while ``/ready`` reports the
truth. Hard-failing at import time makes for a worse learning loop.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import Request

from reglens.config import Settings, get_settings


def get_app_settings() -> Settings:
    return get_settings()


def get_pool(request: Request) -> asyncpg.Pool | None:
    return getattr(request.app.state, "pool", None)


def get_db_health(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "db_health", {"status": "unknown"})
