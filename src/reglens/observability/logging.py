"""Structured JSON logging.

Downstream this feeds the Phase 4 dashboard without a log-shipping agent: one JSON
object per line is trivially grep-able locally and parseable in a container. Trace logs
(``event="trace"``) are emitted by :mod:`reglens.observability.tracing`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Render one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in RESERVED:
                # Standard LogRecord attributes (module, name, lineno, ...) are logger
                # internals, emitted as-is under their own keys. Caller fields that would
                # collide were renamed by `safe_extra` to `field_<key>` and are restored
                # below, which is why they are not touched here.
                payload[key] = value
                continue
            payload[key.removeprefix("field_")] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", *, stream: Any = None) -> None:
    """Configure the root logger once. Safe to call repeatedly.

    Third-party HTTP loggers are dropped to WARNING: a 95-document fetch prints one line
    per request per redirect, which buries the fetch report it belongs to. Those requests
    are still visible as ``http_status`` in the manifest.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    for name in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def safe_extra(**fields: Any) -> dict[str, Any]:
    """Rename keys that collide with reserved ``LogRecord`` attributes.

    ``logging`` raises ``KeyError: Attempt to overwrite 'name' in LogRecord`` for reserved
    keys such as ``name``, ``filename``, ``module`` or ``args``. Silently dropping them
    would hide information, so they are prefixed instead: ``name`` becomes ``field_name``.
    """
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        # Reserved names must never reach `extra=`: logging raises KeyError for them.
        # Prefixing is lossless because the formatter strips it back off.
        safe[f"field_{key}" if key in RESERVED else key] = value
    return safe


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a structured event: ``log_event(log, "retrieved", chunks=12)``."""
    logger.info(message, extra=safe_extra(**fields))
