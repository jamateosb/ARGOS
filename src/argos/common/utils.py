# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""General utilities shared across base and orchestrator packages."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

__all__ = [
    "setup_logging",
    "redact_keys",
    "retry_operation",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging with a consistent formatter.
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def redact_keys(payload: dict[str, Any], keys_to_redact: set[str]) -> dict[str, Any]:
    """
    Return a shallow copy of payload with selected keys replaced by '***'.
    Useful for logs that may contain secrets in future integrations.
    """
    redacted = {}
    for key, value in payload.items():
        redacted[key] = "***" if key in keys_to_redact else value
    return redacted


def retry_operation(
    func: Callable,
    attempts: int = 3,
    wait_seconds: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Execute a callable with simple retry and backoff.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for _attempt in range(1, attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as exc:
                last_exc = exc
                time.sleep(wait_seconds)
        if last_exc:
            raise last_exc

    return wrapper
