# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Utility helpers shared across base components."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from argos.common.constants import DEFAULT_MACHINE_ID, TIMESTAMP_FORMAT

__all__ = [
    "slugify_machine_id",
    "utc_timestamp_str",
    "ensure_dir",
    "flatten_dict",
    "count_jsonl_entries",
    "read_last_jsonl_line",
]


def slugify_machine_id(machine_id: Optional[str] = None) -> str:
    """
    Sanitize the machine identifier to be filesystem-friendly.
    """
    raw = machine_id or DEFAULT_MACHINE_ID or "unknown-host"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()


def utc_timestamp_str(fmt: str = TIMESTAMP_FORMAT) -> str:
    """
    Return the current UTC timestamp formatted for filenames/logging.
    """
    return datetime.now(timezone.utc).strftime(fmt)


def ensure_dir(path: Path) -> Path:
    """
    Create the directory if it does not exist and return the path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flatten nested dictionaries using dot notation for keys.
    Useful for logging/metrics exports.
    """
    items: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def count_jsonl_entries(path: Path) -> int:
    """
    Count the number of non-empty lines in a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of non-empty lines (entries).
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def read_last_jsonl_line(path: Path) -> Optional[str]:
    """
    Read the last non-empty line from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        The last non-empty line, or None if file is empty or unreadable.
    """
    try:
        last_line = None
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        return last_line
    except OSError:
        return None
