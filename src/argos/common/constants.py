# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Shared constants for paths, time formats, and identifiers."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path

_THIS_FILE: Path = Path(__file__).resolve()
REPO_ROOT: Path = _THIS_FILE.parents[3]
SRC_ROOT: Path = REPO_ROOT / "src"
DATA_ROOT: Path = REPO_ROOT / "data"
OBJECTS_ROOT: Path = REPO_ROOT / "objects"

# Standard data sub-directories
RUNS_DIR: Path = DATA_ROOT / "runs"
PERSISTENCE_DIR: Path = DATA_ROOT / "persistence"
TUNING_DIR: Path = DATA_ROOT / "tuning"
MODELS_DIR: Path = DATA_ROOT / "models"
RL_AGENTS_DIR: Path = DATA_ROOT / "rl_agents"
EVALUATION_DIR: Path = DATA_ROOT / "evaluation"
SESSIONS_DIR: Path = DATA_ROOT / "sessions"
SEVILLE_DATA_DIR: Path = DATA_ROOT / "seville_bus"
EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"

# Time representation used in filenames and logs.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"

# Default machine identifier (can be overridden by env vars/config).
DEFAULT_MACHINE_ID = platform.node()

# Supported output formats
OUTPUT_FORMAT_JSON = "json"
OUTPUT_FORMAT_JSONL = "jsonl"
OUTPUT_FORMAT_CSV = "csv"


# Session management
def generate_session_id() -> str:
    """Generate a session ID as YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Reproducibility
DEFAULT_RANDOM_SEED = 42
