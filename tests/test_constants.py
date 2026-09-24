# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for shared filesystem constants."""

from argos.common.constants import DATA_ROOT, EXPERIMENTS_DIR, REPO_ROOT, SRC_ROOT


def test_common_paths_resolve_to_repository_root():
    """Shared paths should not point inside the Python package directory."""
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert SRC_ROOT == REPO_ROOT / "src"
    assert DATA_ROOT == REPO_ROOT / "data"
    assert EXPERIMENTS_DIR == REPO_ROOT / "experiments"
