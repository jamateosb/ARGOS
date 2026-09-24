# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Project configuration loading helpers.

The repository-level ``config.yaml`` is the source of truth for operational
defaults. CLI arguments and environment variables may override those values at
runtime, but scripts should load this module first instead of duplicating
hardcoded defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from argos.common.constants import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def resolve_config_path(path: Optional[Path | str] = None) -> Path:
    """Return the explicit config path, ``ARGOS_CONFIG``, or repo ``config.yaml``."""
    if path is not None:
        return Path(path).expanduser()
    return Path(os.getenv("ARGOS_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()


def load_project_config(path: Optional[Path | str] = None, *, required: bool = False) -> dict[str, Any]:
    """Load a YAML project config.

    Missing optional configs return an empty mapping so local scripts can still
    run in minimal test environments.
    """
    config_path = resolve_config_path(path)
    if not config_path.exists():
        if required:
            raise FileNotFoundError(f"Project config not found: {config_path}")
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def node_endpoint_list(config: dict[str, Any]) -> list[str]:
    """Return node endpoints declared under ``cluster.nodes``."""
    endpoints: list[str] = []
    for node in config.get("cluster", {}).get("nodes", []) or []:
        endpoint = node.get("endpoint") or f"http://{node.get('host')}:{node.get('port', 8000)}"
        if endpoint and "None" not in endpoint:
            endpoints.append(str(endpoint))
    return endpoints


def node_registration_list(config: dict[str, Any]) -> list[str]:
    """Return API node registrations in ``node-id=http://host:port`` format."""
    registrations: list[str] = []
    for node in config.get("cluster", {}).get("nodes", []) or []:
        node_id = node.get("id")
        endpoint = node.get("endpoint") or f"http://{node.get('host')}:{node.get('port', 8000)}"
        if node_id and endpoint and "None" not in endpoint:
            registrations.append(f"{node_id}={endpoint}")
    return registrations
