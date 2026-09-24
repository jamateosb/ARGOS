# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Immutable provenance helpers for canonical experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    """Hash a deterministic set of files relative to ``root``."""
    digest = hashlib.sha256()
    files = sorted({path.resolve() for path in paths if path.is_file()})
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root.resolve()).as_posix()
        content_hash = _sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "total_bytes": total_bytes}


def _git_metadata(root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(f"Unable to capture Git provenance: {detail.strip()}") from exc
        return result.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError(f"Invalid Git commit reported for provenance: {commit!r}")
    status = _run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "changed_path_count": len(status.splitlines()),
    }


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_run_manifest(
    *,
    repository_root: Path,
    config_path: Optional[Path],
    dataset_path: Path,
    command: list[str],
    summary: dict[str, Any],
    request_contract: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a self-contained manifest for one train or evaluation run."""
    code_paths: list[Path] = []
    for relative in ("src", "scripts"):
        code_paths.extend(
            path
            for path in (repository_root / relative).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for filename in ("pyproject.toml", "requirements.txt", "requirements-rl.txt"):
        path = repository_root / filename
        if path.exists():
            code_paths.append(path)

    dataset_files = [path for path in dataset_path.rglob("*") if path.is_file()]
    config_record = None
    if config_path and config_path.exists():
        config_record = {
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
        }

    return {
        "schema_version": "argos.run-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "request_contract": request_contract,
        "command": command,
        "git": _git_metadata(repository_root),
        "code": tree_fingerprint(repository_root, code_paths),
        "dataset": {
            "path": str(dataset_path.resolve()),
            **tree_fingerprint(repository_root, dataset_files),
        },
        "config": config_record,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(("torch", "numpy", "pandas", "scipy", "fastapi", "pydantic")),
        },
        "nodes": nodes,
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest atomically enough for single-process campaign runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
