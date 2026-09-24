"""Tests for immutable experiment provenance manifests."""

import json
import subprocess
from pathlib import Path

import pytest

from argos.provenance import (
    _git_metadata,
    build_run_manifest,
    tree_fingerprint,
    write_run_manifest,
)


def test_tree_fingerprint_is_order_independent_and_content_sensitive(tmp_path: Path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    left = tree_fingerprint(tmp_path, [first, second])
    right = tree_fingerprint(tmp_path, [second, first])
    second.write_text("changed", encoding="utf-8")
    changed = tree_fingerprint(tmp_path, [first, second])

    assert left == right
    assert changed["sha256"] != left["sha256"]


def test_run_manifest_records_code_dataset_contract_and_nodes(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "data" / "seville_bus").mkdir(parents=True)
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "run.py").write_text("print('run')\n", encoding="utf-8")
    (tmp_path / "data" / "seville_bus" / "SB_User0.json").write_text("[]\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("runtime: local\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    manifest = build_run_manifest(
        repository_root=tmp_path,
        config_path=config,
        dataset_path=tmp_path / "data" / "seville_bus",
        command=["python", "run.py"],
        summary={"policy_mode": "evaluate"},
        request_contract={"input_multiplier": 8},
        nodes=[{"node_id": "small", "runtime_mode": "thread"}],
    )

    assert manifest["schema_version"] == "argos.run-manifest.v1"
    assert manifest["code"]["file_count"] == 2
    assert manifest["dataset"]["file_count"] == 1
    assert manifest["request_contract"]["input_multiplier"] == 8
    assert manifest["nodes"][0]["node_id"] == "small"

    target = tmp_path / "output" / "manifest.json"
    write_run_manifest(target, manifest)
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == "argos.run-manifest.v1"


def test_git_metadata_fails_closed_outside_repository(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Unable to capture Git provenance"):
        _git_metadata(tmp_path)
