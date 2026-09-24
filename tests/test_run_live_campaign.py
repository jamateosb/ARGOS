"""Tests for canonical frozen-policy live campaign gates."""

import hashlib
import json

import pytest
from scripts.run_live_campaign import _policy_map


def test_policy_map_resolves_profile_specific_artifacts(tmp_path):
    artifact = tmp_path / "ppo.pt"
    artifact.write_bytes(b"policy")
    index = tmp_path / "campaign_index.json"
    index.write_text(
        json.dumps(
            {
                "completed_at": "2026-07-20T00:00:00+00:00",
                "git_commit": "a" * 40,
                "runs": [
                    {
                        "role": "train",
                        "algorithm": "ppo",
                        "profile": "aggressive-incident",
                        "seed": 11,
                        "policy_artifact": str(artifact),
                        "policy_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    mapping = _policy_map([index], training_seed=11, expected_commit="a" * 40)

    assert mapping == {"aggressive-incident": str(artifact.resolve())}


def test_policy_map_rejects_artifact_hash_mismatch(tmp_path):
    artifact = tmp_path / "ppo.pt"
    artifact.write_bytes(b"policy")
    index = tmp_path / "campaign_index.json"
    index.write_text(
        json.dumps(
            {
                "completed_at": "2026-07-20T00:00:00+00:00",
                "git_commit": "a" * 40,
                "runs": [
                    {
                        "role": "train",
                        "algorithm": "ppo",
                        "profile": "aggressive-incident",
                        "seed": 11,
                        "policy_artifact": str(artifact),
                        "policy_sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hash mismatch"):
        _policy_map([index], training_seed=11, expected_commit="a" * 40)
