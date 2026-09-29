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


def test_trial_command_is_accepted_by_the_live_trial_driver(tmp_path):
    import argparse
    from pathlib import Path

    from argos import live_trial
    from scripts.run_live_campaign import _trial_command

    args = argparse.Namespace(
        config=Path("config.yaml"),
        duration_minutes=30.0,
        poll_interval_seconds=10.0,
        campaign_id="c",
        runtime="thread",
        api_token="",
    )
    command = _trial_command(
        args=args,
        scenario="realistic",
        live_seed=1001,
        variant="dqn",
        run_root=tmp_path / "run",
        persistence_dir=tmp_path / "persistence",
        orchestrator_url="http://127.0.0.1:8101",
    )
    assert command[1:3] == ["-m", "argos.live_trial"]
    parsed = live_trial.parse_args(command[3:])
    assert parsed.require_frozen_policy is True
    assert parsed.random_seed == 1001
