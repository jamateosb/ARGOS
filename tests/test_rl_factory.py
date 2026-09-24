# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for the RL factory, including torch fallback and policy filenames."""

from __future__ import annotations

import pytest

from argos.orchestrator.rl.factory import (
    build_policy_filename,
    create_agent,
    find_latest_policy,
    is_torch_available,
    normalize_algorithm,
)
from argos.orchestrator.rl.qlearning import TabularQLearningAgent


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("qlearning", "qlearning"),
        ("Q-Learning", "qlearning"),
        ("Q_Learning", "qlearning"),
        ("q", "qlearning"),
        ("dqn", "dqn"),
        ("DQN", "dqn"),
        ("PPO", "ppo"),
        ("ppo", "ppo"),
    ],
)
def test_normalize_algorithm_accepts_known_aliases(raw, expected):
    assert normalize_algorithm(raw) == expected


def test_normalize_algorithm_defaults_to_qlearning_when_none():
    assert normalize_algorithm(None) == "qlearning"


def test_normalize_algorithm_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        normalize_algorithm("a3c")


def test_create_agent_qlearning_is_tabular():
    agent, meta = create_agent(
        algorithm="qlearning",
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=42,
    )
    assert isinstance(agent, TabularQLearningAgent)
    assert meta["effective_algorithm"] == "qlearning"
    assert meta["fallback_reason"] is None
    assert meta["requested_algorithm"] == "qlearning"


def test_create_agent_dqn_falls_back_when_torch_missing():
    if is_torch_available():
        pytest.skip("Torch is available; this test exercises the fallback path only")
    agent, meta = create_agent(
        algorithm="dqn",
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=42,
    )
    assert isinstance(agent, TabularQLearningAgent)
    assert meta["effective_algorithm"] == "qlearning"
    assert meta["requested_algorithm"] == "dqn"
    assert meta["fallback_reason"] is not None


def test_create_agent_ppo_falls_back_when_torch_missing():
    if is_torch_available():
        pytest.skip("Torch is available; this test exercises the fallback path only")
    agent, meta = create_agent(
        algorithm="ppo",
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=42,
    )
    assert isinstance(agent, TabularQLearningAgent)
    assert meta["effective_algorithm"] == "qlearning"
    assert meta["requested_algorithm"] == "ppo"
    assert meta["fallback_reason"] is not None


def test_build_policy_filename_uses_normalized_algorithm():
    assert build_policy_filename("req-123", "Q-Learning", ".json") == "agent_req-123_qlearning.json"
    assert build_policy_filename("req-abc", "DQN", ".pt") == "agent_req-abc_dqn.pt"


def test_find_latest_policy_returns_newest(tmp_path):
    older = tmp_path / "agent_req-old_qlearning.json"
    newer = tmp_path / "agent_req-new_qlearning.json"
    older.write_text("{}")
    newer.write_text("{}")
    # Make sure mtimes differ
    import os
    import time

    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))

    result = find_latest_policy(str(tmp_path), "qlearning", ".json")
    assert result == newer


def test_find_latest_policy_returns_none_when_dir_missing(tmp_path):
    result = find_latest_policy(str(tmp_path / "does-not-exist"), "qlearning", ".json")
    assert result is None


def test_find_latest_policy_falls_back_to_legacy_for_qlearning(tmp_path):
    legacy = tmp_path / "agent_req-legacy.json"
    legacy.write_text("{}")
    result = find_latest_policy(str(tmp_path), "qlearning", ".json")
    assert result == legacy
