# =============================================================================
# ARGOS — Adaptive Reinforcement-driven Governance for Orchestrated Services
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for the StaticAgent baseline and its factory integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argos.domain.requests import SUPPORTED_RL_ALGORITHMS, normalize_algorithm_name
from argos.orchestrator.rl import StaticAgent, StaticConfig, create_agent
from argos.orchestrator.rl.base import STANDARD_ACTIONS
from argos.orchestrator.rl.selection import resolve_algorithm


def _build_agent() -> StaticAgent:
    return StaticAgent(StaticConfig(seed=11))


def test_static_is_registered_as_supported_algorithm():
    assert "static" in SUPPORTED_RL_ALGORITHMS


@pytest.mark.parametrize("raw", ["static", "Static", "fixed", "BASELINE", "no-op", "Noop"])
def test_normalize_algorithm_name_maps_aliases_to_static(raw: str):
    assert normalize_algorithm_name(raw) == "static"


def test_select_action_always_returns_hold():
    agent = _build_agent()
    actions_taken = {agent.select_action(("any", "state", i), training=True).action_type for i in range(20)}
    assert actions_taken == {"hold"}


def test_select_action_ignores_training_flag():
    agent = _build_agent()
    greedy = agent.select_action(("s",), training=False)
    exploratory = agent.select_action(("s",), training=True)
    assert greedy.action_type == "hold"
    assert exploratory.action_type == "hold"
    assert agent.last_was_exploration is False


def test_update_records_reward_without_learning():
    agent = _build_agent()
    metrics = agent.update(state=("s",), action=STANDARD_ACTIONS[0], reward=0.42, next_state=("s2",))
    assert metrics["reward"] == pytest.approx(0.42)
    assert metrics["td_error"] == 0.0
    assert agent.step_count == 1
    assert agent.total_reward == pytest.approx(0.42)


def test_get_policy_assigns_full_mass_to_hold():
    agent = _build_agent()
    policy = agent.get_policy(("anything",))
    hold_actions = [a for a in policy if a.action_type == "hold"]
    other_actions = [a for a in policy if a.action_type != "hold"]
    assert len(hold_actions) == 1
    assert policy[hold_actions[0]] == 1.0
    assert all(policy[a] == 0.0 for a in other_actions)


def test_decay_exploration_is_a_no_op():
    agent = _build_agent()
    agent.decay_exploration()  # must not raise


def test_save_and_load_round_trip(tmp_path: Path):
    agent = _build_agent()
    agent.update(state=("s",), action=STANDARD_ACTIONS[0], reward=1.0, next_state=("s2",))
    agent.update(state=("s",), action=STANDARD_ACTIONS[0], reward=2.0, next_state=("s3",))
    target = tmp_path / "static_policy.json"
    agent.save(str(target))

    payload = json.loads(target.read_text())
    assert payload["policy"] == "static"
    assert payload["step_count"] == 2
    assert payload["total_reward"] == pytest.approx(3.0)

    fresh = _build_agent()
    fresh.load(str(target))
    assert fresh.step_count == 2
    assert fresh.total_reward == pytest.approx(3.0)


def test_factory_builds_static_agent():
    agent, meta = create_agent(
        algorithm="static",
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    assert isinstance(agent, StaticAgent)
    assert meta["effective_algorithm"] == "static"
    assert meta["requested_algorithm"] == "static"
    assert meta["fallback_reason"] is None


def test_factory_accepts_aliases_for_static():
    for alias in ("baseline", "fixed", "noop"):
        agent, meta = create_agent(
            algorithm=alias,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1,
            exploration_decay=0.99,
            min_exploration_rate=0.01,
            seed=11,
        )
        assert isinstance(agent, StaticAgent)
        assert meta["effective_algorithm"] == "static"


def test_resolve_algorithm_accepts_static_explicitly():
    assert resolve_algorithm("static") == "static"
    assert resolve_algorithm("BASELINE") == "static"


def test_auto_ranking_ignores_static_rows(tmp_path: Path, monkeypatch):
    """The benchmark ranking CSV may contain static rows; auto must skip them."""
    csv_path = tmp_path / "model_ranking_ci.csv"
    csv_path.write_text(
        "runtime,mode,algorithm,n,reward_mean,reward_stdev,reward_ci95_low,reward_ci95_high\n"
        "process,tuned,static,25,99.0,1.0,98.0,100.0\n"
        "process,tuned,ppo,25,25.0,5.0,23.0,27.0\n"
        "process,tuned,qlearning,25,20.0,4.0,18.0,22.0\n"
    )
    monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(csv_path))

    # Even though static would win on raw reward, auto must pick the best
    # learning agent (ppo here), never the static baseline.
    assert resolve_algorithm("auto") == "ppo"
