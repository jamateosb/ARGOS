# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Behavioural tests for the tabular Q-learning agent.

These tests exercise the agent on a deterministic minimal MDP so a
regression in the Bellman update would surface immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from argos.orchestrator.rl.base import RLAction
from argos.orchestrator.rl.qlearning import QLearningConfig, TabularQLearningAgent


def _make_agent(*, seed: int = 0, epsilon: float = 0.0) -> TabularQLearningAgent:
    config = QLearningConfig(
        alpha=0.5,
        gamma=0.9,
        epsilon=epsilon,
        exploration_decay=0.9,
        min_exploration_rate=0.01,
        seed=seed,
    )
    actions = [
        RLAction("good", 0.0, 0.0, 0.0, None, 0),
        RLAction("bad", 0.0, 0.0, 0.0, None, 1),
    ]
    return TabularQLearningAgent(config, actions=actions)


def test_q_value_update_converges_to_immediate_reward_when_terminal():
    agent = _make_agent()
    state = ("s0",)
    next_state = ("s1",)
    good = agent._actions[0]

    # With alpha=0.5 and gamma=0.9, repeated terminal updates with reward=1
    # converge to 1.0 (within floating point tolerance).
    for _ in range(50):
        agent.update(state, good, reward=1.0, next_state=next_state, done=True)

    assert agent.get_q_value(state, good) == pytest.approx(1.0)


def test_agent_learns_to_prefer_high_reward_action():
    agent = _make_agent(seed=7, epsilon=0.0)
    state = ("s0",)
    next_state = ("s1",)
    good, bad = agent._actions

    # Train: "good" yields +1, "bad" yields -1.
    for _ in range(100):
        agent.update(state, good, reward=1.0, next_state=next_state, done=True)
        agent.update(state, bad, reward=-1.0, next_state=next_state, done=True)

    selected = agent.select_action(state, training=False)
    assert selected.action_type == "good"
    assert agent.get_best_action(state).action_type == "good"
    assert agent.last_was_exploration is False


def test_explore_flag_is_set_when_random_branch_taken():
    agent = _make_agent(seed=1, epsilon=1.0)
    state = ("s0",)
    agent.select_action(state, training=True)
    assert agent.last_was_exploration is True


def test_frozen_selection_does_not_mutate_q_table():
    agent = _make_agent(seed=3)
    before = agent.policy_fingerprint()

    agent.select_action(("unseen",), training=False)

    assert agent.policy_fingerprint() == before
    assert agent.state_count == 0


def test_action_mask_applies_to_exploration_and_greedy_selection():
    agent = _make_agent(seed=3, epsilon=1.0)
    state = ("masked",)

    exploratory = {agent.select_action(state, training=True, allowed_action_indices=(1,)).index for _ in range(5)}
    greedy = agent.select_action(state, training=False, allowed_action_indices=(1,))

    assert exploratory == {1}
    assert greedy.index == 1


def test_decay_exploration_respects_min_floor():
    agent = _make_agent(epsilon=0.5)
    for _ in range(200):
        agent.decay_exploration()
    assert agent.epsilon == agent.qconfig.min_exploration_rate


def test_set_epsilon_clamps_to_unit_interval():
    agent = _make_agent()
    agent.set_epsilon(1.5)
    assert agent.epsilon == 1.0
    agent.set_epsilon(-0.2)
    assert agent.epsilon == 0.0


def test_save_and_load_round_trip(tmp_path: Path):
    agent = _make_agent(seed=11, epsilon=0.3)
    state = (0.1, 0.2, 0.3, 0.4, 0.5, 1)
    next_state = (0.2, 0.3, 0.4, 0.5, 0.6, 2)
    good = agent._actions[0]

    agent.update(state, good, reward=0.7, next_state=next_state, done=True)
    agent.episode_complete()

    path = tmp_path / "agent.json"
    agent.save(str(path))

    payload = json.loads(path.read_text())
    assert "q_table" in payload
    assert payload["policy_metadata"]["algorithm"] == "qlearning"

    fresh = _make_agent(seed=11, epsilon=0.99)
    fresh.load(str(path))

    assert fresh.epsilon == agent.epsilon
    assert fresh.episode_count == agent.episode_count
    assert fresh.get_q_value(state, good) == agent.get_q_value(state, good)


def test_policy_metadata_includes_fallback_reason_after_set_runtime_metadata():
    agent = _make_agent()
    agent.set_runtime_metadata(
        requested_algorithm="dqn",
        fallback_reason="Torch missing",
        backend="python",
    )
    metadata = agent.get_policy_metadata()
    assert metadata["requested_algorithm"] == "dqn"
    assert metadata["fallback_reason"] == "Torch missing"
    assert metadata["backend"] == "python"
    assert metadata["algorithm"] == "qlearning"


def test_state_count_grows_as_new_states_appear():
    agent = _make_agent()
    state1 = ("s1",)
    state2 = ("s2",)
    state3 = ("s3",)
    agent.update(state1, agent._actions[0], 0.1, state2)
    agent.update(state2, agent._actions[0], 0.1, state3)
    assert agent.state_count >= 2
