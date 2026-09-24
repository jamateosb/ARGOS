# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for the abstract RL base classes and shared action space."""

from __future__ import annotations

from argos.orchestrator.rl.base import (
    STANDARD_ACTIONS,
    RLAction,
    RLConfig,
    RLExperience,
    RLState,
)


def test_rl_state_to_tuple_is_hashable_and_rounded():
    state = RLState(
        cpu_utilization=0.673,
        memory_utilization=0.4321,
        coverage=0.555,
        sample_rate=0.4,
        freshness=90.0,
        active_nodes=3,
    )
    tuple_state = state.to_tuple()
    assert isinstance(tuple_state, tuple)
    assert hash(tuple_state)
    assert tuple_state == (0.67, 0.43, 0.56, 0.4, 1.5, 3)


def test_rl_state_to_vector_normalizes_dimensions():
    state = RLState(
        cpu_utilization=0.5,
        memory_utilization=0.6,
        network_utilization=0.1,
        coverage=0.4,
        sample_rate=0.3,
        freshness=60.0,
        response_time_ratio=0.8,
        cost_ratio=0.9,
        active_nodes=2,
        total_nodes=4,
        active_requests=5,
    )
    vector = state.to_vector()
    assert len(vector) == 10
    assert vector[0] == 0.5
    assert vector[5] == 0.5  # freshness/120
    assert vector[8] == 0.5  # active_nodes / total_nodes
    assert vector[9] == 0.5  # active_requests / 10


def test_rl_state_to_vector_handles_zero_total_nodes():
    state = RLState(total_nodes=0, active_nodes=1)
    vector = state.to_vector()
    assert vector[8] == 1.0  # uses max(total_nodes, 1)


def test_rl_action_equality_uses_type_and_index():
    a = RLAction("hold", 0.0, 0.0, 0.0, None, 0)
    b = RLAction("hold", 1.0, 2.0, 3.0, "coverage", 0)
    c = RLAction("hold", 0.0, 0.0, 0.0, None, 1)
    assert a == b  # same type and index regardless of deltas
    assert a != c  # different index
    assert hash(a) == hash(b)
    assert a != "hold"


def test_rl_action_eq_returns_false_for_non_action_objects():
    a = RLAction("hold", 0.0, 0.0, 0.0, None, 0)
    assert (a == 42) is False


def test_standard_actions_cover_all_dimensions():
    types = [action.action_type for action in STANDARD_ACTIONS]
    assert types == [
        "hold",
        "increase_coverage",
        "decrease_coverage",
        "increase_sample",
        "decrease_sample",
        "decrease_freshness",
        "increase_freshness",
    ]
    indices = [action.index for action in STANDARD_ACTIONS]
    assert indices == list(range(len(STANDARD_ACTIONS)))
    assert STANDARD_ACTIONS[5].freshness_delta == 10.0
    assert STANDARD_ACTIONS[6].freshness_delta == -10.0


def test_rl_experience_carries_state_action_reward_done():
    state = RLState(cpu_utilization=0.1)
    next_state = RLState(cpu_utilization=0.2)
    action = STANDARD_ACTIONS[1]
    exp = RLExperience(state=state, action=action, reward=0.5, next_state=next_state, done=True)
    assert exp.reward == 0.5
    assert exp.done is True
    assert exp.info == {}


def test_rl_config_defaults_are_reproducible():
    config = RLConfig()
    assert config.learning_rate == 0.1
    assert config.discount_factor == 0.95
    assert config.algorithm == "generic"
    assert config.policy_version == "generic-v1"
    assert config.state_schema_version == "argos.mdp.v4"


def test_rl_config_is_frozen():
    config = RLConfig()
    try:
        config.learning_rate = 0.5
    except Exception as exc:
        assert "frozen" in str(exc) or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("RLConfig should be frozen")


def test_reset_run_metrics_preserves_policy_but_clears_counters():
    from argos.orchestrator.rl.static import StaticAgent, StaticConfig

    agent = StaticAgent(StaticConfig())
    agent.update((0,), STANDARD_ACTIONS[0], 2.5, (1,))

    agent.reset_run_metrics()

    assert agent.step_count == 0
    assert agent.total_reward == 0.0
    assert agent.select_action((0,), training=False).action_type == "hold"
