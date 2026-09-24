# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Smoke tests for the optional DQN and PPO backends.

The tests are silently skipped when PyTorch is not installed so the suite
still runs in minimal CI environments. When Torch is present they exercise
agent construction, action selection, learning updates, and the save/load
round-trip to catch ABI or interface regressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argos.orchestrator.rl.base import RLState
from argos.orchestrator.rl.factory import create_agent, is_torch_available
from argos.orchestrator.rl.ppo import PPOAgent, PPOConfig

pytestmark = pytest.mark.skipif(not is_torch_available(), reason="Torch backend not installed")


def _state(cpu: float = 0.4, mem: float = 0.4) -> RLState:
    return RLState(
        cpu_utilization=cpu,
        memory_utilization=mem,
        coverage=0.5,
        sample_rate=0.5,
        freshness=60.0,
        active_nodes=2,
        total_nodes=3,
        active_requests=1,
    )


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_create_agent_with_torch_returns_concrete_backend(algorithm):
    agent, meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    assert meta["fallback_reason"] is None
    assert meta["effective_algorithm"] == algorithm
    assert meta["requested_algorithm"] == algorithm


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_torch_agent_select_action_returns_valid_action(algorithm):
    agent, _meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    state = _state()
    action = agent.select_action(state, training=False)
    assert hasattr(action, "action_type")
    assert hasattr(action, "index")


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_torch_agent_update_returns_metrics_dict(algorithm):
    agent, _meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    state = _state()
    next_state = _state(cpu=0.5, mem=0.5)
    action = agent.select_action(state, training=True)
    metrics = agent.update(state, action, reward=0.5, next_state=next_state, done=False)
    assert isinstance(metrics, dict)


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_torch_agent_save_and_load_round_trip(algorithm, tmp_path: Path):
    agent, _meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    state = _state()
    next_state = _state(cpu=0.5, mem=0.5)
    action = agent.select_action(state, training=True)
    agent.update(state, action, reward=0.5, next_state=next_state, done=False)

    extension = agent.policy_extension
    path = tmp_path / f"agent{extension}"
    agent.save(str(path))
    assert path.exists()

    other, _other_meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )
    other.load(str(path))
    metadata = other.get_policy_metadata()
    assert metadata["algorithm"] == algorithm


def test_ppo_waits_for_complete_rollout_before_optimization():
    agent = PPOAgent(
        PPOConfig(
            learning_rate=0.001,
            seed=11,
            rollout_size=4,
            minibatch_size=2,
            update_epochs=2,
        )
    )
    state = _state()
    next_state = _state(cpu=0.5)

    for _ in range(3):
        action = agent.select_action(state, training=True)
        metrics = agent.update(state, action, reward=0.5, next_state=next_state)

    assert metrics["optimized"] == 0.0
    assert agent.get_metrics()["ppo_updates"] == 0
    assert agent.get_metrics()["rollout_pending"] == 3

    action = agent.select_action(state, training=True)
    metrics = agent.update(state, action, reward=0.5, next_state=next_state)

    assert metrics["optimized"] == 1.0
    assert agent.get_metrics()["ppo_updates"] == 1
    assert agent.get_metrics()["rollout_pending"] == 0


def test_ppo_frozen_evaluation_is_seeded_and_does_not_queue_rollout():
    state = _state()
    first = PPOAgent(PPOConfig(seed=22, rollout_size=4))
    first_actions = [first.select_action(state, training=False).index for _ in range(10)]
    second = PPOAgent(PPOConfig(seed=22, rollout_size=4))
    second_actions = [second.select_action(state, training=False).index for _ in range(10)]

    assert first_actions == second_actions
    assert len(set(first_actions)) == 1
    assert first.get_metrics()["rollout_pending"] == 0
    assert first.epsilon == 0.0


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_torch_policy_fingerprint_is_stable_during_frozen_selection(algorithm):
    agent, _meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=31,
    )
    before = agent.policy_fingerprint()

    for _ in range(5):
        agent.select_action(_state(), training=False)

    assert agent.policy_fingerprint() == before


def test_ppo_seed_controls_network_initialization():
    first = PPOAgent(PPOConfig(seed=41))
    second = PPOAgent(PPOConfig(seed=41))
    different = PPOAgent(PPOConfig(seed=42))

    assert first.policy_fingerprint() == second.policy_fingerprint()
    assert first.policy_fingerprint() != different.policy_fingerprint()


def test_ppo_guard_rewrite_records_log_prob_under_original_action_mask():
    agent = PPOAgent(PPOConfig(seed=43, rollout_size=8))
    state = _state()
    allowed = (0, 1)
    selected = agent.select_action(state, training=True, allowed_action_indices=allowed)
    executed_index = 1 if selected.index == 0 else 0
    executed = agent._actions[executed_index]

    state_tensor = agent._state_tensor(state)
    with agent._torch.no_grad():
        logits = agent._network.actor(state_tensor)
        masked = agent._torch.full_like(logits, float("-inf"))
        masked[:, list(allowed)] = logits[:, list(allowed)]
        distribution = agent._torch.distributions.Categorical(logits=masked)
        expected = float(
            distribution.log_prob(agent._torch.tensor([executed_index])).item()
        )

    agent.update(
        state,
        executed,
        reward=0.1,
        next_state=_state(cpu=0.5),
        allowed_action_indices=allowed,
        next_allowed_action_indices=allowed,
    )

    assert agent._rollout[0].old_log_prob == pytest.approx(expected)
    assert agent._rollout[0].allowed_indices == allowed


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_agent_construction_does_not_rewind_process_global_rng(algorithm):
    import random

    import torch

    random.seed(77)
    torch.manual_seed(77)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=0.2,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_torch_action_mask_excludes_invalid_actions(algorithm):
    agent, _meta = create_agent(
        algorithm=algorithm,
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=51,
    )

    selected = {
        agent.select_action(_state(), training=True, allowed_action_indices=(2,)).index
        for _ in range(5)
    }

    assert selected == {2}
