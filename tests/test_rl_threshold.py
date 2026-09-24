"""Tests for the deterministic threshold baseline."""

from pathlib import Path

from argos.settings import RL_STATE_KEYS
from argos.orchestrator.rl.factory import create_agent, normalize_algorithm
from argos.orchestrator.rl.threshold import ThresholdAgent, ThresholdConfig


def _state(cpu: int = 0, memory: int = 0, pending: int = 0, violations: int = 0) -> tuple[int, ...]:
    values = [0] * len(RL_STATE_KEYS)
    values[RL_STATE_KEYS.index("cpu")] = cpu
    values[RL_STATE_KEYS.index("memory")] = memory
    values[RL_STATE_KEYS.index("pending_requests")] = pending
    values[RL_STATE_KEYS.index("recent_slo_violations")] = violations
    return tuple(values)


def test_threshold_cycles_quality_actions_without_pressure():
    agent = ThresholdAgent(ThresholdConfig())

    actions = [agent.select_action(_state()).index for _ in range(6)]

    assert actions == [1, 3, 6, 1, 3, 6]


def test_threshold_cycles_relief_actions_under_pressure():
    agent = ThresholdAgent(ThresholdConfig(pressure_bucket=3))

    actions = [agent.select_action(_state(cpu=3)).index for _ in range(6)]

    assert actions == [4, 2, 5, 4, 2, 5]


def test_threshold_responds_to_queue_and_violation_context():
    queue_agent = ThresholdAgent(ThresholdConfig())
    violation_agent = ThresholdAgent(ThresholdConfig())

    assert queue_agent.select_action(_state(pending=2)).index == 4
    assert violation_agent.select_action(_state(violations=1)).index == 4


def test_threshold_leaves_pressure_mode_after_violation_window_clears():
    agent = ThresholdAgent(ThresholdConfig())

    assert agent.select_action(_state(violations=1)).index == 4
    assert agent.select_action(_state(violations=0)).index in {1, 3, 6}


def test_threshold_reset_restores_round_robin_phase():
    agent = ThresholdAgent(ThresholdConfig())
    agent.select_action(_state())
    agent.select_action(_state())

    agent.reset_run_metrics()

    assert agent.select_action(_state()).index == 1


def test_threshold_factory_and_aliases():
    agent, metadata = create_agent(
        algorithm="reactive",
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.1,
        exploration_decay=0.99,
        min_exploration_rate=0.01,
        seed=11,
    )

    assert isinstance(agent, ThresholdAgent)
    assert metadata["effective_algorithm"] == "threshold"
    assert normalize_algorithm("heuristic") == "threshold"


def test_threshold_checkpoint_preserves_sequence(tmp_path: Path):
    agent = ThresholdAgent(ThresholdConfig())
    agent.select_action(_state())
    target = tmp_path / "threshold.json"
    agent.save(str(target))

    restored = ThresholdAgent(ThresholdConfig())
    restored.load(str(target))

    assert restored.select_action(_state()).index == 3
