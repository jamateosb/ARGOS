# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Deterministic threshold baseline for multidimensional elasticity."""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.settings import RL_STATE_KEYS
from argos.orchestrator.rl.base import STANDARD_ACTIONS, RLAction, RLAgent, RLConfig

CPU_INDEX = RL_STATE_KEYS.index("cpu")
MEMORY_INDEX = RL_STATE_KEYS.index("memory")
PENDING_INDEX = RL_STATE_KEYS.index("pending_requests")
VIOLATION_INDEX = RL_STATE_KEYS.index("recent_slo_violations")
SERVICE_DUTY_INDEX = RL_STATE_KEYS.index("service_duty")


@dataclass(frozen=True)
class ThresholdConfig(RLConfig):
    """Configuration for the non-learning reactive baseline."""

    pressure_bucket: int = 3
    algorithm: str = "threshold"
    policy_version: str = "threshold-v2"


class ThresholdAgent(RLAgent[Hashable, RLAction]):
    """React to pressure with a fixed, reproducible quality-control rule.

    The encoded state starts with CPU and memory buckets. Under pressure, the
    policy cycles through sample reduction, coverage reduction, and freshness
    relaxation. Otherwise it cycles through the inverse quality improvements.
    No reward is used to alter the rule.
    """

    def __init__(self, config: ThresholdConfig, actions: Optional[list[RLAction]] = None) -> None:
        super().__init__(config)
        self.threshold_config = config
        self._actions = actions or STANDARD_ACTIONS
        self._selection_count = 0
        self._last_was_exploration = False
        self.set_runtime_metadata(backend="python")

    @property
    def last_was_exploration(self) -> bool:
        return False

    @property
    def epsilon(self) -> float:
        return 0.0

    def decay_exploration(self) -> None:
        return None

    def reset_run_metrics(self) -> None:
        """Reset accounting and the deterministic round-robin phase."""
        super().reset_run_metrics()
        self._selection_count = 0

    @staticmethod
    def _state_values(state: Hashable) -> tuple[int, ...]:
        if isinstance(state, tuple):
            return tuple(int(value) for value in state)
        if isinstance(state, list):
            return tuple(int(value) for value in state)
        return ()

    def select_action(
        self,
        state: Hashable,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> RLAction:
        del training
        values = self._state_values(state)
        cpu_bucket = values[CPU_INDEX] if len(values) > CPU_INDEX else 0
        memory_bucket = values[MEMORY_INDEX] if len(values) > MEMORY_INDEX else 0
        duty_bucket = values[SERVICE_DUTY_INDEX] if len(values) > SERVICE_DUTY_INDEX else 0
        pending_bucket = values[PENDING_INDEX] if len(values) > PENDING_INDEX else 0
        violation_bucket = values[VIOLATION_INDEX] if len(values) > VIOLATION_INDEX else 0
        pressured = (
            max(cpu_bucket, memory_bucket, duty_bucket) >= self.threshold_config.pressure_bucket
            or pending_bucket > 1
            or violation_bucket > 0
        )

        pressure_actions = (4, 2, 5)
        quality_actions = (1, 3, 6)
        sequence = pressure_actions if pressured else quality_actions
        action_idx = sequence[self._selection_count % len(sequence)]
        self._selection_count += 1
        allowed = set(allowed_action_indices or range(len(self._actions)))
        return self._actions[action_idx if action_idx in allowed else 0]

    def update(
        self,
        state: Hashable,
        action: RLAction,
        reward: float,
        next_state: Hashable,
        done: bool = False,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
        next_allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> dict[str, float]:
        del state, action, next_state, done
        self.step_complete(reward)
        return {"reward": float(reward), "td_error": 0.0}

    def get_policy(self, state: Hashable) -> dict[RLAction, float]:
        values = self._state_values(state)
        cpu_bucket = values[CPU_INDEX] if len(values) > CPU_INDEX else 0
        memory_bucket = values[MEMORY_INDEX] if len(values) > MEMORY_INDEX else 0
        duty_bucket = values[SERVICE_DUTY_INDEX] if len(values) > SERVICE_DUTY_INDEX else 0
        pending_bucket = values[PENDING_INDEX] if len(values) > PENDING_INDEX else 0
        violation_bucket = values[VIOLATION_INDEX] if len(values) > VIOLATION_INDEX else 0
        pressured = (
            max(cpu_bucket, memory_bucket, duty_bucket) >= self.threshold_config.pressure_bucket
            or pending_bucket > 1
            or violation_bucket > 0
        )
        valid = {4, 2, 5} if pressured else {1, 3, 6}
        probability = 1.0 / len(valid)
        return {action: (probability if action.index in valid else 0.0) for action in self._actions}

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "policy": "threshold",
            "policy_metadata": self.get_policy_metadata(),
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "selection_count": self._selection_count,
            "pressure_bucket": self.threshold_config.pressure_bucket,
        }
        save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.validate_policy_metadata(payload.get("policy_metadata", {}))
        self._episode_count = int(payload.get("episode_count", 0))
        self._step_count = int(payload.get("step_count", 0))
        self._total_reward = float(payload.get("total_reward", 0.0))
        self._selection_count = int(payload.get("selection_count", 0))

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        metrics.update({"epsilon": 0.0, "states_visited": self.step_count})
        return metrics
