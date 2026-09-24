# =============================================================================
# ARGOS — Adaptive Reinforcement-driven Governance for Orchestrated Services
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Static (non-learning) baseline agent.

The static agent never adapts. It always returns the ``hold`` action,
which keeps the effective configuration pinned to whatever the
orchestrator set when admitting the request. Because the orchestrator
seeds every request with ``request.midpoint_config()``, this agent
effectively fixes the operating point at the midpoint of the client SLO
range for the entire request lifetime.

Used as a control in static-vs-dynamic comparisons: the same workload is
run once with ``algorithm="static"`` (no RL adaptation, fixed midpoint)
and once with a learning agent (``auto``, ``qlearning``, ``dqn`` or
``ppo``). The reward, overload-penalty count, SLO fulfillment and cost
deltas between the two campaigns measure how much value the RL adds.
"""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.orchestrator.rl.base import STANDARD_ACTIONS, RLAction, RLAgent, RLConfig


@dataclass(frozen=True)
class StaticConfig(RLConfig):
    """Configuration for the static baseline agent.

    The static agent ignores every learning hyperparameter, but the
    fields are kept so the same factory entry point can build it.
    """

    algorithm: str = "static"
    policy_version: str = "static-v1"


class StaticAgent(RLAgent[Hashable, RLAction]):
    """Baseline agent that never adapts the configuration.

    Always selects the ``hold`` action regardless of state, so the
    effective configuration stays at the midpoint of the client range
    for the entire request. ``update`` records the reward stream into
    the base class counters but never updates any policy.
    """

    def __init__(self, config: StaticConfig, actions: Optional[list[RLAction]] = None) -> None:
        super().__init__(config)
        self._actions = actions or STANDARD_ACTIONS
        self._hold_action = next(
            (a for a in self._actions if a.action_type == "hold"),
            self._actions[0],
        )
        self._last_was_exploration: bool = False
        self.set_runtime_metadata(backend="python")

    def select_action(
        self,
        state: Hashable,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> RLAction:
        self._last_was_exploration = False
        return self._hold_action

    @property
    def last_was_exploration(self) -> bool:
        return self._last_was_exploration

    @property
    def epsilon(self) -> float:
        return 0.0

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
        # Track reward stream so total_reward / step_count are comparable
        # with learning agents, but never update any policy.
        self.step_complete(reward)
        return {"reward": reward, "td_error": 0.0}

    def get_policy(self, state: Hashable) -> dict[RLAction, float]:
        return {action: (1.0 if action.action_type == "hold" else 0.0) for action in self._actions}

    def decay_exploration(self) -> None:
        # No exploration to decay.
        return None

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "policy": "static",
            "policy_metadata": self.get_policy_metadata(),
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "config": {
                "algorithm": self.config.algorithm,
                "policy_version": self.config.policy_version,
            },
        }
        with save_path.open("w") as handle:
            json.dump(data, handle, indent=2)

    def load(self, path: str) -> None:
        with open(path) as handle:
            data = json.load(handle)
        policy_metadata = data.get("policy_metadata", {})
        self.validate_policy_metadata(policy_metadata)
        self._episode_count = int(data.get("episode_count", 0))
        self._step_count = int(data.get("step_count", 0))
        self._total_reward = float(data.get("total_reward", 0.0))
        self.set_runtime_metadata(
            requested_algorithm=policy_metadata.get("requested_algorithm", self.config.algorithm),
            fallback_reason=policy_metadata.get("fallback_reason"),
            backend=policy_metadata.get("backend", "python"),
        )

    def get_metrics(self) -> dict[str, Any]:
        metrics = super().get_metrics()
        metrics.update({"epsilon": 0.0, "states_visited": 0})
        return metrics
