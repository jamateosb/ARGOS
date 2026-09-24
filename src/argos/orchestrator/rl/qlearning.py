# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Q-Learning implementation for ARGOS orchestration.

This is one possible implementation of the RLAgent interface.
Other algorithms (PPO, DQN, etc.) can be implemented similarly.

Note: The production RL algorithm should be selected from reproducible
benchmarks, latency constraints and operational risk.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argos.orchestrator.rl.base import (
    STANDARD_ACTIONS,
    RLAction,
    RLAgent,
    RLConfig,
)


@dataclass(frozen=True)
class QLearningConfig(RLConfig):
    """Configuration specific to Q-Learning."""

    # Q-Learning specific parameters
    alpha: float = 0.1  # Learning rate (alias of learning_rate)
    gamma: float = 0.95  # Discount factor (same as discount_factor)
    epsilon: float = 0.1  # Exploration rate (same as exploration_rate)
    optimistic_q_value: float = 3.0

    # Eligibility traces (for Q(λ))
    use_eligibility_traces: bool = False
    lambda_trace: float = 0.9

    algorithm: str = "qlearning"
    policy_version: str = "qlearning-v3"


class TabularQLearningAgent(RLAgent[Hashable, RLAction]):
    """
    Tabular Q-Learning agent for discrete state/action spaces.

    This implementation uses a dictionary-based Q-table, suitable for
    discretized state spaces. For continuous states, consider using
    function approximation methods (DQN, etc.).
    """

    def __init__(
        self,
        config: QLearningConfig,
        actions: Optional[list[RLAction]] = None,
    ):
        """
        Initialize the Q-Learning agent.

        Args:
            config: Q-Learning configuration.
            actions: List of available actions. Defaults to STANDARD_ACTIONS.
        """
        super().__init__(config)
        self.qconfig = config
        self._actions = actions or STANDARD_ACTIONS
        self._num_actions = len(self._actions)

        # Q-table: state -> {action_index -> Q-value}
        self._q_table: dict[Hashable, dict[int, float]] = defaultdict(
            lambda: {i: self.qconfig.optimistic_q_value for i in range(self._num_actions)}
        )

        # Eligibility traces (if enabled)
        self._e_traces: dict[Hashable, dict[int, float]] = defaultdict(
            lambda: {i: 0.0 for i in range(self._num_actions)}
        )

        # Current exploration rate (mutable)
        self._epsilon = config.epsilon
        # Sync base class rate so get_metrics() is consistent
        self._exploration_rate = config.epsilon

        # Track whether the last select_action() call was exploratory
        self._last_was_exploration: bool = False

        self._rng = random.Random(config.seed)
        self.set_runtime_metadata(backend="python")

    def select_action(
        self,
        state: Hashable,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> RLAction:
        """
        Select an action using ε-greedy policy.

        Args:
            state: Current state (must be hashable).
            training: If True, uses ε-greedy. If False, purely greedy.

        Returns:
            Selected action.
        """
        allowed = tuple(sorted(set(allowed_action_indices or range(self._num_actions)))) or (0,)
        if training and self._rng.random() < self._epsilon:
            # Explore: random action
            self._last_was_exploration = True
            return self._actions[self._rng.choice(allowed)]

        # Exploit: best action according to Q-values
        self._last_was_exploration = False
        q_values = self._q_table[state] if training else self._q_table.get(state)
        if q_values is None:
            q_values = {i: self.qconfig.optimistic_q_value for i in range(self._num_actions)}
        best_value = max(q_values[index] for index in allowed)
        tied = [
            index
            for index in allowed
            if math.isclose(q_values[index], best_value, rel_tol=1e-12, abs_tol=1e-12)
        ]
        best_action_idx = self._rng.choice(tied)
        return self._actions[best_action_idx]

    @property
    def last_was_exploration(self) -> bool:
        """Whether the most recent :meth:`select_action` call explored."""
        return self._last_was_exploration

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
        """
        Update Q-table using the Bellman equation.

        Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]

        Args:
            state: Previous state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode terminated.

        Returns:
            Training metrics including TD error.
        """
        action_idx = action.index if hasattr(action, "index") else self._actions.index(action)

        q_values = self._q_table[state]
        next_q_values = self._q_table[next_state]

        # Current Q-value
        current_q = q_values[action_idx]

        # Target Q-value
        if done:
            target = reward
        else:
            next_allowed = next_allowed_action_indices or tuple(next_q_values)
            best_next_q = max(next_q_values[index] for index in next_allowed)
            target = reward + self.qconfig.gamma * best_next_q

        # TD error
        td_error = target - current_q

        # Update Q-value
        new_q = current_q + self.qconfig.alpha * td_error
        q_values[action_idx] = new_q

        # Track metrics
        self.step_complete(reward)

        return {
            "td_error": td_error,
            "q_value": new_q,
            "reward": reward,
        }

    def get_policy(self, state: Hashable) -> dict[RLAction, float]:
        """
        Get Q-values for all actions in the given state.

        Args:
            state: State to query.

        Returns:
            Dictionary mapping actions to their Q-values.
        """
        q_values = self._q_table.get(state)
        if q_values is None:
            q_values = {i: self.qconfig.optimistic_q_value for i in range(self._num_actions)}
        return {self._actions[i]: q_values[i] for i in range(self._num_actions)}

    def decay_exploration(self) -> None:
        """Apply epsilon decay."""
        self._epsilon = self._scheduled_exploration_rate(
            initial=self.qconfig.epsilon,
            current=self._epsilon,
        )
        # Keep base class in sync
        self._exploration_rate = self._epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Manually set exploration rate."""
        self._epsilon = max(0.0, min(1.0, epsilon))
        self._exploration_rate = self._epsilon

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    @property
    def state_count(self) -> int:
        """Number of unique states in Q-table."""
        return len(self._q_table)

    def get_q_value(self, state: Hashable, action: RLAction) -> float:
        """Get Q-value for a specific state-action pair."""
        action_idx = action.index if hasattr(action, "index") else self._actions.index(action)
        q_values = self._q_table.get(state)
        return q_values[action_idx] if q_values is not None else self.qconfig.optimistic_q_value

    def get_best_action(self, state: Hashable) -> RLAction:
        """Get the greedy best action for a state."""
        q_values = self._q_table.get(state)
        if q_values is None:
            q_values = {i: self.qconfig.optimistic_q_value for i in range(self._num_actions)}
        best_idx = max(q_values, key=q_values.get)
        return self._actions[best_idx]

    def save(self, path: str) -> None:
        """
        Save Q-table to JSON file.

        Args:
            path: File path to save to.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Q-table to serializable format
        serializable_q = {str(state): values for state, values in self._q_table.items()}

        data = {
            "q_table": serializable_q,
            "epsilon": self._epsilon,
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "policy_metadata": self.get_policy_metadata(),
            "config": {
                "alpha": self.qconfig.alpha,
                "gamma": self.qconfig.gamma,
                "epsilon": self.qconfig.epsilon,
                "optimistic_q_value": self.qconfig.optimistic_q_value,
                "exploration_decay_steps": self.qconfig.exploration_decay_steps,
                "algorithm": self.qconfig.algorithm,
                "policy_version": self.qconfig.policy_version,
            },
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load Q-table from JSON file.

        Args:
            path: File path to load from.
        """
        with open(path) as f:
            data = json.load(f)

        policy_metadata = data.get("policy_metadata", {})
        self.validate_policy_metadata(policy_metadata)

        # Restore Q-table
        self._q_table = defaultdict(
            lambda: {i: self.qconfig.optimistic_q_value for i in range(self._num_actions)}
        )
        for state_str, values in data["q_table"].items():
            # Convert string keys back to tuple if possible
            try:
                state = ast.literal_eval(state_str)
            except (ValueError, SyntaxError):
                state = state_str
            self._q_table[state] = {int(k): v for k, v in values.items()}

        # Restore metadata
        self._epsilon = data.get("epsilon", self.qconfig.epsilon)
        self._exploration_rate = self._epsilon
        self._episode_count = data.get("episode_count", 0)
        self._step_count = data.get("step_count", 0)
        self._total_reward = data.get("total_reward", 0.0)
        self.set_runtime_metadata(
            requested_algorithm=policy_metadata.get("requested_algorithm", self.qconfig.algorithm),
            fallback_reason=policy_metadata.get("fallback_reason"),
            backend=policy_metadata.get("backend", "python"),
        )

    def table_snapshot(self) -> dict[Hashable, dict[int, float]]:
        """Return a shallow copy of the current Q-table."""
        return dict(self._q_table)

    def policy_fingerprint(self) -> str:
        """Hash only the learned Q-table, excluding run counters."""
        payload = {
            str(state): {str(index): value for index, value in sorted(values.items())}
            for state, values in sorted(self._q_table.items(), key=lambda item: str(item[0]))
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get_metrics(self) -> dict[str, Any]:
        """Get current training metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update(
            {
                "epsilon": self._epsilon,
                "states_visited": self.state_count,
                "alpha": self.qconfig.alpha,
                "gamma": self.qconfig.gamma,
            }
        )
        return base_metrics

