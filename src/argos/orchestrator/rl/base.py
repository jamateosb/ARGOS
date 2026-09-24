# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Abstract base classes for Reinforcement Learning agents.

This module defines the interface that any RL algorithm must implement
to work with the ARGOS orchestration system. The design is intentionally
algorithm-agnostic to allow experimentation with different approaches:
- Tabular methods (Q-Learning, SARSA)
- Function approximation (DQN, Double DQN)
- Policy gradient (PPO, A2C, REINFORCE)
- Actor-Critic methods

The production algorithm should be selected from reproducible benchmarks,
latency constraints and operational risk.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from argos.settings import DEFAULT_STEPS

# Generic type variables for state and action spaces
StateType = TypeVar("StateType", bound=Hashable)
ActionType = TypeVar("ActionType")


@dataclass(frozen=True)
class RLConfig:
    """
    Base configuration for RL agents.

    Subclasses may extend this with algorithm-specific parameters.
    """

    # Learning parameters (common to most algorithms)
    learning_rate: float = 0.1
    discount_factor: float = 0.95

    # Exploration parameters
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    exploration_decay_steps: int = 0

    # Random seed for reproducibility
    seed: Optional[int] = None

    # Training parameters
    batch_size: int = 32
    memory_size: int = 10000

    # Algorithm identifier (for logging/serialization)
    algorithm: str = "generic"
    policy_version: str = "generic-v1"
    state_schema_version: str = "argos.mdp.v4"
    action_schema_version: str = "argos.actions.masked-discrete.v2"


@dataclass
class RLState:
    """
    Representation of the orchestration state for RL.

    This is a flexible container that can hold different state
    representations depending on the algorithm's needs.
    """

    # Resource metrics (normalized 0-1)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0

    # Analytics requirements (current effective values)
    coverage: float = 0.0
    sample_rate: float = 0.0
    freshness: float = 0.0

    # Quality metrics
    response_time_ratio: float = 1.0  # actual/target
    cost_ratio: float = 1.0  # actual/budget

    # Context
    active_nodes: int = 0
    total_nodes: int = 0
    active_requests: int = 0

    # Additional features (algorithm-specific)
    extra: dict[str, float] = field(default_factory=dict)

    def to_tuple(self) -> tuple:
        """Convert to hashable tuple for tabular methods."""
        return (
            round(self.cpu_utilization, 2),
            round(self.memory_utilization, 2),
            round(self.coverage, 2),
            round(self.sample_rate, 2),
            round(self.freshness / 60, 1),  # Normalize to minutes
            self.active_nodes,
        )

    def to_vector(self) -> list[float]:
        """Convert to feature vector for function approximation."""
        return [
            self.cpu_utilization,
            self.memory_utilization,
            self.network_utilization,
            self.coverage,
            self.sample_rate,
            self.freshness / 120.0,  # Normalize to [0, 1] assuming max 120s
            self.response_time_ratio,
            self.cost_ratio,
            self.active_nodes / max(self.total_nodes, 1),
            self.active_requests / 10.0,  # Normalize assuming max 10 requests
        ]


@dataclass
class RLAction:
    """
    Representation of an RL action.

    Actions modify the effective configuration within allowed ranges.
    """

    # Action type identifier
    action_type: str  # e.g., "increase_coverage", "decrease_sample", "hold"

    # Delta values (how much to change)
    coverage_delta: float = 0.0
    sample_delta: float = 0.0
    freshness_delta: float = 0.0

    # Target dimension (which parameter to adjust)
    target_dimension: Optional[str] = None

    # Action index (for discrete action spaces)
    index: int = 0

    def __hash__(self) -> int:
        return hash((self.action_type, self.index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RLAction):
            return False
        return self.action_type == other.action_type and self.index == other.index


@dataclass
class RLExperience:
    """
    A single experience tuple for learning.

    Used by experience replay and batch learning algorithms.
    """

    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class RLAgent(ABC, Generic[StateType, ActionType]):
    """
    Abstract base class for RL agents.

    Any RL algorithm used in ARGOS must implement this interface.
    This ensures consistent integration with the orchestration loop
    regardless of the underlying algorithm.

    Usage:
        agent = SomeRLAgent(config, action_space)

        # In orchestration loop:
        state = environment.get_state()
        action = agent.select_action(state)
        next_state, reward = environment.step(action)
        agent.update(state, action, reward, next_state)
    """

    def __init__(self, config: RLConfig):
        """
        Initialize the agent.

        Args:
            config: Configuration parameters for the agent.
        """
        self.config = config
        self._episode_count = 0
        self._step_count = 0
        self._total_reward = 0.0
        # Mutable exploration rate (config is frozen, so we track it separately)
        self._exploration_rate = config.exploration_rate
        self._requested_algorithm = config.algorithm
        self._fallback_reason: Optional[str] = None
        self._backend = "python"

    @property
    def policy_extension(self) -> str:
        """Default artifact extension for persisted policies."""
        return ".json"

    def set_runtime_metadata(
        self,
        *,
        requested_algorithm: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        """Attach runtime metadata such as fallback decisions."""
        if requested_algorithm:
            self._requested_algorithm = requested_algorithm
        self._fallback_reason = fallback_reason
        if backend:
            self._backend = backend

    def get_policy_metadata(self) -> dict[str, Any]:
        """Return algorithm/policy metadata shared by all backends."""
        return {
            "requested_algorithm": self._requested_algorithm,
            "algorithm": self.config.algorithm,
            "effective_algorithm": self.config.algorithm,
            "policy_version": self.config.policy_version,
            "state_schema_version": self.config.state_schema_version,
            "action_schema_version": self.config.action_schema_version,
            "seed": self.config.seed,
            "fallback_reason": self._fallback_reason,
            "backend": self._backend,
            "policy_extension": self.policy_extension,
        }

    def policy_fingerprint(self) -> str:
        """Return a stable fingerprint of a parameter-free policy."""
        payload = json.dumps(self.get_policy_metadata(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_policy_metadata(self, metadata: dict[str, Any]) -> None:
        """Reject checkpoints built for incompatible state or action schemas."""
        state_schema = metadata.get("state_schema_version")
        action_schema = metadata.get("action_schema_version")
        if state_schema != self.config.state_schema_version:
            raise ValueError(
                f"Incompatible state schema: checkpoint={state_schema!r}, "
                f"runtime={self.config.state_schema_version!r}"
            )
        if action_schema != self.config.action_schema_version:
            raise ValueError(
                f"Incompatible action schema: checkpoint={action_schema!r}, "
                f"runtime={self.config.action_schema_version!r}"
            )

    @abstractmethod
    def select_action(
        self,
        state: StateType,
        training: bool = True,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> ActionType:
        """
        Select an action for the given state.

        Args:
            state: Current environment state.
            training: If True, may include exploration. If False, purely greedy.

        Returns:
            The selected action.
        """
        pass

    @abstractmethod
    def update(
        self,
        state: StateType,
        action: ActionType,
        reward: float,
        next_state: StateType,
        done: bool = False,
        allowed_action_indices: Optional[tuple[int, ...]] = None,
        next_allowed_action_indices: Optional[tuple[int, ...]] = None,
    ) -> dict[str, float]:
        """
        Update the agent based on an experience.

        Args:
            state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            done: Whether episode terminated.

        Returns:
            Dictionary of training metrics (e.g., loss, td_error).
        """
        pass

    @abstractmethod
    def get_policy(self, state: StateType) -> dict[ActionType, float]:
        """
        Get the action probabilities/values for a state.

        Args:
            state: State to query.

        Returns:
            Dictionary mapping actions to probabilities or Q-values.
        """
        pass

    def decay_exploration(self) -> None:
        """Apply exploration decay (e.g., epsilon decay for ε-greedy)."""
        self._exploration_rate = self._scheduled_exploration_rate(
            initial=self.config.exploration_rate,
            current=self._exploration_rate,
        )

    def _scheduled_exploration_rate(self, *, initial: float, current: float) -> float:
        """Return horizon-calibrated epsilon or the multiplicative decay."""
        floor = max(0.0, float(self.config.min_exploration_rate))
        horizon = max(0, int(self.config.exploration_decay_steps))
        start = max(floor, float(initial))
        if horizon > 0 and start > floor:
            progress = min(1.0, max(0.0, self.step_count / horizon))
            if floor > 0.0:
                return max(floor, start * ((floor / start) ** progress))
            return max(0.0, start * (1.0 - progress))
        return max(floor, float(current) * self.config.exploration_decay)

    def episode_complete(self) -> None:
        """Signal that an episode has ended."""
        self._episode_count += 1
        self.decay_exploration()

    def end_trajectory(self, final_state: Optional[StateType] = None, done: bool = True) -> None:
        """Flush trajectory-specific state and mark the episode complete.

        Value-based and static agents have no pending rollout to flush. Policy
        gradient agents override this hook before delegating to the base
        implementation.
        """
        del final_state, done
        self.episode_complete()

    def step_complete(self, reward: float) -> None:
        """Signal that a step has completed."""
        self._step_count += 1
        self._total_reward += reward

    def reset_run_metrics(self) -> None:
        """Reset counters while preserving learned policy parameters."""
        self._episode_count = 0
        self._step_count = 0
        self._total_reward = 0.0

    @property
    def episode_count(self) -> int:
        """Number of completed episodes."""
        return self._episode_count

    @property
    def step_count(self) -> int:
        """Total number of steps taken."""
        return self._step_count

    @property
    def total_reward(self) -> float:
        """Cumulative reward across all episodes."""
        return self._total_reward

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's learned parameters.

        Args:
            path: File path to save to.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load previously saved parameters.

        Args:
            path: File path to load from.
        """
        pass

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current training metrics.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "episodes": self._episode_count,
            "steps": self._step_count,
            "total_reward": self._total_reward,
            "exploration_rate": self._exploration_rate,
            **self.get_policy_metadata(),
        }


# Standard action space for ARGOS orchestration
STANDARD_ACTIONS: list[RLAction] = [
    RLAction("hold", 0.0, 0.0, 0.0, None, 0),
    RLAction("increase_coverage", DEFAULT_STEPS.coverage, 0.0, 0.0, "coverage", 1),
    RLAction("decrease_coverage", -DEFAULT_STEPS.coverage, 0.0, 0.0, "coverage", 2),
    RLAction("increase_sample", 0.0, DEFAULT_STEPS.sample, 0.0, "sample", 3),
    RLAction("decrease_sample", 0.0, -DEFAULT_STEPS.sample, 0.0, "sample", 4),
    RLAction("decrease_freshness", 0.0, 0.0, DEFAULT_STEPS.freshness_seconds, "freshness", 5),
    RLAction("increase_freshness", 0.0, 0.0, -DEFAULT_STEPS.freshness_seconds, "freshness", 6),
]
