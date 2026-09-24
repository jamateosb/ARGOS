# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Factory for selecting RL backends per execution/request."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from argos.domain.requests import SUPPORTED_RL_ALGORITHMS
from argos.orchestrator.rl.base import RLAction
from argos.orchestrator.rl.qlearning import QLearningConfig, TabularQLearningAgent
from argos.orchestrator.rl.torch_common import torch_available


def normalize_algorithm(name: Optional[str]) -> str:
    """Normalize algorithm identifiers to the public stable names."""
    raw = (name or "qlearning").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "q": "qlearning",
        "qlearning": "qlearning",
        "dqn": "dqn",
        "ppo": "ppo",
        "static": "static",
        "fixed": "static",
        "baseline": "static",
        "noop": "static",
        "threshold": "threshold",
        "reactive": "threshold",
        "heuristic": "threshold",
    }
    normalized = aliases.get(raw)
    if normalized not in SUPPORTED_RL_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{name}'. Expected one of {SUPPORTED_RL_ALGORITHMS}")
    return normalized


def is_torch_available() -> bool:
    """Expose optional torch availability for tests and CLI checks."""
    return torch_available()


def create_agent(
    *,
    algorithm: str,
    learning_rate: float,
    discount_factor: float,
    exploration_rate: float,
    exploration_decay: float,
    min_exploration_rate: float,
    exploration_decay_steps: int = 0,
    seed: Optional[int] = None,
    actions: Optional[list[RLAction]] = None,
) -> tuple[object, dict[str, Optional[str]]]:
    """Create the requested agent, falling back safely when optional extras are missing."""
    requested = normalize_algorithm(algorithm)
    effective = requested
    fallback_reason = None

    if requested == "static":
        from argos.orchestrator.rl.static import StaticAgent, StaticConfig

        agent = StaticAgent(
            StaticConfig(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate,
                exploration_decay=exploration_decay,
                min_exploration_rate=min_exploration_rate,
                exploration_decay_steps=exploration_decay_steps,
                seed=seed,
            ),
            actions=actions,
        )
    elif requested == "threshold":
        from argos.orchestrator.rl.threshold import ThresholdAgent, ThresholdConfig

        agent = ThresholdAgent(
            ThresholdConfig(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate,
                exploration_decay=exploration_decay,
                min_exploration_rate=min_exploration_rate,
                exploration_decay_steps=exploration_decay_steps,
                seed=seed,
            ),
            actions=actions,
        )
    elif requested == "qlearning":
        agent = TabularQLearningAgent(
            QLearningConfig(
                alpha=learning_rate,
                gamma=discount_factor,
                epsilon=exploration_rate,
                exploration_decay=exploration_decay,
                min_exploration_rate=min_exploration_rate,
                exploration_decay_steps=exploration_decay_steps,
                seed=seed,
            ),
            actions=actions,
        )
    elif requested == "dqn":
        if not is_torch_available():
            fallback_reason = "Torch backend unavailable; falling back to qlearning"
            effective = "qlearning"
            agent = TabularQLearningAgent(
                QLearningConfig(
                    alpha=learning_rate,
                    gamma=discount_factor,
                    epsilon=exploration_rate,
                    exploration_decay=exploration_decay,
                    min_exploration_rate=min_exploration_rate,
                    exploration_decay_steps=exploration_decay_steps,
                    seed=seed,
                ),
                actions=actions,
            )
        else:
            from argos.orchestrator.rl.dqn import DQNAgent, DQNConfig

            agent = DQNAgent(
                DQNConfig(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_rate=exploration_rate,
                    exploration_decay=exploration_decay,
                    min_exploration_rate=min_exploration_rate,
                    exploration_decay_steps=exploration_decay_steps,
                    seed=seed,
                ),
                actions=actions,
            )
    else:  # ppo
        if not is_torch_available():
            fallback_reason = "Torch backend unavailable; falling back to qlearning"
            effective = "qlearning"
            agent = TabularQLearningAgent(
                QLearningConfig(
                    alpha=learning_rate,
                    gamma=discount_factor,
                    epsilon=exploration_rate,
                    exploration_decay=exploration_decay,
                    min_exploration_rate=min_exploration_rate,
                    exploration_decay_steps=exploration_decay_steps,
                    seed=seed,
                ),
                actions=actions,
            )
        else:
            from argos.orchestrator.rl.ppo import PPOAgent, PPOConfig

            agent = PPOAgent(
                PPOConfig(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    exploration_rate=exploration_rate,
                    exploration_decay=exploration_decay,
                    min_exploration_rate=min_exploration_rate,
                    exploration_decay_steps=exploration_decay_steps,
                    seed=seed,
                ),
                actions=actions,
            )

    agent.set_runtime_metadata(
        requested_algorithm=requested,
        fallback_reason=fallback_reason,
    )
    metadata = {
        "requested_algorithm": requested,
        "effective_algorithm": effective,
        "fallback_reason": fallback_reason,
        "policy_version": agent.get_policy_metadata()["policy_version"],
    }
    return agent, metadata


def build_policy_filename(request_id: str, algorithm: str, extension: str) -> str:
    """Generate a stable artifact filename for persisted policies."""
    return f"agent_{request_id}_{normalize_algorithm(algorithm)}{extension}"


def find_latest_policy(base_dir: str, algorithm: str, extension: str) -> Optional[Path]:
    """Locate the newest saved policy for one algorithm."""
    root = Path(base_dir)
    if not root.exists():
        return None

    normalized = normalize_algorithm(algorithm)
    candidates = sorted(
        root.glob(f"agent_*_{normalized}{extension}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    if normalized == "qlearning":
        tabular = sorted(
            root.glob("agent_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if tabular:
            return tabular[0]

    return None
