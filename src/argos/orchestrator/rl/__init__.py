# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Reinforcement Learning module for adaptive orchestration.

This module provides an abstract interface for RL-based decision making,
allowing different algorithms (Q-Learning, PPO, DQN, etc.) to be plugged
into the orchestration loop.

Current implementations are reference implementations for experimentation and
benchmark-driven selection.
"""

from argos.orchestrator.rl.base import (
    RLAction,
    RLAgent,
    RLConfig,
    RLExperience,
    RLState,
)
from argos.orchestrator.rl.factory import create_agent, is_torch_available, normalize_algorithm
from argos.orchestrator.rl.qlearning import TabularQLearningAgent
from argos.orchestrator.rl.selection import resolve_algorithm, select_best_policy_from_benchmark
from argos.orchestrator.rl.static import StaticAgent, StaticConfig
from argos.orchestrator.rl.threshold import ThresholdAgent, ThresholdConfig

__all__ = [
    "RLAgent",
    "RLConfig",
    "RLState",
    "RLAction",
    "RLExperience",
    "TabularQLearningAgent",
    "StaticAgent",
    "StaticConfig",
    "ThresholdAgent",
    "ThresholdConfig",
    "create_agent",
    "is_torch_available",
    "normalize_algorithm",
    "resolve_algorithm",
    "select_best_policy_from_benchmark",
]
