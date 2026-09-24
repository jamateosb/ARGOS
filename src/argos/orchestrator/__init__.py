# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Orchestrator package — public API surface.

This package wires together core domain models, the RL subsystem, the
orchestration loop, and experiment logging.

Key entry points:
    - :class:`OrchestrationLoop` — central polling + RL loop.
    - :class:`OrchestratorAPI` — FastAPI application (see ``api.py``).
    - :mod:`argos.orchestrator.rl` — algorithm-agnostic RL agents.
"""

from argos.domain.architecture import ArchitectureState, ServiceMetrics
from argos.domain.monitor import ArchitectureMonitor
from argos.domain.requests import AnalyticsRequest, EffectiveConfiguration
from argos.domain.requirements import AnalyticsRequirements, RequirementAction
from argos.orchestrator.coverage import CoverageCalculator, CoverageResult
from argos.orchestrator.environment import RANGE_ACTIONS, OrchestrationEnvironment, RangeAdjustmentAction
from argos.orchestrator.logger import ExperimentLogger, ExperimentLoggerConfig
from argos.orchestrator.loop import NodeInfo, OrchestrationLoop, OrchestrationLoopConfig
from argos.orchestrator.rl import RLAgent, RLConfig, TabularQLearningAgent

__all__ = [
    # Core models
    "ArchitectureState",
    "ServiceMetrics",
    "AnalyticsRequirements",
    "RequirementAction",
    "AnalyticsRequest",
    "EffectiveConfiguration",
    "ArchitectureMonitor",
    # RL (preferred)
    "RLAgent",
    "TabularQLearningAgent",
    "RLConfig",
    # Environment
    "OrchestrationEnvironment",
    "RangeAdjustmentAction",
    "RANGE_ACTIONS",
    # Orchestration loop
    "OrchestrationLoop",
    "OrchestrationLoopConfig",
    "NodeInfo",
    # Logging and metrics
    "ExperimentLogger",
    "ExperimentLoggerConfig",
    "CoverageCalculator",
    "CoverageResult",
]
