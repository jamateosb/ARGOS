# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Analytics requirements models (coverage, sample, freshness, etc.)."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from argos.settings import STATE_BINS, ActionSteps, RequirementBounds
from argos.domain.numeric_contract import canonical_contract_value


class RequirementAction(Enum):
    """
    Discrete actions that tweak analytics requirements.
    RL can consume these, but the data collector can also expose them for manual control.
    """

    INCREASE_COVERAGE = auto()
    DECREASE_COVERAGE = auto()
    INCREASE_SAMPLE = auto()
    DECREASE_SAMPLE = auto()
    MAKE_FRESHER = auto()  # reduce the refresh interval (seconds) -> fresher data
    MAKE_STALER = auto()  # increase the refresh interval -> less fresh
    HOLD = auto()


def bucketize(value: float, edges: Sequence[float]) -> int:
    """Convert a numeric value into a bucket index using binary search.

    Uses ``bisect_left`` so a value exactly on an edge remains in the lower
    bucket. Inputs and edges must be finite and edges must be sorted.
    """
    observed = float(value)
    normalized_edges = tuple(float(edge) for edge in edges)
    if not math.isfinite(observed):
        raise ValueError(f"Bucket value must be finite, got {value!r}")
    if any(not math.isfinite(edge) for edge in normalized_edges):
        raise ValueError("Bucket edges must be finite")
    if any(left > right for left, right in zip(normalized_edges, normalized_edges[1:])):
        raise ValueError("Bucket edges must be sorted")
    return bisect_left(normalized_edges, observed)


@dataclass
class AnalyticsRequirements:
    """
    Data analytics requirements tracked by the orchestrator or nodes.
    - coverage: fraction of locations/devices to include (0.0-1.0).
    - sample: fraction of data points to include from selected locations (0.0-1.0).
    - freshness_seconds: max acceptable age of the data; lower means fresher.
    - response_time_s: optional response-time requirement.
    - cost_budget: optional budget marker.
    """

    coverage: float = 0.6
    sample: float = 0.5
    freshness_seconds: float = 60.0
    response_time_s: Optional[float] = None
    cost_budget: Optional[float] = None

    def clamp(self, bounds: RequirementBounds) -> None:
        """Clamp requirements within allowed bounds."""
        self.coverage = canonical_contract_value(self.coverage, bounds.coverage)
        self.sample = canonical_contract_value(self.sample, bounds.sample)
        self.freshness_seconds = canonical_contract_value(
            self.freshness_seconds,
            bounds.freshness_seconds,
        )

    def apply_action(self, action: RequirementAction, bounds: RequirementBounds, steps: ActionSteps) -> None:
        """Apply an action to the requirements and clamp the result."""
        if action == RequirementAction.INCREASE_COVERAGE:
            self.coverage += steps.coverage
        elif action == RequirementAction.DECREASE_COVERAGE:
            self.coverage -= steps.coverage
        elif action == RequirementAction.INCREASE_SAMPLE:
            self.sample += steps.sample
        elif action == RequirementAction.DECREASE_SAMPLE:
            self.sample -= steps.sample
        elif action == RequirementAction.MAKE_FRESHER:
            self.freshness_seconds -= steps.freshness_seconds
        elif action == RequirementAction.MAKE_STALER:
            self.freshness_seconds += steps.freshness_seconds
        self.clamp(bounds)

    def as_vector(self) -> dict[str, float]:
        """Return requirements as a numeric vector."""
        return {
            "coverage": self.coverage,
            "sample": self.sample,
            "freshness_seconds": self.freshness_seconds,
        }

    def as_state_tuple(self) -> tuple[int, int, int]:
        """Discretize requirements to a compact tuple for the Q-table."""
        return (
            bucketize(self.coverage, STATE_BINS["coverage"]),
            bucketize(self.sample, STATE_BINS["sample"]),
            bucketize(self.freshness_seconds, STATE_BINS["freshness"]),
        )

    def freshness_pressure(self) -> float:
        """Higher when the requested freshness is aggressive (low seconds)."""
        return 1.0 / max(self.freshness_seconds, 1.0)

    def estimated_cost(self) -> float:
        """
        Estimate cost/effort caused by the requirements.
        This connects coverage/sample/freshness with architecture usage.
        """
        return self.coverage * self.sample * (1.0 + self.freshness_pressure())

    def requirement_quality(self, target: RequirementBounds) -> float:
        """
        Score how well the requirements satisfy the desired lower bounds.
        Values are clipped to be non-negative.
        """
        coverage_score = (self.coverage - target.coverage[0]) / max(target.coverage[1] - target.coverage[0], 1e-6)
        sample_score = (self.sample - target.sample[0]) / max(target.sample[1] - target.sample[0], 1e-6)

        if self.freshness_seconds <= target.freshness_seconds[0]:
            freshness_score = 1.0
        else:
            freshness_score = target.freshness_seconds[0] / max(self.freshness_seconds, 1e-6)

        return max(0.0, coverage_score + sample_score + freshness_score) / 3.0
