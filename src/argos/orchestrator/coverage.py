# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Coverage calculator for orchestration.

Implements the formula: Nodos_Activos = Total_Nodos * Coverage

Provides utilities for calculating how many nodes should be active
based on coverage requirements and current system state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CoverageResult:
    """Result of a coverage calculation."""

    total_nodes: int
    required_active: int
    current_active: int
    coverage_target: float
    coverage_achieved: float
    deficit: int  # How many more nodes needed (negative = excess)

    @property
    def is_satisfied(self) -> bool:
        """Check if coverage requirement is met."""
        return self.current_active >= self.required_active

    @property
    def excess(self) -> int:
        """How many nodes above minimum (0 if in deficit)."""
        return max(0, self.current_active - self.required_active)


class CoverageCalculator:
    """
    Calculate node coverage for analytics requests.

    Coverage represents the fraction of available nodes that should
    be actively processing a request. The orchestrator uses coverage
    along with sample_rate and freshness to optimize resource usage.
    """

    def calculate(
        self,
        total_nodes: int,
        coverage_target: float,
        current_active: int = 0,
    ) -> CoverageResult:
        """
        Calculate coverage metrics.

        Args:
            total_nodes: Total number of registered nodes.
            coverage_target: Desired fraction of nodes (0.0-1.0).
            current_active: Currently active nodes.

        Returns:
            CoverageResult with all calculated metrics.
        """
        coverage_target = max(0.0, min(1.0, coverage_target))
        required_active = max(1, math.ceil(total_nodes * coverage_target))

        coverage_achieved = current_active / max(total_nodes, 1)
        deficit = required_active - current_active

        return CoverageResult(
            total_nodes=total_nodes,
            required_active=required_active,
            current_active=current_active,
            coverage_target=coverage_target,
            coverage_achieved=coverage_achieved,
            deficit=deficit,
        )

    def distribute_load(
        self,
        total_nodes: int,
        requests: list[tuple[str, float]],  # (request_id, coverage)
    ) -> dict:
        """
        Distribute nodes across multiple requests with different coverage needs.

        Uses proportional allocation when total coverage > 1.0.

        Args:
            total_nodes: Total available nodes.
            requests: List of (request_id, coverage_fraction) tuples.

        Returns:
            Dict mapping request_id to number of assigned nodes.
        """
        if not requests:
            return {}

        total_coverage = sum(cov for _, cov in requests)

        if total_coverage <= 1.0:
            # Simple allocation - each gets what they asked for
            return {req_id: max(1, math.ceil(total_nodes * cov)) for req_id, cov in requests}

        # Proportional allocation when oversubscribed
        allocation = {}
        remaining = total_nodes

        for i, (req_id, cov) in enumerate(requests):
            if i == len(requests) - 1:
                # Last request gets remaining nodes
                allocation[req_id] = max(1, remaining)
            else:
                # Proportional share
                share = cov / total_coverage
                assigned = max(1, math.ceil(total_nodes * share))
                allocation[req_id] = min(assigned, remaining)
                remaining -= allocation[req_id]

        return allocation

    def optimize_coverage(
        self,
        coverage_range: tuple[float, float],
        current_load: float,
        resource_pressure: float,
    ) -> float:
        """
        Suggest optimal coverage within allowed range.

        Balances coverage against resource pressure:
        - High pressure → reduce coverage toward minimum
        - Low pressure → increase coverage toward maximum

        Args:
            coverage_range: (min, max) allowed coverage.
            current_load: Current system load (0.0-1.0).
            resource_pressure: Resource pressure metric (0.0+).

        Returns:
            Suggested coverage value within range.
        """
        min_cov, max_cov = coverage_range

        # Simple linear interpolation based on pressure
        # High pressure (> 0.7) → push toward minimum
        # Low pressure (< 0.3) → push toward maximum

        if resource_pressure > 0.7:
            # Under pressure - reduce coverage
            pressure_factor = min(1.0, (resource_pressure - 0.7) / 0.3)
            return max_cov - (max_cov - min_cov) * pressure_factor
        elif resource_pressure < 0.3:
            # Room to grow - increase coverage
            headroom_factor = (0.3 - resource_pressure) / 0.3
            return min_cov + (max_cov - min_cov) * headroom_factor
        else:
            # Middle ground - use midpoint
            return (min_cov + max_cov) / 2

    def elasticity_score(
        self,
        coverage_achieved: float,
        coverage_target: float,
        response_time_ms: float,
        target_response_ms: float = 100.0,
    ) -> float:
        """
        Calculate an elasticity score combining coverage and performance.

        Score ranges from 0.0 (poor) to 1.0 (excellent).

        Args:
            coverage_achieved: Actual coverage fraction.
            coverage_target: Desired coverage fraction.
            response_time_ms: Actual response time.
            target_response_ms: Desired response time.

        Returns:
            Elasticity score (0.0-1.0).
        """
        # Coverage component (50% weight)
        coverage_ratio = min(1.0, coverage_achieved / max(coverage_target, 0.01))
        coverage_score = coverage_ratio * 0.5

        # Response time component (50% weight)
        response_ratio = min(1.0, target_response_ms / max(response_time_ms, 1.0))
        response_score = response_ratio * 0.5

        return coverage_score + response_score
