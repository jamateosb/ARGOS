# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for argos.orchestrator.coverage module - CoverageCalculator."""

import pytest

from argos.orchestrator.coverage import CoverageCalculator, CoverageResult


class TestCoverageResult:
    """Tests for CoverageResult dataclass."""

    def test_is_satisfied_when_met(self):
        """Test is_satisfied when coverage is met."""
        result = CoverageResult(
            total_nodes=10,
            required_active=5,
            current_active=6,
            coverage_target=0.5,
            coverage_achieved=0.6,
            deficit=-1,
        )
        assert result.is_satisfied is True

    def test_is_satisfied_when_not_met(self):
        """Test is_satisfied when coverage is not met."""
        result = CoverageResult(
            total_nodes=10,
            required_active=5,
            current_active=3,
            coverage_target=0.5,
            coverage_achieved=0.3,
            deficit=2,
        )
        assert result.is_satisfied is False

    def test_excess_when_surplus(self):
        """Test excess when there's a surplus of nodes."""
        result = CoverageResult(
            total_nodes=10,
            required_active=5,
            current_active=8,
            coverage_target=0.5,
            coverage_achieved=0.8,
            deficit=-3,
        )
        assert result.excess == 3

    def test_excess_when_deficit(self):
        """Test excess is zero when in deficit."""
        result = CoverageResult(
            total_nodes=10,
            required_active=5,
            current_active=3,
            coverage_target=0.5,
            coverage_achieved=0.3,
            deficit=2,
        )
        assert result.excess == 0


class TestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    def test_calculate_basic(self):
        """Test basic coverage calculation."""
        calc = CoverageCalculator()
        result = calc.calculate(total_nodes=10, coverage_target=0.5, current_active=5)

        assert result.total_nodes == 10
        assert result.required_active == 5
        assert result.current_active == 5
        assert result.coverage_achieved == pytest.approx(0.5)
        assert result.is_satisfied is True
        assert result.deficit == 0

    def test_calculate_with_deficit(self):
        """Test calculation with node deficit."""
        calc = CoverageCalculator()
        result = calc.calculate(total_nodes=10, coverage_target=0.7, current_active=3)

        assert result.required_active == 7
        assert result.deficit == 4
        assert result.is_satisfied is False

    def test_calculate_clamps_coverage(self):
        """Test that coverage target is clamped to [0, 1]."""
        calc = CoverageCalculator()

        result = calc.calculate(total_nodes=10, coverage_target=1.5)
        assert result.coverage_target == 1.0

        result = calc.calculate(total_nodes=10, coverage_target=-0.5)
        assert result.coverage_target == 0.0

    def test_calculate_minimum_one_node(self):
        """Test that at least one node is required."""
        calc = CoverageCalculator()
        result = calc.calculate(total_nodes=10, coverage_target=0.01)

        assert result.required_active >= 1

    def test_distribute_load_single_request(self):
        """Test load distribution for a single request."""
        calc = CoverageCalculator()
        allocation = calc.distribute_load(
            total_nodes=10,
            requests=[("req-1", 0.5)],
        )

        assert allocation["req-1"] == 5

    def test_distribute_load_multiple_requests(self):
        """Test load distribution across multiple requests."""
        calc = CoverageCalculator()
        allocation = calc.distribute_load(
            total_nodes=10,
            requests=[("req-1", 0.3), ("req-2", 0.4)],
        )

        assert allocation["req-1"] == 3
        assert allocation["req-2"] == 4

    def test_distribute_load_oversubscribed(self):
        """Test proportional allocation when oversubscribed."""
        calc = CoverageCalculator()
        allocation = calc.distribute_load(
            total_nodes=10,
            requests=[("req-1", 0.6), ("req-2", 0.6)],  # Total 1.2 > 1.0
        )

        # Should be proportional: each gets 50% of 10 = 5
        total_assigned = sum(allocation.values())
        assert total_assigned <= 10

    def test_distribute_load_empty(self):
        """Test empty request list."""
        calc = CoverageCalculator()
        allocation = calc.distribute_load(total_nodes=10, requests=[])

        assert allocation == {}

    def test_optimize_coverage_high_pressure(self):
        """Test coverage optimization under high resource pressure."""
        calc = CoverageCalculator()
        coverage = calc.optimize_coverage(
            coverage_range=(0.3, 0.8),
            current_load=0.8,
            resource_pressure=0.9,  # High pressure
        )

        # Should suggest lower coverage
        assert coverage < 0.6

    def test_optimize_coverage_low_pressure(self):
        """Test coverage optimization with low resource pressure."""
        calc = CoverageCalculator()
        coverage = calc.optimize_coverage(
            coverage_range=(0.3, 0.8),
            current_load=0.2,
            resource_pressure=0.1,  # Low pressure
        )

        # Should suggest higher coverage
        assert coverage > 0.5

    def test_optimize_coverage_medium_pressure(self):
        """Test coverage optimization with medium pressure."""
        calc = CoverageCalculator()
        coverage = calc.optimize_coverage(
            coverage_range=(0.3, 0.8),
            current_load=0.5,
            resource_pressure=0.5,  # Medium
        )

        # Should be near midpoint
        assert 0.4 <= coverage <= 0.7

    def test_elasticity_score_perfect(self):
        """Test elasticity score with perfect metrics."""
        calc = CoverageCalculator()
        score = calc.elasticity_score(
            coverage_achieved=0.8,
            coverage_target=0.8,
            response_time_ms=50.0,
            target_response_ms=100.0,
        )

        # Perfect coverage (0.5) + good response (0.5) = 1.0
        assert score == pytest.approx(1.0)

    def test_elasticity_score_poor_coverage(self):
        """Test elasticity score with poor coverage."""
        calc = CoverageCalculator()
        score = calc.elasticity_score(
            coverage_achieved=0.4,
            coverage_target=0.8,
            response_time_ms=50.0,
            target_response_ms=100.0,
        )

        # Coverage component: 0.4/0.8 * 0.5 = 0.25
        # Response component: 0.5
        assert score == pytest.approx(0.75)

    def test_elasticity_score_poor_response(self):
        """Test elasticity score with poor response time."""
        calc = CoverageCalculator()
        score = calc.elasticity_score(
            coverage_achieved=0.8,
            coverage_target=0.8,
            response_time_ms=200.0,
            target_response_ms=100.0,
        )

        # Coverage component: 0.5
        # Response component: 100/200 * 0.5 = 0.25
        assert score == pytest.approx(0.75)
