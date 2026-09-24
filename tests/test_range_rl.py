# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for range-based environment features."""

import pytest

from argos.settings import SamplingConfig
from argos.domain.monitor import ArchitectureMonitor
from argos.domain.requests import AnalyticsRequest, EffectiveConfiguration
from argos.orchestrator.environment import (
    RANGE_ACTIONS,
    OrchestrationEnvironment,
    RangeAdjustmentAction,
)
from argos.orchestrator.rl.base import STANDARD_ACTIONS


class TestRangeAdjustmentAction:
    """Tests for RangeAdjustmentAction dataclass."""

    def test_default_values(self):
        """Test default action is no-op."""
        action = RangeAdjustmentAction()
        assert action.coverage_delta == 0.0
        assert action.sample_delta == 0.0
        assert action.freshness_delta == 0.0

    def test_custom_values(self):
        """Test custom delta values."""
        action = RangeAdjustmentAction(
            coverage_delta=0.1,
            sample_delta=-0.05,
            freshness_delta=5.0,
        )
        assert action.coverage_delta == 0.1
        assert action.sample_delta == -0.05
        assert action.freshness_delta == 5.0


class TestRangeActions:
    """Tests for the RANGE_ACTIONS list."""

    def test_range_actions_defined(self):
        """Test that range actions are defined."""
        assert len(RANGE_ACTIONS) > 0

    def test_first_action_is_noop(self):
        """Test that first action is no-op."""
        noop = RANGE_ACTIONS[0]
        assert noop.coverage_delta == 0.0
        assert noop.sample_delta == 0.0
        assert noop.freshness_delta == 0.0

    def test_has_coverage_actions(self):
        """Test that there are coverage adjustment actions."""
        coverage_actions = [a for a in RANGE_ACTIONS if a.coverage_delta != 0.0]
        assert len(coverage_actions) >= 2  # At least inc and dec

    def test_has_sample_actions(self):
        """Test that there are sample adjustment actions."""
        sample_actions = [a for a in RANGE_ACTIONS if a.sample_delta != 0.0]
        assert len(sample_actions) >= 2

    def test_has_freshness_actions(self):
        """Test that there are freshness adjustment actions."""
        freshness_actions = [a for a in RANGE_ACTIONS if a.freshness_delta != 0.0]
        assert len(freshness_actions) >= 2

    def test_range_actions_share_deltas_with_agent_action_space(self):
        assert [
            (action.coverage_delta, action.sample_delta, action.freshness_delta) for action in RANGE_ACTIONS
        ] == [
            (action.coverage_delta, action.sample_delta, action.freshness_delta) for action in STANDARD_ACTIONS
        ]


class TestOrchestrationEnvironmentRange:
    """Tests for range-based features in OrchestrationEnvironment."""

    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        monitor = ArchitectureMonitor(SamplingConfig())
        return OrchestrationEnvironment(monitor=monitor)

    def test_set_request(self, environment):
        """Test setting an analytics request."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.8),
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
        )

        config = environment.set_request(request)

        assert environment.current_request == request
        assert environment.effective_config is not None
        assert config.target_coverage == pytest.approx(0.6)
        assert config.target_sample == pytest.approx(0.45)

    def test_num_range_actions(self, environment):
        """Test number of available range actions."""
        assert environment.num_range_actions == len(RANGE_ACTIONS)

    def test_step_range_requires_request(self, environment):
        """Test that step_range requires a request to be set."""
        with pytest.raises(ValueError, match="No request set"):
            environment.step_range(0)

    def test_step_range_noop(self, environment):
        """Test no-op action doesn't change config."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.8),
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
        )
        environment.set_request(request)

        initial_coverage = environment.effective_config.target_coverage

        # Action 0 is no-op
        environment.step_range(0)

        assert environment.effective_config.target_coverage == initial_coverage

    def test_step_range_increases_coverage(self, environment):
        """Test coverage increase action."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.8),
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
        )
        environment.set_request(request)

        initial_coverage = environment.effective_config.target_coverage

        # Action 1 is INC_COVERAGE
        environment.step_range(1)

        assert environment.effective_config.target_coverage > initial_coverage

    def test_step_range_clamped_to_max(self, environment):
        """Test that actions are clamped to range maximum."""
        request = AnalyticsRequest(
            coverage_range=(0.7, 0.8),  # Narrow range at high end
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
        )
        environment.set_request(request)

        # Try to increase coverage multiple times
        for _ in range(20):
            environment.step_range(1)  # INC_COVERAGE

        assert environment.effective_config.target_coverage <= 0.8

    def test_step_range_clamped_to_min(self, environment):
        """Test that actions are clamped to range minimum."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.5),  # Narrow range at low end
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
        )
        environment.set_request(request)

        # Try to decrease coverage multiple times
        for _ in range(20):
            environment.step_range(2)  # DEC_COVERAGE

        assert environment.effective_config.target_coverage >= 0.4

    def test_step_range_returns_step_result(self, environment):
        """Test that step_range returns a valid StepResult."""
        request = AnalyticsRequest()
        environment.set_request(request)

        result = environment.step_range(0)

        assert result.encoded_state is not None
        assert result.reward is not None
        assert result.observation is not None
        assert result.effective_config is not None

    def test_step_range_preserves_request_context(self, environment):
        """Updated effective configs must keep non-range request metadata."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.8),
            sample_range=(0.3, 0.6),
            freshness_range=(30.0, 90.0),
            tenant_id="tenant-a",
            priority="critical",
            algorithm="ppo",
        )
        environment.set_request(request)

        environment.step_range(1)

        assert environment.effective_config.algorithm == "ppo"
        assert environment.effective_config.tenant_id == "tenant-a"
        assert environment.effective_config.priority == "critical"

    def test_update_node_metrics(self, environment):
        """Test updating node metrics."""
        metrics = {
            "node-1": {"cpu": 50.0, "memory": 60.0},
            "node-2": {"cpu": 40.0, "memory": 55.0},
        }

        environment.update_node_metrics(metrics)

        assert environment._node_metrics == metrics


class TestRewardBreakdownCoverage:
    """Tests for coverage penalty in reward calculation."""

    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        monitor = ArchitectureMonitor(SamplingConfig())
        return OrchestrationEnvironment(monitor=monitor)

    def test_no_penalty_within_range(self, environment):
        """Test no coverage penalty when within range."""
        request = AnalyticsRequest(coverage_range=(0.4, 0.8))
        environment.set_request(request)

        # Midpoint (0.6) is within range
        cov_p, smp_p, frs_p = environment._range_penalties()
        assert cov_p == 0.0
        assert smp_p == 0.0
        assert frs_p == 0.0

    def test_penalty_below_range(self, environment):
        """Test coverage penalty when below range."""
        request = AnalyticsRequest(coverage_range=(0.6, 0.8))
        environment.set_request(request)

        # Manually set effective config below range
        environment._effective_config = EffectiveConfiguration(
            request_id=request.request_id,
            service_type="heatmap",
            target_coverage=0.4,  # Below minimum of 0.6
            target_sample=0.5,
            target_freshness=60.0,
        )

        cov_p, smp_p, frs_p = environment._range_penalties()
        assert cov_p > 0.0
