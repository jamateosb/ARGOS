"""Tests for the orchestration environment."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.settings import DEFAULT_BOUNDS, DEFAULT_STEPS  # noqa: E402
from argos.domain.architecture import ArchitectureState, ComputeMetrics, DataMetrics, NetworkMetrics  # noqa: E402
from argos.domain.monitor import ArchitectureSample  # noqa: E402
from argos.domain.requirements import RequirementAction  # noqa: E402
from argos.orchestrator.environment import OrchestrationEnvironment, RewardBreakdown  # noqa: E402


class TestRewardBreakdown(unittest.TestCase):
    """Tests for RewardBreakdown dataclass."""

    def test_total_calculation(self):
        """Total should be quality minus penalties."""
        breakdown = RewardBreakdown(
            requirement_quality=1.0,
            resource_penalty=0.3,
            cost_penalty=0.2,
        )
        self.assertAlmostEqual(breakdown.total, 0.5, places=5)

    def test_total_clamped_to_range(self):
        """Total must be clamped to [-1, +1]."""
        low = RewardBreakdown(
            requirement_quality=0.0,
            resource_penalty=1.0,
            cost_penalty=1.0,
            range_penalty_coverage=0.17,
            range_penalty_sample=0.17,
            range_penalty_freshness=0.16,
        )
        self.assertEqual(low.total, -1.0)
        high = RewardBreakdown(requirement_quality=1.0, resource_penalty=0.0, cost_penalty=0.0)
        self.assertAlmostEqual(high.total, 1.0, places=5)


class TestOrchestrationEnvironment(unittest.TestCase):
    """Tests for OrchestrationEnvironment."""

    def setUp(self):
        """Set up mock monitor."""
        self.mock_monitor = MagicMock()
        self.sample = ArchitectureSample(
            timestamp=datetime.now(timezone.utc),
            state=ArchitectureState(
                compute=ComputeMetrics(cpu_utilization=50.0, memory_utilization=60.0),
                data=DataMetrics(volume_mb=1000.0),
                network=NetworkMetrics(latency_ms=10.0, bandwidth_mbps=80.0),
            ),
            bandwidth_window_s=5.0,
        )
        self.mock_monitor.sample.return_value = self.sample

    def test_observe_returns_step_result(self):
        """observe() should return a StepResult with encoded state and reward."""
        env = OrchestrationEnvironment(self.mock_monitor)
        result = env.observe()

        self.assertIsNotNone(result.encoded_state)
        self.assertIsInstance(result.encoded_state, tuple)
        self.assertIsInstance(result.reward, float)
        self.assertIsInstance(result.reward_details, RewardBreakdown)
        self.assertEqual(result.observation, self.sample)

    def test_step_applies_action(self):
        """step() should apply the action to requirements."""
        env = OrchestrationEnvironment(self.mock_monitor)
        initial_coverage = env.requirements.coverage

        result = env.step(RequirementAction.INCREASE_COVERAGE)

        expected_coverage = min(initial_coverage + DEFAULT_STEPS.coverage, DEFAULT_BOUNDS.coverage[1])
        self.assertAlmostEqual(env.requirements.coverage, expected_coverage, places=5)
        self.assertIsNotNone(result.encoded_state)

    def test_step_decrease_coverage(self):
        """step() with DECREASE_COVERAGE should reduce coverage."""
        env = OrchestrationEnvironment(self.mock_monitor)
        initial_coverage = env.requirements.coverage

        env.step(RequirementAction.DECREASE_COVERAGE)

        expected_coverage = max(initial_coverage - DEFAULT_STEPS.coverage, DEFAULT_BOUNDS.coverage[0])
        self.assertAlmostEqual(env.requirements.coverage, expected_coverage, places=5)

    def test_encode_produces_tuple(self):
        """_encode should produce a tuple of integers."""
        env = OrchestrationEnvironment(self.mock_monitor)
        encoded = env._encode(self.sample.state)

        self.assertIsInstance(encoded, tuple)
        self.assertTrue(all(isinstance(x, int) for x in encoded))
        # 3 architecture + 3 requirement + 2 workload + 5 orchestration features
        self.assertEqual(len(encoded), 13)

    def test_requirement_pressure_returns_dict(self):
        """_requirement_pressure should return pressure values for each resource."""
        env = OrchestrationEnvironment(self.mock_monitor)
        pressure = env._requirement_pressure()

        self.assertIn("cpu", pressure)
        self.assertIn("memory", pressure)
        self.assertIn("bandwidth", pressure)
        self.assertTrue(all(isinstance(v, float) for v in pressure.values()))

    def test_service_duty_and_input_level_are_encoded_as_workload_state(self):
        from argos.settings import RL_STATE_KEYS
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(request_id="duty", input_multiplier=8)
        env.set_request(request)
        env.update_node_metrics(
            {
                "node-1": {
                    "analytics": [
                        {
                            "request_id": "duty",
                            "last_result_at": "2026-01-01T00:00:00+00:00",
                            "last_processing_time_ms": 5000.0,
                            "target_freshness": 10.0,
                        }
                    ]
                }
            }
        )

        encoded = env._encode(self.sample.state)

        self.assertEqual(env._service_duty_percent, 50.0)
        self.assertEqual(encoded[RL_STATE_KEYS.index("service_duty")], 2)
        self.assertEqual(encoded[RL_STATE_KEYS.index("input_multiplier")], 3)

    def test_available_actions_exclude_clamped_boundary_moves(self):
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(
            request_id="masked",
            coverage_range=(0.5, 0.6),
            sample_range=(0.5, 0.6),
            freshness_range=(10.0, 20.0),
        )
        env.set_request(request)
        env.step_range(1)
        env.step_range(3)
        env.step_range(6)

        available = env.available_action_indices()

        self.assertIn(0, available)
        self.assertNotIn(1, available)
        self.assertNotIn(3, available)
        self.assertNotIn(6, available)
        self.assertIn(2, available)
        self.assertIn(4, available)
        self.assertIn(5, available)

    def test_contract_relative_state_distinguishes_narrow_freshness_values(self):
        from argos.settings import RL_STATE_KEYS
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(
            request_id="relative",
            coverage_range=(0.75, 1.0),
            sample_range=(0.7, 1.0),
            freshness_range=(5.0, 30.0),
        )
        env.set_request(request)
        midpoint = env.observe().encoded_state[RL_STATE_KEYS.index("freshness")]
        env.step_range(6)
        fresher = env.observe().encoded_state[RL_STATE_KEYS.index("freshness")]

        self.assertNotEqual(midpoint, fresher)

    def test_logically_equal_configs_have_path_independent_encoded_state(self):
        from argos.domain.requests import AnalyticsRequest

        request = AnalyticsRequest(
            request_id="stable-path",
            coverage_range=(0.2, 0.5),
            sample_range=(0.2, 0.4),
            freshness_range=(30.0, 90.0),
        )
        first = OrchestrationEnvironment(self.mock_monitor)
        first.set_request(request)
        for action in (4, 4, 3):
            first.step_range(action)

        second = OrchestrationEnvironment(self.mock_monitor)
        second.set_request(request)
        for action in (3, 3, 3, 4, 4, 4):
            second.step_range(action)

        self.assertEqual(first.effective_config.target_sample, 0.25)
        self.assertEqual(second.effective_config.target_sample, 0.25)
        self.assertEqual(
            first.observe().encoded_state,
            second.observe().encoded_state,
        )

    def test_service_duty_preserves_over_capacity_bucket(self):
        from argos.settings import RL_STATE_KEYS
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(request_id="over-duty")
        env.set_request(request)
        env.update_node_metrics(
            {
                "node-1": {
                    "analytics": [
                        {
                            "request_id": request.request_id,
                            "last_result_at": "2026-01-01T00:00:00+00:00",
                            "last_processing_time_ms": 15000.0,
                            "target_freshness": 10.0,
                        }
                    ]
                }
            }
        )

        encoded = env._encode(self.sample.state)
        self.assertEqual(env._service_duty_percent, 150.0)
        self.assertEqual(encoded[RL_STATE_KEYS.index("service_duty")], 5)

    def test_reward_computation(self):
        """_reward should compute reward breakdown."""
        env = OrchestrationEnvironment(self.mock_monitor)
        reward = env._reward(self.sample.state)

        self.assertIsInstance(reward, RewardBreakdown)
        self.assertIsInstance(reward.requirement_quality, float)
        self.assertIsInstance(reward.resource_penalty, float)
        self.assertIsInstance(reward.cost_penalty, float)

    def test_reward_linear_penalty_values(self):
        """Verify linear penalty formula uses observed resource pressure."""
        from argos.settings import (
            DEFAULT_REWARD_WEIGHTS,
        )

        env = OrchestrationEnvironment(self.mock_monitor)
        # Set known requirement values for deterministic computation
        env.requirements.coverage = 0.6
        env.requirements.sample = 0.5
        env.requirements.freshness_seconds = 60.0

        reward = env._reward(self.sample.state)

        # Manually compute expected penalties
        fp = 1.0 / 60.0  # freshness_pressure
        cpu_p = 50.0 / 75.0
        mem_p = 60.0 / 80.0
        bw_p = 80.0 / 150.0
        pressure_avg = (cpu_p + mem_p + bw_p) / 3.0

        w_r = DEFAULT_REWARD_WEIGHTS.resource_pressure
        w_c = DEFAULT_REWARD_WEIGHTS.cost
        w_total = w_r + w_c
        expected_rp = (w_r / w_total) * pressure_avg
        expected_cp_raw = 0.6 * 0.5 * (1.0 + fp)
        bounds = DEFAULT_BOUNDS
        max_cost = bounds.coverage[1] * bounds.sample[1] * (1.0 + 1.0 / max(bounds.freshness_seconds[0], 1.0))
        cost_norm = min(expected_cp_raw / max(max_cost, 1e-6), 1.0)
        expected_cp = (w_c / w_total) * cost_norm

        self.assertAlmostEqual(reward.resource_penalty, expected_rp, places=5)
        self.assertAlmostEqual(reward.cost_penalty, expected_cp, places=5)

        # Confirm penalties are LINEAR (no ** 2)
        # If quadratic, resource_penalty would equal (w_r/w_total)*pressure_avg**2
        quadratic_rp = (w_r / w_total) * pressure_avg**2
        self.assertNotAlmostEqual(reward.resource_penalty, quadratic_rp, places=3)

    def test_reward_uses_request_ranges_for_quality_and_cost(self):
        """Active requests should score quality/cost against their own ranges."""
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(
            request_id="test",
            service_type="bus",
            coverage_range=(0.5, 0.9),
            sample_range=(0.3, 0.8),
            freshness_range=(30.0, 120.0),
        )
        env.set_request(request)

        reward = env._reward(self.sample.state)

        # Midpoint of every request range should score 0.5 per dimension.
        self.assertAlmostEqual(reward.requirement_quality, 0.5, places=5)
        old_global_quality = env.requirements.requirement_quality(DEFAULT_BOUNDS)
        self.assertGreater(reward.requirement_quality, old_global_quality)

        raw_cost = env.requirements.estimated_cost()
        max_request_cost = 0.9 * 0.8 * (1.0 + 1.0 / 30.0)
        expected_cost_penalty = (0.5 / (1.2 + 0.5)) * (raw_cost / max_request_cost)
        self.assertAlmostEqual(reward.cost_penalty, expected_cost_penalty, places=5)

    def test_reward_falls_back_to_heuristic_pressure_without_observed_metrics(self):
        """Requirement pressure remains available when telemetry is empty."""
        from argos.settings import DEFAULT_PRESSURE_WEIGHTS, DEFAULT_REWARD_WEIGHTS

        env = OrchestrationEnvironment(self.mock_monitor)
        env.requirements.coverage = 0.6
        env.requirements.sample = 0.5
        env.requirements.freshness_seconds = 60.0

        empty_state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=0.0, memory_utilization=0.0),
            data=DataMetrics(),
            network=NetworkMetrics(latency_ms=None, bandwidth_mbps=0.0),
        )
        reward = env._reward(empty_state)

        pw = DEFAULT_PRESSURE_WEIGHTS
        fp = 1.0 / 60.0
        pressure_avg = (
            pw.cpu_coverage * 0.6
            + pw.cpu_sample * 0.5
            + pw.cpu_freshness * fp
            + pw.memory_coverage * 0.6
            + pw.memory_sample * 0.5
            + pw.memory_freshness * fp
            + pw.bandwidth_coverage * 0.6
            + pw.bandwidth_sample * 0.5
            + pw.bandwidth_freshness * fp
        ) / 3.0
        expected = (
            DEFAULT_REWARD_WEIGHTS.resource_pressure
            / (DEFAULT_REWARD_WEIGHTS.resource_pressure + DEFAULT_REWARD_WEIGHTS.cost)
        ) * pressure_avg

        self.assertAlmostEqual(reward.resource_penalty, expected, places=5)

    def test_range_penalties_within_range_is_zero(self):
        """All range penalties must be 0 when dimensions are within user's ranges."""
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(
            request_id="test",
            service_type="bus",
            coverage_range=(0.5, 0.9),
            sample_range=(0.3, 0.8),
            freshness_range=(30, 120),
        )
        env.set_request(request)
        # Midpoint should be inside range for all dimensions
        cov_p, smp_p, frs_p = env._range_penalties()
        self.assertEqual(cov_p, 0.0)
        self.assertEqual(smp_p, 0.0)
        self.assertEqual(frs_p, 0.0)

    def test_range_penalties_outside_range_has_gradient(self):
        """Per-dimension range penalties grow smoothly and are independent."""
        from argos.domain.requests import AnalyticsRequest

        env = OrchestrationEnvironment(self.mock_monitor)
        request = AnalyticsRequest(
            request_id="test",
            service_type="bus",
            coverage_range=(0.5, 0.9),
            sample_range=(0.3, 0.8),
            freshness_range=(30, 120),
        )
        env.set_request(request)

        # Coverage slightly below range — only coverage penalty fires
        env._effective_config.target_coverage = 0.48
        cov_p, smp_p, frs_p = env._range_penalties()
        self.assertGreater(cov_p, 0.0)
        self.assertEqual(smp_p, 0.0)
        self.assertEqual(frs_p, 0.0)
        p_small = cov_p

        # Coverage further below range → larger coverage penalty
        env._effective_config.target_coverage = 0.30
        cov_p2, _, _ = env._range_penalties()
        self.assertGreater(cov_p2, p_small)

        # Reset coverage inside range, move sample outside — only sample fires
        env._effective_config.target_coverage = 0.6
        env._effective_config.target_sample = 0.1  # Below 0.3
        cov_p3, smp_p3, frs_p3 = env._range_penalties()
        self.assertEqual(cov_p3, 0.0)
        self.assertGreater(smp_p3, 0.0)
        self.assertEqual(frs_p3, 0.0)

        # Reset sample inside range, move freshness outside — only freshness fires
        env._effective_config.target_sample = 0.5
        env._effective_config.target_freshness = 10.0  # Below 30
        cov_p4, smp_p4, frs_p4 = env._range_penalties()
        self.assertEqual(cov_p4, 0.0)
        self.assertEqual(smp_p4, 0.0)
        self.assertGreater(frs_p4, 0.0)

        # All inside → all zero
        env._effective_config.target_coverage = 0.6
        env._effective_config.target_sample = 0.5
        env._effective_config.target_freshness = 60.0
        cov_p5, smp_p5, frs_p5 = env._range_penalties()
        self.assertEqual(cov_p5, 0.0)
        self.assertEqual(smp_p5, 0.0)
        self.assertEqual(frs_p5, 0.0)

    def test_quality_weight_is_applied(self):
        """requirement_quality must be scaled by reward_weights.quality."""
        from argos.settings import RewardWeights

        # Use quality weight != 1.0 to verify it takes effect
        custom_weights = RewardWeights(requirement_quality=0.5, resource_pressure=1.2, cost=0.5)
        env = OrchestrationEnvironment(self.mock_monitor, reward_weights=custom_weights)
        reward_half = env._reward(self.sample.state)

        env_full = OrchestrationEnvironment(self.mock_monitor)
        reward_full = env_full._reward(self.sample.state)

        self.assertAlmostEqual(
            reward_half.requirement_quality,
            reward_full.requirement_quality * 0.5,
            places=5,
        )


class TestResourceOverloadPenalty(unittest.TestCase):
    """Tests for the resource overload penalty based on actual node metrics."""

    def setUp(self):
        self.mock_monitor = MagicMock()
        self.sample = ArchitectureSample(
            timestamp=datetime.now(timezone.utc),
            state=ArchitectureState(
                compute=ComputeMetrics(cpu_utilization=50.0, memory_utilization=60.0),
                data=DataMetrics(volume_mb=1000.0),
                network=NetworkMetrics(latency_ms=10.0, bandwidth_mbps=80.0),
            ),
            bandwidth_window_s=5.0,
        )
        self.mock_monitor.sample.return_value = self.sample

    def test_no_metrics_returns_zero(self):
        """Penalty should be 0 when no node metrics are available."""
        env = OrchestrationEnvironment(self.mock_monitor)
        self.assertEqual(env._resource_overload_penalty(), 0.0)

    def test_below_threshold_returns_zero(self):
        """Penalty should be 0 when CPU and memory are below hard limits."""
        env = OrchestrationEnvironment(self.mock_monitor)
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 50.0, "memory_utilization": 60.0}}},
                "node2": {"state": {"compute": {"cpu_utilization": 70.0, "memory_utilization": 80.0}}},
            }
        )
        # Defaults: cpu_hard_limit=80, memory_hard_limit=85
        # avg_cpu=60, avg_mem=70 → both below limits
        self.assertEqual(env._resource_overload_penalty(), 0.0)

    def test_cpu_above_limit_returns_penalty(self):
        """Penalty should be > 0 when average CPU exceeds hard limit."""
        env = OrchestrationEnvironment(self.mock_monitor)
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 90.0, "memory_utilization": 50.0}}},
            }
        )
        penalty = env._resource_overload_penalty()
        self.assertGreater(penalty, 0.0)

    def test_memory_above_limit_returns_penalty(self):
        """Penalty should be > 0 when average memory exceeds hard limit."""
        env = OrchestrationEnvironment(self.mock_monitor)
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 50.0, "memory_utilization": 95.0}}},
            }
        )
        penalty = env._resource_overload_penalty()
        self.assertGreater(penalty, 0.0)

    def test_penalty_scales_with_excess(self):
        """Higher overload should produce a larger penalty."""
        env = OrchestrationEnvironment(self.mock_monitor)

        # Moderate overload: CPU at 85 (5 above limit)
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 85.0, "memory_utilization": 50.0}}},
            }
        )
        p_moderate = env._resource_overload_penalty()

        # Severe overload: CPU at 99
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 99.0, "memory_utilization": 50.0}}},
            }
        )
        p_severe = env._resource_overload_penalty()

        self.assertGreater(p_severe, p_moderate)

    def test_overload_penalty_included_in_reward_total(self):
        """RewardBreakdown.total must subtract resource_overload_penalty."""
        bd = RewardBreakdown(
            requirement_quality=1.0,
            resource_penalty=0.0,
            cost_penalty=0.0,
            resource_overload_penalty=0.3,
        )
        self.assertAlmostEqual(bd.total, 0.7, places=5)

    def test_both_cpu_and_memory_overloaded(self):
        """Penalty should be higher when both CPU and memory are above limits."""
        env = OrchestrationEnvironment(self.mock_monitor)

        # Only CPU overloaded
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 90.0, "memory_utilization": 50.0}}},
            }
        )
        p_cpu_only = env._resource_overload_penalty()

        # Both overloaded
        env.update_node_metrics(
            {
                "node1": {"state": {"compute": {"cpu_utilization": 90.0, "memory_utilization": 95.0}}},
            }
        )
        p_both = env._resource_overload_penalty()

        self.assertGreater(p_both, p_cpu_only)


if __name__ == "__main__":
    unittest.main()
