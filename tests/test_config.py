"""Tests for configuration dataclasses."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.settings import (  # noqa: E402
    DEFAULT_AGENT_PARAMS,
    DEFAULT_BOUNDS,
    DEFAULT_COLLECTOR,
    DEFAULT_COLLECTOR_RUNTIME,
    DEFAULT_HTTP_SINK,
    DEFAULT_LATENCY_PROBE,
    DEFAULT_PRESSURE_WEIGHTS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_SAMPLING,
    DEFAULT_SCORING_WEIGHTS,
    DEFAULT_STEPS,
    DEFAULT_THRESHOLDS,
    STATE_BINS,
    ActionSteps,
    AgentHyperParams,
    CollectorConfig,
    CollectorRunArgs,
    HttpSinkConfig,
    LatencyProbeConfig,
    PressureWeights,
    RequirementBounds,
    ResourceThresholds,
    RewardWeights,
    SamplingConfig,
    ScoringWeights,
)


class TestRequirementBounds(unittest.TestCase):
    """Tests for RequirementBounds dataclass."""

    def test_default_values(self):
        bounds = RequirementBounds()
        self.assertEqual(bounds.coverage, (0.2, 1.0))
        self.assertEqual(bounds.sample, (0.1, 1.0))
        self.assertEqual(bounds.freshness_seconds, (5.0, 600.0))

    def test_frozen(self):
        """Frozen dataclass should be immutable."""
        bounds = RequirementBounds()
        with self.assertRaises(AttributeError):
            bounds.coverage = (0.0, 0.5)

    def test_hashable(self):
        """Frozen dataclass should be hashable."""
        bounds = RequirementBounds()
        self.assertIsInstance(hash(bounds), int)


class TestActionSteps(unittest.TestCase):
    """Tests for ActionSteps dataclass."""

    def test_default_values(self):
        steps = ActionSteps()
        self.assertEqual(steps.coverage, 0.05)
        self.assertEqual(steps.sample, 0.05)
        self.assertEqual(steps.freshness_seconds, 10.0)


class TestResourceThresholds(unittest.TestCase):
    """Tests for ResourceThresholds dataclass."""

    def test_default_values(self):
        thresholds = ResourceThresholds()
        self.assertEqual(thresholds.cpu_percent, 75.0)
        self.assertEqual(thresholds.memory_percent, 80.0)
        self.assertEqual(thresholds.bandwidth_mbps, 150.0)


class TestRewardWeights(unittest.TestCase):
    """Tests for RewardWeights dataclass."""

    def test_default_values(self):
        weights = RewardWeights()
        self.assertEqual(weights.requirement_quality, 1.0)
        self.assertEqual(weights.resource_pressure, 1.2)
        self.assertEqual(weights.cost, 0.5)


class TestAgentHyperParams(unittest.TestCase):
    """Tests for AgentHyperParams dataclass."""

    def test_default_values(self):
        params = AgentHyperParams()
        self.assertEqual(params.alpha, 0.15)
        self.assertEqual(params.gamma, 0.9)
        self.assertEqual(params.epsilon, 0.1)
        self.assertEqual(params.seed, 42)

    def test_custom_values(self):
        params = AgentHyperParams(alpha=0.2, gamma=0.95, epsilon=0.05, seed=123)
        self.assertEqual(params.alpha, 0.2)
        self.assertEqual(params.seed, 123)


class TestScoringWeights(unittest.TestCase):
    """Tests for ScoringWeights dataclass."""

    def test_default_values(self):
        weights = ScoringWeights()
        self.assertEqual(weights.cpu_available, 0.5)
        self.assertEqual(weights.memory_available, 0.4)
        self.assertEqual(weights.bandwidth_headroom, 0.1)

    def test_weights_sum_to_one(self):
        weights = ScoringWeights()
        total = weights.cpu_available + weights.memory_available + weights.bandwidth_headroom
        self.assertAlmostEqual(total, 1.0, places=5)


class TestPressureWeights(unittest.TestCase):
    """Tests for PressureWeights dataclass."""

    def test_cpu_weights_sum_to_one(self):
        weights = PressureWeights()
        total = weights.cpu_coverage + weights.cpu_sample + weights.cpu_freshness
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_memory_weights_sum_to_one(self):
        weights = PressureWeights()
        total = weights.memory_coverage + weights.memory_sample + weights.memory_freshness
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_bandwidth_weights_sum_to_one(self):
        weights = PressureWeights()
        total = weights.bandwidth_coverage + weights.bandwidth_sample + weights.bandwidth_freshness
        self.assertAlmostEqual(total, 1.0, places=5)


class TestSamplingConfig(unittest.TestCase):
    """Tests for SamplingConfig dataclass."""

    def test_default_values(self):
        config = SamplingConfig()
        self.assertEqual(config.interval_seconds, 1.0)
        self.assertEqual(config.network_window_seconds, 5.0)


class TestLatencyProbeConfig(unittest.TestCase):
    """Tests for LatencyProbeConfig dataclass."""

    def test_default_values(self):
        config = LatencyProbeConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 80)
        self.assertEqual(config.timeout_seconds, 0.5)


class TestCollectorConfig(unittest.TestCase):
    """Tests for CollectorConfig dataclass."""

    def test_default_values(self):
        config = CollectorConfig()
        self.assertTrue(config.persist)
        self.assertEqual(config.rotation_seconds, 1200)
        self.assertEqual(config.max_entries_per_file, 300)


class TestHttpSinkConfig(unittest.TestCase):
    """Tests for HttpSinkConfig dataclass."""

    def test_default_values(self):
        config = HttpSinkConfig()
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.retry_wait_seconds, 1.0)


class TestCollectorRunArgs(unittest.TestCase):
    """Tests for CollectorRunArgs dataclass."""

    def _make_args(self, **overrides):
        """Create CollectorRunArgs with reasonable defaults."""
        defaults = {
            "machine_id": "test-node",
            "interval_seconds": 1.0,
            "output_format": "jsonl",
            "persist": True,
            "sink": "local",
            "http_endpoint": None,
            "http_timeout": 5.0,
            "http_verify_ssl": True,
            "data_root": "./data",
            "latency_probe_enabled": True,
            "latency_host": "127.0.0.1",
            "latency_port": 80,
            "latency_timeout": 0.5,
        }
        defaults.update(overrides)
        return CollectorRunArgs(**defaults)

    def test_default_values(self):
        args = self._make_args()
        self.assertEqual(args.sink, "local")
        self.assertIsNone(args.http_endpoint)
        self.assertTrue(args.http_verify_ssl)
        self.assertTrue(args.latency_probe_enabled)

    def test_custom_values(self):
        args = self._make_args(
            machine_id="custom-node",
            interval_seconds=2.0,
            sink="http",
            http_endpoint="http://example.com/ingest",
        )
        self.assertEqual(args.machine_id, "custom-node")
        self.assertEqual(args.interval_seconds, 2.0)
        self.assertEqual(args.sink, "http")


class TestStateBins(unittest.TestCase):
    """Tests for STATE_BINS constant."""

    def test_contains_required_keys(self):
        required_keys = ["cpu", "memory", "bandwidth", "coverage", "sample", "freshness"]
        for key in required_keys:
            self.assertIn(key, STATE_BINS)

    def test_bins_are_sorted(self):
        for _key, bins in STATE_BINS.items():
            self.assertEqual(list(bins), sorted(bins))


class TestDefaults(unittest.TestCase):
    """Tests for default configuration instances."""

    def test_defaults_exist(self):
        self.assertIsNotNone(DEFAULT_BOUNDS)
        self.assertIsNotNone(DEFAULT_STEPS)
        self.assertIsNotNone(DEFAULT_THRESHOLDS)
        self.assertIsNotNone(DEFAULT_REWARD_WEIGHTS)
        self.assertIsNotNone(DEFAULT_AGENT_PARAMS)
        self.assertIsNotNone(DEFAULT_SAMPLING)
        self.assertIsNotNone(DEFAULT_LATENCY_PROBE)
        self.assertIsNotNone(DEFAULT_COLLECTOR)
        self.assertIsNotNone(DEFAULT_SCORING_WEIGHTS)
        self.assertIsNotNone(DEFAULT_PRESSURE_WEIGHTS)
        self.assertIsNotNone(DEFAULT_HTTP_SINK)
        self.assertIsNotNone(DEFAULT_COLLECTOR_RUNTIME)


if __name__ == "__main__":
    unittest.main()
