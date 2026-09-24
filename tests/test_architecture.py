import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.settings import ResourceThresholds  # noqa: E402
from argos.domain.architecture import ArchitectureState, ComputeMetrics, DataMetrics, NetworkMetrics  # noqa: E402


class TestComputeMetrics(unittest.TestCase):
    """Tests for ComputeMetrics."""

    def test_as_dict_excludes_none(self):
        metrics = ComputeMetrics(cpu_utilization=50.0, memory_utilization=60.0)
        result = metrics.as_dict()
        self.assertEqual(result["cpu_utilization"], 50.0)
        self.assertEqual(result["memory_utilization"], 60.0)
        self.assertNotIn("energy_consumption", result)
        self.assertNotIn("computing_cost", result)

    def test_as_dict_includes_optional_when_set(self):
        metrics = ComputeMetrics(
            cpu_utilization=50.0,
            memory_utilization=60.0,
            energy_consumption=100.0,
            computing_cost=0.05,
        )
        result = metrics.as_dict()
        self.assertEqual(result["energy_consumption"], 100.0)
        self.assertEqual(result["computing_cost"], 0.05)


class TestDataMetrics(unittest.TestCase):
    """Tests for DataMetrics."""

    def test_as_dict_excludes_none(self):
        metrics = DataMetrics(volume_mb=1024.0)
        result = metrics.as_dict()
        self.assertEqual(result["volume_mb"], 1024.0)
        self.assertNotIn("refresh_rate_s", result)

    def test_as_dict_all_fields(self):
        metrics = DataMetrics(
            volume_mb=1024.0,
            refresh_rate_s=60.0,
            processing_time_ms=5.0,
            storage_cost=0.01,
        )
        result = metrics.as_dict()
        self.assertEqual(len(result), 4)


class TestNetworkMetrics(unittest.TestCase):
    """Tests for NetworkMetrics."""

    def test_as_dict_excludes_none(self):
        metrics = NetworkMetrics(latency_ms=10.0, bandwidth_mbps=100.0)
        result = metrics.as_dict()
        self.assertEqual(result["latency_ms"], 10.0)
        self.assertEqual(result["bandwidth_mbps"], 100.0)
        self.assertNotIn("transfer_cost", result)


class TestArchitectureState(unittest.TestCase):
    """Tests for ArchitectureState."""

    def setUp(self):
        self.state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=50.0, memory_utilization=60.0),
            data=DataMetrics(volume_mb=1024.0, refresh_rate_s=30.0),
            network=NetworkMetrics(latency_ms=15.0, bandwidth_mbps=85.0),
        )

    def test_flatten_creates_prefixed_keys(self):
        result = self.state.flatten()
        self.assertIn("compute.cpu_utilization", result)
        self.assertIn("compute.memory_utilization", result)
        self.assertIn("data.volume_mb", result)
        self.assertIn("network.latency_ms", result)
        self.assertEqual(result["compute.cpu_utilization"], 50.0)

    def test_resource_pressure_over_thresholds(self):
        state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=80.0, memory_utilization=70.0),
            data=DataMetrics(volume_mb=1.0),
            network=NetworkMetrics(latency_ms=None, bandwidth_mbps=200.0),
        )
        thresholds = ResourceThresholds(cpu_percent=75.0, memory_percent=80.0, bandwidth_mbps=150.0)
        pressure = state.resource_pressure(thresholds)

        self.assertAlmostEqual(pressure["cpu"], (80.0 - 75.0) / 75.0, places=6)
        self.assertAlmostEqual(pressure["memory"], 0.0, places=6)
        self.assertAlmostEqual(pressure["bandwidth"], (200.0 - 150.0) / 150.0, places=6)

    def test_resource_pressure_under_thresholds(self):
        state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=30.0, memory_utilization=40.0),
            data=DataMetrics(volume_mb=1.0),
            network=NetworkMetrics(latency_ms=10.0, bandwidth_mbps=50.0),
        )
        thresholds = ResourceThresholds(cpu_percent=75.0, memory_percent=80.0, bandwidth_mbps=150.0)
        pressure = state.resource_pressure(thresholds)

        self.assertEqual(pressure["cpu"], 0.0)
        self.assertEqual(pressure["memory"], 0.0)
        self.assertEqual(pressure["bandwidth"], 0.0)

    def test_resource_pressure_none_bandwidth(self):
        state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=50.0, memory_utilization=50.0),
            data=DataMetrics(volume_mb=1.0),
            network=NetworkMetrics(latency_ms=None, bandwidth_mbps=None),
        )
        thresholds = ResourceThresholds(cpu_percent=75.0, memory_percent=80.0, bandwidth_mbps=150.0)
        pressure = state.resource_pressure(thresholds)

        self.assertEqual(pressure["bandwidth"], 0.0)


if __name__ == "__main__":
    unittest.main()
