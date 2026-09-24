"""Tests for the architecture monitor."""

import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from argos.settings import LatencyProbeConfig, SamplingConfig  # noqa: E402
from argos.domain.monitor import ArchitectureMonitor, ArchitectureSample, _default_mountpoint  # noqa: E402


class TestArchitectureMonitor(unittest.TestCase):
    """Tests for ArchitectureMonitor."""

    def test_sample_returns_architecture_sample(self):
        """sample() should return an ArchitectureSample."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        monitor = ArchitectureMonitor(sampling)

        result = monitor.sample()

        self.assertIsInstance(result, ArchitectureSample)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertIsNotNone(result.state)
        self.assertIsNotNone(result.state.compute)
        self.assertIsNotNone(result.state.network)

    def test_sample_captures_cpu_and_memory(self):
        """sample() should capture CPU and memory utilization."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        monitor = ArchitectureMonitor(sampling)

        result = monitor.sample()

        self.assertGreaterEqual(result.state.compute.cpu_utilization, 0.0)
        self.assertLessEqual(result.state.compute.cpu_utilization, 100.0)
        self.assertGreaterEqual(result.state.compute.memory_utilization, 0.0)
        self.assertLessEqual(result.state.compute.memory_utilization, 100.0)

    def test_sample_with_latency_probe_disabled(self):
        """sample() with latency probe disabled should have None latency."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        latency_probe = LatencyProbeConfig(enabled=False)
        monitor = ArchitectureMonitor(sampling, latency_probe=latency_probe)

        result = monitor.sample()

        self.assertIsNone(result.state.network.latency_ms)

    def test_bandwidth_is_non_negative(self):
        """Bandwidth should always be non-negative."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        monitor = ArchitectureMonitor(sampling)

        result = monitor.sample()

        self.assertGreaterEqual(result.state.network.bandwidth_mbps, 0.0)

    def test_data_metrics_populated(self):
        """Data metrics should be populated."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        monitor = ArchitectureMonitor(sampling)

        result = monitor.sample()

        self.assertIsNotNone(result.state.data.volume_mb)
        self.assertGreater(result.state.data.volume_mb, 0.0)

    def test_probe_latency_with_invalid_host(self):
        """_probe_latency should return None for invalid host."""
        sampling = SamplingConfig(interval_seconds=1.0, network_window_seconds=5.0)
        monitor = ArchitectureMonitor(sampling)

        result = monitor._probe_latency("invalid.host.that.does.not.exist.local", 80, 0.1)

        self.assertIsNone(result)

    def test_default_mountpoint(self):
        """_default_mountpoint should return a string."""
        result = _default_mountpoint()
        self.assertIsInstance(result, str)


class TestBandwidthCalculation(unittest.TestCase):
    """Tests for bandwidth calculation with sliding window."""

    def test_multiple_samples_accumulate_bandwidth(self):
        """Multiple samples should use sliding window for bandwidth."""
        sampling = SamplingConfig(interval_seconds=0.1, network_window_seconds=1.0)
        monitor = ArchitectureMonitor(sampling)

        # Take multiple samples
        results = [monitor.sample() for _ in range(3)]

        # All should have valid bandwidth
        for r in results:
            self.assertGreaterEqual(r.state.network.bandwidth_mbps, 0.0)


class TestBandwidthWindow(unittest.TestCase):
    """Tests for bandwidth window calculation."""

    def test_bandwidth_with_network_window(self):
        """Bandwidth should use windowed calculation when network_window_seconds > 0."""
        from argos.settings import SamplingConfig

        sampling_config = SamplingConfig(
            interval_seconds=0.1,
            network_window_seconds=1.0,
        )
        monitor = ArchitectureMonitor(sampling=sampling_config)

        # Take multiple samples to populate window
        for _ in range(5):
            sample = monitor.sample()
            time.sleep(0.05)

        # Bandwidth should be calculated using window
        self.assertGreaterEqual(sample.state.network.bandwidth_mbps, 0.0)
        self.assertEqual(sample.bandwidth_window_s, 1.0)

    def test_bandwidth_without_window(self):
        """Bandwidth should use delta calculation when network_window_seconds <= 0."""
        from argos.settings import SamplingConfig

        sampling_config = SamplingConfig(
            interval_seconds=0.1,
            network_window_seconds=0.0,
        )
        monitor = ArchitectureMonitor(sampling=sampling_config)

        # Take samples
        _ = monitor.sample()
        time.sleep(0.05)
        sample = monitor.sample()

        # Bandwidth should still be calculated
        self.assertGreaterEqual(sample.state.network.bandwidth_mbps, 0.0)


class TestProbeLatencySuccess(unittest.TestCase):
    """Tests for successful latency probe."""

    @patch("socket.create_connection")
    def test_probe_latency_success(self, mock_conn):
        """Successful connection should return latency in ms."""
        from argos.settings import LatencyProbeConfig, SamplingConfig

        # Mock socket context manager
        mock_socket = MagicMock()
        mock_conn.return_value.__enter__ = MagicMock(return_value=mock_socket)
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        latency_config = LatencyProbeConfig(
            enabled=True,
            host="localhost",
            port=8080,
            timeout_seconds=1.0,
        )
        sampling_config = SamplingConfig(interval_seconds=0.1)

        monitor = ArchitectureMonitor(
            latency_probe=latency_config,
            sampling=sampling_config,
        )

        sample = monitor.sample()

        # Latency should be a non-negative number
        self.assertIsNotNone(sample.state.network.latency_ms)
        self.assertGreaterEqual(sample.state.network.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
