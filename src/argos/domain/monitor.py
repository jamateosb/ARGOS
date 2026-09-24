# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Monitor node-level architecture metrics (compute, data, network)."""

from __future__ import annotations

import socket
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import psutil

from argos.settings import LatencyProbeConfig, SamplingConfig
from argos.domain.architecture import (
    ArchitectureState,
    ComputeMetrics,
    DataMetrics,
    NetworkMetrics,
    ServiceMetrics,
)


def _default_mountpoint() -> str:
    """Return the first disk mountpoint or root as fallback."""
    partitions = psutil.disk_partitions()
    return partitions[0].mountpoint if partitions else "/"


@dataclass
class ArchitectureSample:
    """Container for a timestamped architecture snapshot."""

    timestamp: datetime
    state: ArchitectureState
    bandwidth_window_s: float


class ArchitectureMonitor:
    """
    Lightweight monitor that captures CPU, memory, storage usage and network throughput.
    Designed to work on laptops, edge devices, and cloud VMs without extra dependencies.
    """

    def __init__(self, sampling: SamplingConfig, latency_probe: Optional[LatencyProbeConfig] = None):
        """Instantiate the monitor with the given sampling cadence."""
        self._sampling = sampling
        self._latency_probe = latency_probe or LatencyProbeConfig()
        self._last_net = psutil.net_io_counters()
        self._last_ts = time.time()
        self._net_history = deque([(self._last_ts, self._last_net.bytes_sent + self._last_net.bytes_recv)])
        self._mountpoint = _default_mountpoint()
        self._service_metrics: Optional[ServiceMetrics] = None

    def set_service_metrics(self, service_metrics: ServiceMetrics) -> None:
        """Update service metrics from an external service."""
        self._service_metrics = service_metrics

    def _bandwidth_mbps(self, now: float, counters: psutil._common.snetio) -> float:
        """
        Compute bandwidth in Mbps over a sliding window.

        Uses `SamplingConfig.network_window_seconds` as a smoothing window. If the
        window is <= 0, it falls back to the delta since the last sample.
        """
        total_bytes = counters.bytes_sent + counters.bytes_recv
        self._net_history.append((now, total_bytes))

        window_s = float(getattr(self._sampling, "network_window_seconds", 0.0) or 0.0)
        if window_s <= 0.0:
            delta_bytes = (counters.bytes_sent - self._last_net.bytes_sent) + (
                counters.bytes_recv - self._last_net.bytes_recv
            )
            delta_time = max(now - self._last_ts, 1e-6)
            return (delta_bytes * 8.0) / delta_time / 1e6

        cutoff = now - window_s
        while len(self._net_history) > 1 and self._net_history[0][0] < cutoff:
            self._net_history.popleft()

        oldest_ts, oldest_bytes = self._net_history[0]
        delta_bytes = max(0.0, total_bytes - oldest_bytes)
        delta_time = max(now - oldest_ts, 1e-6)
        return (delta_bytes * 8.0) / delta_time / 1e6

    def sample(self) -> ArchitectureSample:
        """
        Take a single snapshot of architecture metrics.
        Energy consumption and costs are left unset (provider-specific).
        """
        t_start = time.time()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_pct = psutil.virtual_memory().percent

        disk_usage = psutil.disk_usage(self._mountpoint)
        net_counters = psutil.net_io_counters()
        now_ts = time.time()
        bandwidth = self._bandwidth_mbps(now_ts, net_counters)

        latency_ms = None
        if self._latency_probe.enabled and self._latency_probe.host:
            latency_ms = self._probe_latency(
                self._latency_probe.host, self._latency_probe.port, self._latency_probe.timeout_seconds
            )

        # Update baselines for the next sample
        self._last_net = net_counters
        self._last_ts = now_ts

        collection_time_ms = (time.time() - t_start) * 1000.0

        state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=cpu_pct, memory_utilization=mem_pct),
            data=DataMetrics(
                volume_mb=disk_usage.used / 1e6,
                storage_cost=None,
                refresh_rate_s=self._sampling.interval_seconds,
                processing_time_ms=collection_time_ms,
            ),
            network=NetworkMetrics(latency_ms=latency_ms, bandwidth_mbps=bandwidth),
            service=self._service_metrics,
        )

        return ArchitectureSample(
            timestamp=datetime.now(timezone.utc),
            state=state,
            bandwidth_window_s=self._sampling.network_window_seconds,
        )

    def _probe_latency(self, host: str, port: int, timeout: float) -> Optional[float]:
        """Measure TCP connect latency to a target host/port."""
        start = time.time()
        try:
            with socket.create_connection((host, port), timeout=timeout):
                end = time.time()
                return (end - start) * 1000.0
        except OSError:
            return None
