# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Architecture dimension models (compute, data, network)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from argos.settings import ResourceThresholds


@dataclass
class ComputeMetrics:
    """
    Compute-related metrics for a node.
    energy_consumption and computing_cost remain optional because they
    require provider-specific telemetry (e.g., AWS billing/energy APIs).
    """

    cpu_utilization: float
    memory_utilization: float
    energy_consumption: Optional[float] = None  # Watts or provider-specific unit
    computing_cost: Optional[float] = None  # Currency/time unit

    def as_dict(self) -> dict[str, float]:
        """Return a dictionary without None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataMetrics:
    """
    Data-related metrics for a node.
    volume_mb and storage_cost are proxies until a data catalogue or billing
    feed is connected.
    """

    volume_mb: Optional[float] = None
    refresh_rate_s: Optional[float] = None
    processing_time_ms: Optional[float] = None
    storage_cost: Optional[float] = None

    def as_dict(self) -> dict[str, float]:
        """Return a dictionary without None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class NetworkMetrics:
    """
    Network-related metrics for a node.
    transfer_cost is left optional until a cost model is integrated.
    """

    latency_ms: Optional[float]
    bandwidth_mbps: Optional[float]
    transfer_cost: Optional[float] = None

    def as_dict(self) -> dict[str, float]:
        """Return a dictionary without None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ServiceMetrics:
    """
    Metrics from analytics services running on the node.

    Captures the performance of services like Heatmap that consume
    node resources to produce analytics results.
    """

    service_type: str = "none"
    is_active: bool = False
    response_time_ms: float = 0.0
    data_volume_mb: float = 0.0
    sample_rate: float = 0.0
    last_request_id: Optional[str] = None

    def as_dict(self) -> dict[str, float]:
        """Return numeric metrics as dictionary."""
        return {
            "is_active": float(self.is_active),
            "response_time_ms": self.response_time_ms,
            "data_volume_mb": self.data_volume_mb,
            "sample_rate": self.sample_rate,
        }


@dataclass
class ArchitectureState:
    """
    Aggregate view of the architecture dimensions for a node.
    """

    compute: ComputeMetrics
    data: DataMetrics
    network: NetworkMetrics
    service: Optional[ServiceMetrics] = None

    def flatten(self) -> dict[str, float]:
        """
        Flatten all metrics with category prefixes for logging or exports.
        """
        features: dict[str, float] = {}
        for prefix, payload in (
            ("compute", self.compute.as_dict()),
            ("data", self.data.as_dict()),
            ("network", self.network.as_dict()),
        ):
            for key, value in payload.items():
                features[f"{prefix}.{key}"] = value

        # Include service metrics if available
        if self.service:
            features["service.type"] = hash(self.service.service_type) % 100
            for key, value in self.service.as_dict().items():
                features[f"service.{key}"] = value

        return features

    def resource_pressure(self, thresholds: ResourceThresholds) -> dict[str, float]:
        """
        Measure how much the architecture is over (or under) the accepted thresholds.
        Values >= 0 mean healthy or within limits, values > 0 mean over target.
        """
        cpu_over = max(0.0, (self.compute.cpu_utilization - thresholds.cpu_percent) / thresholds.cpu_percent)
        mem_over = max(0.0, (self.compute.memory_utilization - thresholds.memory_percent) / thresholds.memory_percent)

        bandwidth = self.network.bandwidth_mbps or 0.0
        bandwidth_over = max(0.0, (bandwidth - thresholds.bandwidth_mbps) / max(thresholds.bandwidth_mbps, 1e-6))

        return {
            "cpu": cpu_over,
            "memory": mem_over,
            "bandwidth": bandwidth_over,
        }
