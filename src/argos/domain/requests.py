# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Analytics request and effective configuration models.

Separates user-facing requests (with ranges) from node-facing configurations
(with fixed values), including tenant metadata, per-request resource limits,
and placement limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from argos.domain.numeric_contract import canonical_contract_value, within_contract_bounds

PRIORITY_LEVELS = ("critical", "standard", "best_effort")
SUPPORTED_RL_ALGORITHMS = ("qlearning", "dqn", "ppo", "static", "threshold")


def normalize_algorithm_name(value: str) -> str:
    """Normalize public RL algorithm identifiers."""
    raw = (value or "qlearning").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "q": "qlearning",
        "qlearning": "qlearning",
        "dqn": "dqn",
        "ppo": "ppo",
        "static": "static",
        "fixed": "static",
        "baseline": "static",
        "noop": "static",
        "threshold": "threshold",
        "reactive": "threshold",
        "heuristic": "threshold",
    }
    return aliases.get(raw, raw)


@dataclass
class ResourceLimits:
    """Soft/hard runtime resource limits for one analytics request."""

    cpu_max_percent: float = 80.0
    memory_max_percent: float = 85.0

    def validate(self) -> bool:
        return 0.0 < self.cpu_max_percent <= 100.0 and 0.0 < self.memory_max_percent <= 100.0

    def to_dict(self) -> dict[str, float]:
        return {
            "cpu_max_percent": float(self.cpu_max_percent),
            "memory_max_percent": float(self.memory_max_percent),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> ResourceLimits:
        data = data or {}
        return cls(
            cpu_max_percent=float(data.get("cpu_max_percent", 80.0)),
            memory_max_percent=float(data.get("memory_max_percent", 85.0)),
        )


@dataclass
class PlacementLimits:
    """Placement constraints for a request."""

    max_nodes: Optional[int] = None
    max_jobs_per_node: Optional[int] = None

    def validate(self) -> bool:
        if self.max_nodes is not None and self.max_nodes <= 0:
            return False
        return not (self.max_jobs_per_node is not None and self.max_jobs_per_node <= 0)

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            "max_nodes": self.max_nodes,
            "max_jobs_per_node": self.max_jobs_per_node,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> PlacementLimits:
        data = data or {}
        max_nodes = data.get("max_nodes")
        max_jobs_per_node = data.get("max_jobs_per_node")
        return cls(
            max_nodes=int(max_nodes) if max_nodes is not None else None,
            max_jobs_per_node=int(max_jobs_per_node) if max_jobs_per_node is not None else None,
        )


@dataclass
class AnalyticsRequest:
    """
    What the user (e.g., Ayuntamiento) requests.

    All requirements are expressed as ranges [min, max] so the orchestrator/RL
    can negotiate the optimal value within the allowed bounds.

    Attributes:
        request_id: Unique identifier for this analytics request.
        service_type: Type of analytics service (e.g., "heatmap").
        coverage_range: Fraction of nodes to use [min, max] (0.0-1.0).
        sample_range: Fraction of data to sample per node [min, max] (0.0-1.0).
        freshness_range: Refresh interval in seconds [min, max] (lower = fresher).
        response_time_range: Optional max response time [min, max] seconds.
        cost_budget: Optional maximum cost allowed.
        input_multiplier: Controlled input-volume multiplier used by experiments.
    """

    service_type: str = "heatmap"
    coverage_range: tuple[float, float] = (0.5, 0.8)
    sample_range: tuple[float, float] = (0.3, 0.7)
    freshness_range: tuple[float, float] = (30.0, 120.0)
    response_time_range: Optional[tuple[float, float]] = None
    cost_budget: Optional[float] = None
    profile_name: str = ""
    tenant_id: str = "default"
    priority: str = "standard"  # critical | standard | best_effort
    algorithm: str = "qlearning"
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    placement_limits: PlacementLimits = field(default_factory=PlacementLimits)
    input_multiplier: int = 1
    request_id: str = field(default_factory=lambda: str(uuid4())[:8])

    def validate(self) -> bool:
        """Check that all ranges are valid (min <= max, values in bounds)."""
        self.algorithm = normalize_algorithm_name(self.algorithm)
        if not (0.0 <= self.coverage_range[0] <= self.coverage_range[1] <= 1.0):
            return False
        if not (0.0 <= self.sample_range[0] <= self.sample_range[1] <= 1.0):
            return False
        if not (0.0 < self.freshness_range[0] <= self.freshness_range[1]):
            return False
        if self.response_time_range and not (0.0 < self.response_time_range[0] <= self.response_time_range[1]):
            return False
        if self.priority not in PRIORITY_LEVELS:
            return False
        if self.algorithm not in SUPPORTED_RL_ALGORITHMS:
            return False
        if not self.resource_limits.validate():
            return False
        if not 1 <= self.input_multiplier <= 100:
            return False
        return self.placement_limits.validate()

    def midpoint_config(self) -> EffectiveConfiguration:
        """Create an EffectiveConfiguration using the midpoint of each range."""
        return EffectiveConfiguration(
            request_id=self.request_id,
            service_type=self.service_type,
            target_coverage=canonical_contract_value(
                (self.coverage_range[0] + self.coverage_range[1]) / 2,
                self.coverage_range,
            ),
            target_sample=canonical_contract_value(
                (self.sample_range[0] + self.sample_range[1]) / 2,
                self.sample_range,
            ),
            target_freshness=canonical_contract_value(
                (self.freshness_range[0] + self.freshness_range[1]) / 2,
                self.freshness_range,
            ),
            tenant_id=self.tenant_id,
            priority=self.priority,
            algorithm=self.algorithm,
            cpu_max_percent=self.resource_limits.cpu_max_percent,
            memory_max_percent=self.resource_limits.memory_max_percent,
            input_multiplier=self.input_multiplier,
        )


@dataclass
class EffectiveConfiguration:
    """
    What the node actually executes.

    Contains fixed scalar values decided by the orchestrator/RL within the
    ranges specified in the original AnalyticsRequest.

    Attributes:
        request_id: Links back to the original AnalyticsRequest.
        service_type: Type of analytics service to run.
        target_coverage: Decided fraction of nodes (scalar).
        target_sample: Decided fraction of data per node (scalar).
        target_freshness: Decided refresh interval in seconds (scalar).
        assigned_node_count: How many nodes are assigned to this request.
        total_nodes: Total number of nodes in the cluster.
    """

    request_id: str
    service_type: str
    target_coverage: float
    target_sample: float
    target_freshness: float
    assigned_node_count: int = 1
    total_nodes: int = 3
    tenant_id: str = "default"
    priority: str = "standard"
    algorithm: str = "qlearning"
    cpu_max_percent: float = 80.0
    memory_max_percent: float = 85.0
    placement_limits: PlacementLimits = field(default_factory=PlacementLimits)
    input_multiplier: int = 1

    def to_dict(self) -> dict:
        """Serialize for API transmission."""
        return {
            "request_id": self.request_id,
            "service_type": self.service_type,
            "target_coverage": self.target_coverage,
            "target_sample": self.target_sample,
            "target_freshness": self.target_freshness,
            "assigned_node_count": self.assigned_node_count,
            "total_nodes": self.total_nodes,
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "algorithm": self.algorithm,
            "cpu_max_percent": self.cpu_max_percent,
            "memory_max_percent": self.memory_max_percent,
            "resource_limits": {
                "cpu_max_percent": self.cpu_max_percent,
                "memory_max_percent": self.memory_max_percent,
            },
            "placement_limits": self.placement_limits.to_dict(),
            "input_multiplier": self.input_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EffectiveConfiguration:
        """Deserialize from API payload."""
        return cls(
            request_id=data["request_id"],
            service_type=data["service_type"],
            target_coverage=data["target_coverage"],
            target_sample=data["target_sample"],
            target_freshness=data["target_freshness"],
            assigned_node_count=data.get("assigned_node_count", 1),
            total_nodes=data.get("total_nodes", 3),
            tenant_id=data.get("tenant_id", "default"),
            priority=data.get("priority", "standard"),
            algorithm=normalize_algorithm_name(data.get("algorithm", "qlearning")),
            cpu_max_percent=float(
                data.get(
                    "cpu_max_percent",
                    (data.get("resource_limits") or {}).get("cpu_max_percent", 80.0),
                )
            ),
            memory_max_percent=float(
                data.get(
                    "memory_max_percent",
                    (data.get("resource_limits") or {}).get("memory_max_percent", 85.0),
                )
            ),
            placement_limits=PlacementLimits.from_dict(data.get("placement_limits")),
            input_multiplier=int(data.get("input_multiplier", 1)),
        )


@dataclass
class NegotiationResult:
    """
    Result of the orchestrator's negotiation process.

    Shows what the user requested vs what the system can provide.
    """

    original_request: AnalyticsRequest
    effective_config: EffectiveConfiguration
    accepted: bool = True
    rejection_reason: Optional[str] = None
    assigned_nodes: list = field(default_factory=list)

    def coverage_fulfilled(self) -> float:
        """How well coverage was fulfilled (1.0 = fully within range)."""
        req = self.original_request
        eff = self.effective_config.target_coverage
        if within_contract_bounds(eff, *req.coverage_range):
            return 1.0
        # Outside range: measure distance
        if eff < req.coverage_range[0]:
            return eff / req.coverage_range[0]
        return req.coverage_range[1] / eff
