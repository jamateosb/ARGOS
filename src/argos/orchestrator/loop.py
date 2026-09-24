# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Central orchestrator polling loop with RL integration.

The orchestrator actively polls all nodes via GET /metrics and
pushes configurations via POST /configure. This is the heart of
the active orchestration model.

Features:
- Integrated RL agent for adaptive configuration decisions
- Per-request metrics tracking
- Full persistence of decisions, errors, and state changes
- Proper node assignment with ceiling rounding

Future enhancements:
- Auto-deactivate idle nodes: After N iterations without work, send POST /deactivate
  to nodes that haven't been assigned any request. This saves resources.
- Dynamic node wake-up: When a new request arrives and all nodes are sleeping,
  send POST /activate to wake them up before assigning work.
- Container orchestration: Instead of pre-running containers, integrate with
  Docker API or Kubernetes to start/stop node containers on demand.
- Cost model: Track and optimize for cost_budget using real pricing APIs.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import os
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from argos.settings import SamplingConfig
from argos.domain.monitor import ArchitectureMonitor
from argos.domain.requests import PRIORITY_LEVELS, AnalyticsRequest, EffectiveConfiguration
from argos.common.utils import utc_now_iso
from argos.domain.cost import CostModelRates, estimate_request_cost
from argos.domain.numeric_contract import (
    breaches_lower_bound,
    breaches_upper_bound,
    canonical_contract_mean,
    canonical_contract_value,
)
from argos.orchestrator.control_plane import (
    LeaderElectionService,
    NatsEventBus,
    build_control_plane,
)
from argos.orchestrator.coverage import CoverageCalculator
from argos.orchestrator.environment import OrchestrationEnvironment
from argos.orchestrator.logger import ExperimentLogger
from argos.orchestrator.persistence import (
    ConfigurationChange,
    ErrorType,
    JobStatus,
    NodeAssignment,
    NodePlanApplied,
    PersistedRequest,
    PersistenceManager,
    RequestLifecycleEvent,
    RequestMetrics,
    RLDecision,
    SLODecisionEpoch,
    SLOViolation,
    TenantFairnessRecord,
    get_persistence_manager,
)
from argos.orchestrator.rl.base import RLAgent
from argos.orchestrator.rl.factory import create_agent, find_latest_policy

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Action names for logging
ACTION_NAMES = [
    "hold",
    "increase_coverage",
    "decrease_coverage",
    "increase_sample",
    "decrease_sample",
    "decrease_freshness",  # Less frequent updates
    "increase_freshness",  # More frequent updates
]

PRIORITY_SCORE = {name: idx for idx, name in enumerate(PRIORITY_LEVELS)}

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NodeInfo:
    """Information about a registered node."""

    node_id: str
    endpoint: str  # e.g., "http://node-1.example:8000"
    is_active: bool = True
    last_seen: Optional[datetime] = None
    last_metrics: Optional[dict] = None
    assigned_request_ids: list[str] = field(default_factory=list)
    health_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def assigned_request_id(self) -> Optional[str]:
        """Return the first assigned request, or None."""
        return self.assigned_request_ids[0] if self.assigned_request_ids else None


@dataclass
class OrchestrationLoopConfig:
    """Configuration for the orchestration loop."""

    poll_interval_seconds: float = 10.0
    timeout_seconds: float = 5.0
    max_health_failures: int = 3
    parallel_requests: int = 10
    enable_logging: bool = True
    enable_rl: bool = True
    enable_persistence: bool = True

    # RL Configuration
    rl_alpha: float = 0.1
    rl_gamma: float = 0.95
    rl_epsilon: float = 0.15
    rl_epsilon_decay: float = 0.995
    rl_min_epsilon: float = 0.01
    rl_exploration_decay_steps: int = 2048
    rl_seed: Optional[int] = None
    rl_training: bool = True
    rl_warm_start: bool = False
    slo_window_decisions: int = 20
    initial_coverage_quantile: float = 0.5
    initial_sample_quantile: float = 0.5
    initial_freshness_quantile: float = 0.5

    # Multi-job fallback when nodes have not advertised runtime capacity yet.
    max_jobs_per_node: int = 3

    # Backpressure / concurrency controls
    max_parallel_polls: int = 16
    max_parallel_pushes: int = 16

    # Performance budget per loop iteration
    iteration_budget_ms: float = 2000.0

    # Control plane (leader election + optional NATS gossip)
    enable_control_plane: bool = True
    control_plane_node_id: Optional[str] = None
    control_plane_lease_seconds: int = 15
    control_plane_heartbeat_interval_seconds: float = 5.0
    enable_nats_bus: bool = False
    nats_url: str = "nats://127.0.0.1:4222"
    nats_subject_prefix: str = "argos.control"
    deployment_profile: str = "local"

    # Cost model used for per-request operational cost estimates.
    cost_model_rates: CostModelRates = field(default_factory=CostModelRates)


@dataclass
class IterationMetrics:
    """Metrics for a single orchestration iteration."""

    iteration: int
    timestamp: str
    total_nodes: int
    active_nodes: int
    responding_nodes: int
    configs_pushed: int
    iteration_time_ms: float
    requests_processed: int
    rl_decisions_made: int
    errors_count: int


# =============================================================================
# Orchestration Loop
# =============================================================================


class OrchestrationLoop:
    """
    Main orchestration loop that polls nodes and pushes configurations.

    The loop:
    1. Polls all registered nodes for their current metrics
    2. Collects responses and updates node states
    3. Evaluates current coverage against request requirements
    4. Uses the RL agent to decide on configuration adjustments
    5. Pushes new configurations to affected nodes
    6. Logs all data for experiment analysis
    7. Persists state and decisions for audit trail
    """

    def __init__(
        self,
        config: OrchestrationLoopConfig = None,
        decision_callback: Optional[Callable] = None,
        logger: Optional[ExperimentLogger] = None,
        persistence: Optional[PersistenceManager] = None,
    ):
        self.config = config or OrchestrationLoopConfig()
        self._nodes: dict[str, NodeInfo] = {}
        self._active_requests: dict[str, AnalyticsRequest] = {}
        self._effective_configs: dict[str, EffectiveConfiguration] = {}
        self._pending_requests: list[AnalyticsRequest] = []
        self._request_status: dict[str, JobStatus] = {}
        self._request_status_reason: dict[str, str] = {}
        self._request_queue_order: dict[str, int] = {}
        self._next_queue_order = 0

        # External decision callback (optional)
        self._decision_callback = decision_callback

        # Logging and persistence
        self._logger = logger or ExperimentLogger()
        self._persistence = persistence or (get_persistence_manager() if self.config.enable_persistence else None)

        # Coverage calculator
        self._coverage_calculator = CoverageCalculator()

        # RL components (one environment + agent per request)
        self._rl_agents: dict[str, RLAgent] = {}
        self._rl_environments: dict[str, OrchestrationEnvironment] = {}
        self._agent_seeds: dict[str, Optional[int]] = {}
        self._next_agent_seed_ordinal = 0
        self._pending_policy_loads: dict[str, str] = {}
        self._frozen_policy_fingerprints: dict[str, str] = {}

        # State tracking
        self._running = False
        self._iteration_count = 0

        # Per-request tracking
        self._request_iterations: dict[str, int] = {}
        self._request_latencies: dict[str, list[float]] = {}
        self._tenant_fairness_debt: dict[str, float] = {}
        self._recent_slo_windows: dict[str, deque[int]] = {}

        # Shared auth header for node API calls when STATUS_API_TOKEN is enabled.
        node_api_token = os.getenv("STATUS_API_TOKEN", "").strip()
        self._node_api_headers = {"X-API-Key": node_api_token} if node_api_token else {}

        # Control-plane state (leader election + optional NATS bus).
        self._control_plane_node_id = (
            self.config.control_plane_node_id or os.getenv("ARGOS_MACHINE_ID") or socket.gethostname()
        )
        self._leader_election: Optional[LeaderElectionService] = None
        self._event_bus: Optional[NatsEventBus] = None
        self._event_bus_connected: bool = False
        self._last_heartbeat_at: float = 0.0
        self._last_capability_publish_at: float = 0.0
        self._is_leader: bool = True
        self._plan_version: int = 0

        if (
            self.config.deployment_profile == "aws"
            and self.config.enable_control_plane
            and not self.config.enable_nats_bus
        ):
            raise ValueError("AWS deployment profile requires NATS-enabled control-plane")

        if self.config.enable_control_plane:
            self._leader_election = build_control_plane(
                node_id=self._control_plane_node_id,
                persistence=self._persistence,
            )
            self._leader_election.lease_seconds = max(1, int(self.config.control_plane_lease_seconds))
            self._is_leader = self._leader_election.leader_id in (None, self._control_plane_node_id)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _priority_rank(priority: str) -> int:
        return PRIORITY_SCORE.get(priority, PRIORITY_SCORE["standard"])

    @staticmethod
    def _extract_compute_metrics(metrics: dict) -> tuple[float, float]:
        """Extract CPU and memory from the structured or the flat node payload."""
        state = metrics.get("state", {})
        compute = state.get("compute")
        if isinstance(compute, dict):
            cpu = float(compute.get("cpu_utilization", 0.0) or 0.0)
            mem = float(compute.get("memory_utilization", 0.0) or 0.0)
        else:
            cpu = float(state.get("compute.cpu_utilization", 0.0) or 0.0)
            mem = float(state.get("compute.memory_utilization", 0.0) or 0.0)

        # Fallback: dedicated flattened map.
        if cpu <= 0.0 or mem <= 0.0:
            state_flat = metrics.get("state_flat", {})
            if isinstance(state_flat, dict):
                cpu = cpu if cpu > 0.0 else float(state_flat.get("compute.cpu_utilization", 0.0) or 0.0)
                mem = mem if mem > 0.0 else float(state_flat.get("compute.memory_utilization", 0.0) or 0.0)

        return cpu, mem

    def _observe_node_plan_version(self, metrics: dict) -> None:
        """Advance local plan version from node state after a leadership change."""
        try:
            observed = int(metrics.get("current_plan_version", 0) or 0)
        except (TypeError, ValueError):
            return

        if observed <= self._plan_version:
            return

        self._plan_version = observed
        if self._leader_election:
            self._leader_election.register_plan_version(
                observed,
                source_node=str(metrics.get("plan_source") or metrics.get("node_id") or "unknown"),
            )

    @staticmethod
    def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
        """Parse an ISO timestamp defensively."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def _extract_request_analytics_metrics(cls, metrics: dict, request_id: str) -> dict[str, Any]:
        """
        Extract request-scoped analytics metrics from a node payload.

        The node payload can contain multiple analytics runtimes. For
        per-request evaluation we prefer those request-scoped metrics over the
        generic architecture monitor block to avoid mixing unrelated workloads.
        """
        analytics = metrics.get("analytics")
        candidate: Optional[dict] = None

        if isinstance(analytics, list):
            candidate = next(
                (
                    item
                    for item in analytics
                    if isinstance(item, dict) and str(item.get("request_id", "")) == request_id
                ),
                None,
            )
        elif str(metrics.get("request_id", "")) == request_id:
            candidate = metrics

        if not candidate:
            return {}

        service_metrics = candidate.get("service_metrics")
        if not isinstance(service_metrics, dict):
            service_metrics = {}

        updated_at = cls._parse_iso_timestamp(candidate.get("last_result_at"))
        freshness_age_s: Optional[float] = None
        if updated_at is not None:
            freshness_age_s = max(
                0.0, (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds()
            )

        sample_rate = candidate.get("sample_rate_applied")

        processing_ms = candidate.get("last_processing_time_ms")
        if processing_ms is None:
            processing_ms = service_metrics.get("last_processing_ms")

        data_volume_bytes = candidate.get("last_data_volume_bytes")
        if data_volume_bytes is None:
            data_volume_bytes = service_metrics.get("last_data_bytes")

        hotspots = candidate.get("last_hotspots")
        if hotspots is None:
            hotspots = service_metrics.get("last_hotspots")

        spatial_fidelity = candidate.get("spatial_fidelity")
        if spatial_fidelity is None:
            spatial_fidelity = service_metrics.get("last_spatial_fidelity")
        hotspot_recall = candidate.get("hotspot_recall")
        if hotspot_recall is None:
            hotspot_recall = service_metrics.get("last_hotspot_recall")

        return {
            "sample_rate": float(sample_rate) if sample_rate is not None else None,
            "processing_time_ms": float(processing_ms) if processing_ms is not None else None,
            "data_volume_bytes": float(data_volume_bytes) if data_volume_bytes is not None else None,
            "hotspots": float(hotspots) if hotspots is not None else None,
            "spatial_fidelity": float(spatial_fidelity) if spatial_fidelity is not None else None,
            "hotspot_recall": float(hotspot_recall) if hotspot_recall is not None else None,
            "freshness_age_s": freshness_age_s,
            "output_observed": updated_at is not None,
        }

    def _tenant_debt(self, tenant_id: str) -> float:
        return self._tenant_fairness_debt.get(tenant_id, 0.0)

    def _job_status(self, request_id: str) -> JobStatus:
        return self._request_status.get(request_id, JobStatus.ACCEPTED)

    def _set_job_status(self, request_id: str, status: JobStatus, reason: str = "") -> None:
        previous = self._request_status.get(request_id)
        self._request_status[request_id] = status
        if reason:
            self._request_status_reason[request_id] = reason
        if self._persistence and previous != status:
            self._persistence.update_request_status(request_id, status, reason or None)

    def get_job_status(self, request_id: str) -> JobStatus:
        """Return the current in-memory status for a request."""
        return self._job_status(request_id)

    def get_job_status_reason(self, request_id: str) -> str:
        """Return the latest status reason for a request."""
        return self._request_status_reason.get(request_id, "")

    def pending_request_count(self) -> int:
        """Return queued request count."""
        return len([r for r in self._pending_requests if self._job_status(r.request_id) == JobStatus.QUEUED])

    def running_request_count(self) -> int:
        """Return requests currently eligible for orchestration."""
        return len(
            [
                rid
                for rid in self._active_requests
                if self._job_status(rid) in (JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.DEGRADED)
            ]
        )

    def _persist_lifecycle(
        self,
        request_id: str,
        event_type: str,
        status: Any,
        *,
        reason: str = "",
        assigned_nodes: Optional[list[str]] = None,
        pending_position: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._persistence:
            return
        status_value = status.value if isinstance(status, JobStatus) else str(status)
        self._persistence.save_request_lifecycle(
            RequestLifecycleEvent(
                request_id=request_id,
                timestamp=utc_now_iso(),
                event_type=event_type,
                status=status_value,
                reason=reason,
                assigned_nodes=assigned_nodes or [],
                pending_position=pending_position,
                details=details or {},
            )
        )

    def _capacity_snapshot(self, request: Optional[AnalyticsRequest] = None) -> dict[str, Any]:
        """Return cluster/node capacity seen by the admission controller."""
        total = 0
        used = 0
        free = 0
        nodes: dict[str, dict[str, Any]] = {}
        for node in self._nodes.values():
            if not node.is_active:
                nodes[node.node_id] = {
                    "active": False,
                    "used": len(node.assigned_request_ids),
                    "free": 0,
                    "total": 0,
                }
                continue
            limit = self._node_capacity_limit(node, request) if request else self.config.max_jobs_per_node
            node_used = len(node.assigned_request_ids)
            node_free = max(0, limit - node_used)
            total += limit
            used += node_used
            free += node_free
            nodes[node.node_id] = {
                "active": True,
                "used": node_used,
                "free": node_free,
                "total": limit,
            }
        return {"total": total, "used": used, "free": free, "nodes": nodes}

    def _active_node_count(self) -> int:
        return len([n for n in self._nodes.values() if n.is_active])

    def _minimum_node_count(self, request: AnalyticsRequest) -> int:
        active_nodes = self._active_node_count()
        if active_nodes <= 0:
            return 1
        min_count = max(1, math.ceil(active_nodes * request.coverage_range[0]))
        if request.placement_limits.max_nodes is not None:
            min_count = min(min_count, request.placement_limits.max_nodes)
        return min_count

    def _desired_node_count(self, request: AnalyticsRequest, config: EffectiveConfiguration) -> int:
        active_nodes = self._active_node_count()
        desired = max(1, math.ceil(active_nodes * config.target_coverage))
        if request.placement_limits.max_nodes is not None:
            desired = min(desired, request.placement_limits.max_nodes)
        return desired

    def _available_nodes_for_request(self, request: AnalyticsRequest, config: EffectiveConfiguration) -> list[NodeInfo]:
        available = [
            n
            for n in self._nodes.values()
            if n.is_active
            and len(n.assigned_request_ids) < self._node_capacity_limit(n, request)
            and config.request_id not in n.assigned_request_ids
        ]
        available.sort(key=lambda n: self._node_load_key(n, request))
        return available

    def _queue_position(self, request_id: str) -> Optional[int]:
        ordered = self._ordered_pending_requests()
        for idx, request in enumerate(ordered, 1):
            if request.request_id == request_id:
                return idx
        return None

    def _ordered_pending_requests(self) -> list[AnalyticsRequest]:
        return sorted(
            [r for r in self._pending_requests if self._job_status(r.request_id) == JobStatus.QUEUED],
            key=lambda r: (
                self._priority_rank(r.priority),
                -self._tenant_debt(r.tenant_id),
                self._request_queue_order.get(r.request_id, 0),
            ),
        )

    def _mark_queued(self, request: AnalyticsRequest, reason: str) -> None:
        if request.request_id not in [r.request_id for r in self._pending_requests]:
            self._pending_requests.append(request)
        if request.request_id not in self._request_queue_order:
            self._request_queue_order[request.request_id] = self._next_queue_order
            self._next_queue_order += 1
        self._set_job_status(request.request_id, JobStatus.QUEUED, reason)
        self._persist_lifecycle(
            request.request_id,
            "queued",
            JobStatus.QUEUED,
            reason=reason,
            pending_position=self._queue_position(request.request_id),
            details=self._capacity_snapshot(request),
        )

    def _mark_rejected(self, request: AnalyticsRequest, reason: str) -> None:
        self._pending_requests = [r for r in self._pending_requests if r.request_id != request.request_id]
        self._set_job_status(request.request_id, JobStatus.REJECTED, reason)
        self._persist_lifecycle(
            request.request_id,
            "rejected",
            JobStatus.REJECTED,
            reason=reason,
            details=self._capacity_snapshot(request),
        )

    def _placement_impossible(self, request: AnalyticsRequest) -> Optional[str]:
        active_nodes = self._active_node_count()
        if active_nodes <= 0:
            return None
        required = max(1, math.ceil(active_nodes * request.coverage_range[0]))
        if request.placement_limits.max_nodes is not None and request.placement_limits.max_nodes < required:
            return "placement_limits_below_coverage_min"
        return None

    def _admit_request(
        self,
        request: AnalyticsRequest,
        config: EffectiveConfiguration,
        *,
        reason: str,
        persist_initial_config: bool = False,
    ) -> list[str]:
        should_persist_initial_config = persist_initial_config or request.request_id not in self._request_iterations
        config.assigned_node_count = self._desired_node_count(request, config)
        config.total_nodes = len(self._nodes)
        config.placement_limits = request.placement_limits
        self._initialize_rl_for_request(request, config)
        assigned_nodes = self._assign_nodes(request, config)
        config.assigned_node_count = len(assigned_nodes)
        self._pending_requests = [r for r in self._pending_requests if r.request_id != request.request_id]
        self._request_queue_order.pop(request.request_id, None)
        self._request_iterations.setdefault(request.request_id, 0)
        self._request_latencies.setdefault(request.request_id, [])
        self._set_job_status(request.request_id, JobStatus.ACCEPTED, reason)

        if self._persistence and should_persist_initial_config:
            self._persistence.save_config_change(
                ConfigurationChange(
                    request_id=request.request_id,
                    timestamp=utc_now_iso(),
                    iteration=0,
                    prev_coverage=None,
                    prev_sample=None,
                    prev_freshness=None,
                    prev_node_count=None,
                    new_coverage=config.target_coverage,
                    new_sample=config.target_sample,
                    new_freshness=config.target_freshness,
                    new_node_count=config.assigned_node_count,
                    reason="initial",
                )
            )
        for node_id in assigned_nodes:
            if self._persistence:
                self._persistence.save_assignment(
                    NodeAssignment(
                        node_id=node_id,
                        request_id=request.request_id,
                        assigned_at=utc_now_iso(),
                        reason=reason,
                    )
                )
            self._persist_lifecycle(
                request.request_id,
                "assigned",
                JobStatus.ACCEPTED,
                reason=reason,
                assigned_nodes=[node_id],
            )
        self._persist_lifecycle(
            request.request_id,
            "admitted",
            JobStatus.ACCEPTED,
            reason=reason,
            assigned_nodes=assigned_nodes,
            details=self._capacity_snapshot(request),
        )
        return assigned_nodes

    def _try_admit_or_queue(
        self,
        request: AnalyticsRequest,
        config: EffectiveConfiguration,
        *,
        initial: bool,
    ) -> list[str]:
        impossible_reason = self._placement_impossible(request)
        if impossible_reason:
            self._mark_rejected(request, impossible_reason)
            return []

        required = self._minimum_node_count(request)
        available = self._available_nodes_for_request(request, config)
        if len(available) < required:
            if initial or self._job_status(request.request_id) != JobStatus.QUEUED:
                self._mark_queued(request, "insufficient_capacity")
            return []

        reason = "initial_admission" if initial else "queued_capacity_available"
        return self._admit_request(
            request,
            config,
            reason=reason,
            persist_initial_config=initial,
        )

    def _persist_capacity_snapshot(self) -> None:
        if not self._persistence:
            return
        self._persist_lifecycle(
            "cluster",
            "capacity_snapshot",
            "snapshot",
            details={
                **self._capacity_snapshot(),
                "pending_count": self.pending_request_count(),
                "running_count": self.running_request_count(),
                "iteration": self._iteration_count,
            },
        )

    def _retry_pending_requests(self) -> None:
        for request in self._ordered_pending_requests():
            config = self._effective_configs.get(request.request_id)
            if not config:
                continue
            self._try_admit_or_queue(request, config, initial=False)
        self._persist_capacity_snapshot()

    def _recent_slo_violation_count(self, request_id: str) -> int:
        """Count decision epochs with a violation in the request's recent window."""
        return sum(self._recent_slo_windows.get(request_id, ()))

    def _record_slo_decision_epoch(self, request_id: str, violated: bool) -> None:
        """Append one binary outcome for a completed request decision epoch."""
        window_size = max(1, int(self.config.slo_window_decisions))
        window = self._recent_slo_windows.get(request_id)
        if window is None or window.maxlen != window_size:
            previous = list(window or ())[-window_size:]
            window = deque(previous, maxlen=window_size)
            self._recent_slo_windows[request_id] = window
        window.append(1 if violated else 0)
        if self._persistence:
            self._persistence.save_slo_decision_epoch(
                SLODecisionEpoch(
                    request_id=request_id,
                    timestamp=utc_now_iso(),
                    iteration=self._request_iterations.get(request_id, 0),
                    violated=bool(violated),
                    recent_violation_count=sum(window),
                    window_size=window_size,
                )
            )

    async def _ensure_event_bus_connected(self) -> None:
        """Connect to NATS and subscribe to control-plane subjects."""
        if not self.config.enable_control_plane or not self.config.enable_nats_bus:
            return
        if self._event_bus_connected:
            return

        try:
            self._event_bus = NatsEventBus(self.config.nats_url)
            await self._event_bus.connect()
            prefix = self.config.nats_subject_prefix

            async def _on_capability(payload: dict[str, Any]) -> None:
                if not self._leader_election:
                    return
                node_id = str(payload.get("node_id", "")).strip()
                if not node_id or node_id == self._control_plane_node_id:
                    return
                score = float(payload.get("benchmark_score", 0.0) or 0.0)
                timestamp = float(payload.get("heartbeat_time", time.time()) or time.time())
                self._leader_election.register_score(node_id, score, heartbeat_time=timestamp)

            async def _on_heartbeat(payload: dict[str, Any]) -> None:
                if not self._leader_election:
                    return
                node_id = str(payload.get("node_id", "")).strip()
                if not node_id or node_id == self._control_plane_node_id:
                    return
                score = payload.get("score")
                heartbeat_time = payload.get("heartbeat_time")
                hb = float(heartbeat_time) if heartbeat_time is not None else time.time()
                score_val = float(score) if score is not None else None
                self._leader_election.register_remote_heartbeat(node_id, score=score_val, heartbeat_time=hb)

            await self._event_bus.subscribe(f"{prefix}.capability", _on_capability)
            await self._event_bus.subscribe(f"{prefix}.heartbeat", _on_heartbeat)
            self._event_bus_connected = True
            logger.info("Connected to NATS control-plane bus at %s", self.config.nats_url)
        except Exception as exc:
            self._event_bus_connected = False
            self._event_bus = None
            logger.warning("Control-plane NATS disabled (%s)", exc)

    async def _control_plane_tick(self) -> None:
        """Emit heartbeat and refresh leadership state."""
        if not self._leader_election:
            self._is_leader = True
            return

        if self.config.enable_nats_bus:
            await self._ensure_event_bus_connected()

        now = time.time()
        hb_interval = max(0.5, float(self.config.control_plane_heartbeat_interval_seconds))

        if (now - self._last_heartbeat_at) >= hb_interval:
            self._leader_election.heartbeat(self._control_plane_node_id)
            self._last_heartbeat_at = now

            if self._event_bus_connected and self._event_bus:
                try:
                    await self._event_bus.publish(
                        f"{self.config.nats_subject_prefix}.heartbeat",
                        {
                            "node_id": self._control_plane_node_id,
                            "score": self._leader_election.get_score(self._control_plane_node_id),
                            "heartbeat_time": now,
                            "timestamp": utc_now_iso(),
                        },
                    )
                except Exception:
                    self._event_bus_connected = False
                    self._event_bus = None

        if self._event_bus_connected and self._event_bus and (now - self._last_capability_publish_at) >= 60.0:
            local_cap = self._leader_election.local_capability
            if local_cap is None:
                local_cap = self._leader_election.benchmark_local_node()
            try:
                await self._event_bus.publish(
                    f"{self.config.nats_subject_prefix}.capability",
                    {
                        **local_cap.to_dict(),
                        "heartbeat_time": now,
                    },
                )
                self._last_capability_publish_at = now
            except Exception:
                self._event_bus_connected = False
                self._event_bus = None

        leader = self._leader_election.elect()
        self._is_leader = leader in (None, self._control_plane_node_id)

    async def close_async(self) -> None:
        """Close async resources (NATS connection)."""
        if self._event_bus:
            with contextlib.suppress(Exception):
                await self._event_bus.close()
        self._event_bus_connected = False
        self._event_bus = None

    def get_control_plane_state(self) -> dict[str, Any]:
        """Expose current control-plane state for API/diagnostics."""
        if not self._leader_election:
            return {
                "enabled": False,
                "node_id": self._control_plane_node_id,
                "is_leader": True,
                "leader_id": None,
                "nats_enabled": bool(self.config.enable_nats_bus),
                "nats_connected": False,
            }
        snap = self._leader_election.snapshot()
        return {
            "enabled": True,
            "node_id": self._control_plane_node_id,
            "is_leader": self._is_leader,
            "leader_id": snap.get("leader_id"),
            "leader_term": snap.get("leader_term", 0),
            "lease_seconds": snap.get("lease_seconds"),
            "plan_version": self._plan_version,
            "source": snap.get("source", self._control_plane_node_id),
            "stale_peers": snap.get("stale_peers", []),
            "capability_sync_status": snap.get("capability_sync_status", "local_only"),
            "nats_enabled": bool(self.config.enable_nats_bus),
            "nats_connected": self._event_bus_connected,
            "nodes": snap.get("nodes", {}),
        }

    # =========================================================================
    # Node Management
    # =========================================================================

    def register_node(self, node_id: str, endpoint: str) -> None:
        """Register a node with the orchestrator."""
        self._nodes[node_id] = NodeInfo(node_id=node_id, endpoint=endpoint)

        if self._persistence:
            self._persistence.save_assignment(
                NodeAssignment(
                    node_id=node_id,
                    request_id="",
                    assigned_at=utc_now_iso(),
                    reason="registered",
                )
            )

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the orchestrator."""
        node = self._nodes.pop(node_id, None)

        if node and node.assigned_request_ids and self._persistence:
            for rid in list(node.assigned_request_ids):
                self._persistence.save_unassignment(
                    node_id=node_id,
                    request_id=rid,
                    reason="unregistered",
                )

    # =========================================================================
    # Request Management
    # =========================================================================

    def submit_request(self, request: AnalyticsRequest) -> EffectiveConfiguration:
        """
        Submit a new analytics request.

        The orchestrator negotiates an effective configuration within
        the requested ranges and assigns nodes using ceiling rounding.
        """
        if not request.validate():
            if self._persistence:
                self._persistence.save_error(
                    ErrorType.VALIDATION,
                    f"Invalid analytics request: {request.request_id}",
                    request_id=request.request_id,
                )
            raise ValueError("Invalid analytics request")

        self._active_requests[request.request_id] = request

        effective = request.midpoint_config()
        effective.target_coverage = self._range_quantile(
            request.coverage_range,
            self.config.initial_coverage_quantile,
        )
        effective.target_sample = self._range_quantile(
            request.sample_range,
            self.config.initial_sample_quantile,
        )
        effective.target_freshness = self._range_quantile(
            request.freshness_range,
            self.config.initial_freshness_quantile,
        )
        effective.assigned_node_count = self._desired_node_count(request, effective)
        effective.total_nodes = len(self._nodes)
        effective.placement_limits = request.placement_limits
        self._effective_configs[request.request_id] = effective

        # Persist the request before status transitions so update_request_status()
        # can append the lifecycle state cleanly.
        if self._persistence:
            self._persistence.save_request(
                PersistedRequest(
                    request_id=request.request_id,
                    service_type=request.service_type,
                    coverage_min=request.coverage_range[0],
                    coverage_max=request.coverage_range[1],
                    sample_min=request.sample_range[0],
                    sample_max=request.sample_range[1],
                    freshness_min=request.freshness_range[0],
                    freshness_max=request.freshness_range[1],
                    cost_budget=request.cost_budget,
                    status=JobStatus.PENDING,
                    created_at=utc_now_iso(),
                    updated_at=utc_now_iso(),
                    metadata={
                        "tenant_id": request.tenant_id,
                        "profile_name": request.profile_name,
                        "priority": request.priority,
                        "algorithm": request.algorithm,
                        "resource_limits": request.resource_limits.to_dict(),
                        "placement_limits": request.placement_limits.to_dict(),
                        "input_multiplier": request.input_multiplier,
                    },
                )
            )
            self._persist_lifecycle(
                request.request_id,
                "submitted",
                JobStatus.PENDING,
                details={
                    "tenant_id": request.tenant_id,
                    "profile_name": request.profile_name,
                    "priority": request.priority,
                    "algorithm": request.algorithm,
                    "resource_limits": request.resource_limits.to_dict(),
                    "placement_limits": request.placement_limits.to_dict(),
                    "input_multiplier": request.input_multiplier,
                },
            )

        self._request_status[request.request_id] = JobStatus.PENDING
        self._request_status_reason[request.request_id] = "submitted"
        self._try_admit_or_queue(request, effective, initial=True)

        return effective

    def cancel_request(self, request_id: str, reason: str = "cancelled") -> bool:
        """Cancel a running request."""
        if request_id not in self._active_requests:
            return False

        # Teardown runtime on nodes before unassigning.
        teardown_targets: list[NodeInfo] = []

        # Unassign all nodes from this request
        for node in self._nodes.values():
            if request_id in node.assigned_request_ids:
                teardown_targets.append(node)
                if self._persistence:
                    self._persistence.save_unassignment(
                        node_id=node.node_id,
                        request_id=request_id,
                        reason=reason,
                    )
                self._persist_lifecycle(
                    request_id,
                    "unassigned",
                    JobStatus.CANCELLED,
                    reason=reason,
                    assigned_nodes=[node.node_id],
                )
                node.assigned_request_ids.remove(request_id)

        for node in teardown_targets:
            self._teardown_request_on_node(node, request_id)

        self._pending_requests = [r for r in self._pending_requests if r.request_id != request_id]
        self._request_queue_order.pop(request_id, None)
        self._set_job_status(request_id, JobStatus.CANCELLED, reason)
        self._persist_lifecycle(request_id, "cancelled", JobStatus.CANCELLED, reason=reason)

        # Clean up
        del self._active_requests[request_id]
        self._effective_configs.pop(request_id, None)
        self._rl_agents.pop(request_id, None)
        self._rl_environments.pop(request_id, None)
        self._agent_seeds.pop(request_id, None)
        self._pending_policy_loads.pop(request_id, None)
        self._frozen_policy_fingerprints.pop(request_id, None)
        self._request_iterations.pop(request_id, None)
        self._request_latencies.pop(request_id, None)
        self._recent_slo_windows.pop(request_id, None)
        self._request_status.pop(request_id, None)
        self._request_status_reason.pop(request_id, None)

        # Update persistence
        if self._persistence:
            self._persistence.update_request_status(request_id, JobStatus.CANCELLED, reason)

        return True

    def _teardown_request_on_node(self, node: NodeInfo, request_id: str) -> None:
        """Best-effort runtime teardown on a node when request is cancelled."""
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                client.delete(
                    f"{node.endpoint}/analytics/{request_id}",
                    headers=self._node_api_headers,
                )
        except Exception as exc:
            if self._persistence:
                self._persistence.save_error(
                    ErrorType.CONFIGURE_PUSH,
                    f"Failed teardown for request {request_id} on node {node.node_id}: {exc}",
                    node_id=node.node_id,
                    request_id=request_id,
                )

    def _initialize_rl_for_request(self, request: AnalyticsRequest, effective: EffectiveConfiguration) -> None:
        """Initialize RL agent and environment for a request."""
        if not self.config.enable_rl:
            return

        existing_environment = self._rl_environments.get(request.request_id)
        if existing_environment is not None:
            existing_environment.set_request(request, effective)
        else:
            monitor = ArchitectureMonitor(SamplingConfig())
            environment = OrchestrationEnvironment(monitor)
            environment.set_request(request, effective)
            self._rl_environments[request.request_id] = environment

        if request.request_id in self._rl_agents:
            self._apply_pending_policy_load(request.request_id)
            return

        agent_seed = self._seed_for_request(request.request_id)
        agent, metadata = create_agent(
            algorithm=request.algorithm,
            learning_rate=self.config.rl_alpha,
            discount_factor=self.config.rl_gamma,
            exploration_rate=self.config.rl_epsilon,
            exploration_decay=self.config.rl_epsilon_decay,
            min_exploration_rate=self.config.rl_min_epsilon,
            exploration_decay_steps=self.config.rl_exploration_decay_steps,
            seed=agent_seed,
        )
        self._rl_agents[request.request_id] = agent

        if metadata.get("fallback_reason") and self._persistence:
            self._persistence.save_error(
                ErrorType.RL_DECISION,
                f"Request {request.request_id} fell back to qlearning: {metadata['fallback_reason']}",
                request_id=request.request_id,
            )

        if self.config.rl_warm_start:
            self._try_load_policy(agent)
        self._apply_pending_policy_load(request.request_id)

    def _seed_for_request(self, request_id: str) -> Optional[int]:
        """Derive a stable per-request seed without touching process-global RNGs."""
        if request_id in self._agent_seeds:
            return self._agent_seeds[request_id]
        base_seed = self.config.rl_seed
        if base_seed is None:
            derived = None
        else:
            ordinal = self._next_agent_seed_ordinal
            self._next_agent_seed_ordinal += 1
            payload = f"{base_seed}:{ordinal}".encode()
            derived = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        self._agent_seeds[request_id] = derived
        return derived

    def _apply_pending_policy_load(self, request_id: str) -> Optional[str]:
        """Apply and verify a deferred frozen-policy load after admission."""
        path = self._pending_policy_loads.get(request_id)
        if path is None:
            return None
        agent = self._rl_agents.get(request_id)
        if agent is None:
            raise RuntimeError(f"Cannot apply deferred policy before agent admission: {request_id}")
        agent.load(path)
        agent.reset_run_metrics()
        fingerprint = agent.policy_fingerprint()
        self._frozen_policy_fingerprints[request_id] = fingerprint
        self._pending_policy_loads.pop(request_id, None)
        return fingerprint

    @staticmethod
    def _range_quantile(bounds: tuple[float, float], quantile: float) -> float:
        """Return a clamped linear quantile inside a request range."""
        fraction = max(0.0, min(1.0, float(quantile)))
        return canonical_contract_value(
            bounds[0] + fraction * (bounds[1] - bounds[0]),
            bounds,
        )

    def _assign_nodes(self, request: AnalyticsRequest, config: EffectiveConfiguration) -> list[str]:
        """Assign available nodes to a request.

        Supports multi-job: a node can serve up to ``max_jobs_per_node``
        concurrent requests.  Nodes are sorted by current load (fewest
        assigned jobs first) to spread work evenly.
        """
        available = self._available_nodes_for_request(request, config)
        assigned = available[: config.assigned_node_count]

        for node in assigned:
            node.assigned_request_ids.append(config.request_id)

        return [n.node_id for n in assigned]

    def _request_assigned_nodes(self, request_id: str) -> list[NodeInfo]:
        return [n for n in self._nodes.values() if request_id in n.assigned_request_ids]

    def _unassign_inactive_node_workloads(self) -> int:
        """Drop logical assignments from nodes that are no longer active."""
        removed = 0
        for node in self._nodes.values():
            if node.is_active or not node.assigned_request_ids:
                continue
            for request_id in list(node.assigned_request_ids):
                node.assigned_request_ids.remove(request_id)
                removed += 1
                if self._persistence:
                    self._persistence.save_unassignment(
                        node_id=node.node_id,
                        request_id=request_id,
                        reason="node_inactive",
                    )
                self._persist_lifecycle(
                    request_id,
                    "unassigned",
                    self._job_status(request_id),
                    reason="node_inactive",
                    assigned_nodes=[node.node_id],
                )
        return removed

    def _node_capacity_limit(self, node: NodeInfo, request: AnalyticsRequest) -> int:
        """Return the per-node concurrency limit for one request placement."""
        if request.placement_limits.max_jobs_per_node is not None:
            return request.placement_limits.max_jobs_per_node

        metrics = node.last_metrics or {}
        try:
            advertised = int(metrics.get("max_analytics_per_node") or 0)
        except (TypeError, ValueError):
            advertised = 0
        if advertised > 0:
            return advertised

        return self.config.max_jobs_per_node

    def _node_load_key(self, node: NodeInfo, request: AnalyticsRequest) -> tuple[float, int, str]:
        """Sort nodes by relative and absolute load for stable placement."""
        capacity = max(1, self._node_capacity_limit(node, request))
        assigned = len(node.assigned_request_ids)
        return assigned / capacity, assigned, node.node_id

    def _persist_coverage_violation(
        self,
        *,
        request: AnalyticsRequest,
        config: EffectiveConfiguration,
        current_coverage: float,
        active_assigned: int,
        reason: str,
    ) -> bool:
        """Persist either lower or upper coverage breaches and report the outcome."""
        lower = breaches_lower_bound(current_coverage, request.coverage_range[0])
        upper = breaches_upper_bound(current_coverage, request.coverage_range[1])
        if not lower and not upper:
            return False
        limit = request.coverage_range[0] if lower else request.coverage_range[1]
        if self._persistence:
            self._persistence.save_slo_violation(
                SLOViolation(
                    request_id=request.request_id,
                    tenant_id=request.tenant_id,
                    node_id="cluster",
                    timestamp=utc_now_iso(),
                    metric="coverage",
                    measured_value=current_coverage,
                    limit_value=limit,
                    severity="critical" if active_assigned == 0 else "warning",
                    details={
                        "bound": "minimum" if lower else "maximum",
                        "reason": reason,
                        "assigned_node_count": active_assigned,
                        "desired_node_count": config.assigned_node_count,
                        "total_nodes": len(self._nodes),
                        "active_nodes": self._active_node_count(),
                    },
                )
            )
        return True

    def _persist_observed_quality_violations(
        self,
        *,
        request: AnalyticsRequest,
        actual_sample: Optional[float],
        freshness_age_s: Optional[float],
    ) -> bool:
        """Persist observed quality breaches and report whether any occurred."""
        violated = False
        sample_breaches = actual_sample is not None and (
            breaches_lower_bound(actual_sample, request.sample_range[0])
            or breaches_upper_bound(actual_sample, request.sample_range[1])
        )
        if sample_breaches:
            violated = True
            lower_violation = actual_sample < request.sample_range[0]
            limit = request.sample_range[0] if lower_violation else request.sample_range[1]
            if self._persistence:
                self._persistence.save_slo_violation(
                    SLOViolation(
                        request_id=request.request_id,
                        tenant_id=request.tenant_id,
                        node_id="cluster",
                        timestamp=utc_now_iso(),
                        metric="sample_rate",
                        measured_value=actual_sample,
                        limit_value=limit,
                        severity="critical" if abs(actual_sample - limit) > 0.1 else "warning",
                        details={"bound": "minimum" if lower_violation else "maximum", "source": "service_output"},
                    )
                )

        if freshness_age_s is not None and breaches_upper_bound(freshness_age_s, request.freshness_range[1]):
            violated = True
            excess = freshness_age_s - request.freshness_range[1]
            if self._persistence:
                self._persistence.save_slo_violation(
                    SLOViolation(
                        request_id=request.request_id,
                        tenant_id=request.tenant_id,
                        node_id="cluster",
                        timestamp=utc_now_iso(),
                        metric="freshness_age_seconds",
                        measured_value=freshness_age_s,
                        limit_value=request.freshness_range[1],
                        severity="critical" if excess > max(10.0, request.freshness_range[1] * 0.25) else "warning",
                        details={"bound": "maximum", "source": "last_completed_service_output"},
                    )
                )
        return violated

    def _rebalance_request_assignments(self, request: AnalyticsRequest, config: EffectiveConfiguration) -> None:
        """
        Rebalance node assignments to match current target coverage.

        This is required for dynamic scale-out/scale-in when RL changes
        `target_coverage` over time.
        """
        desired = max(1, config.assigned_node_count)
        if request.placement_limits.max_nodes is not None:
            desired = min(desired, request.placement_limits.max_nodes)

        current_nodes = self._request_assigned_nodes(config.request_id)
        current_count = len(current_nodes)

        if current_count < desired:
            needed = desired - current_count
            candidates = [
                n
                for n in self._nodes.values()
                if n.is_active
                and len(n.assigned_request_ids) < self._node_capacity_limit(n, request)
                and config.request_id not in n.assigned_request_ids
            ]
            candidates.sort(key=lambda n: self._node_load_key(n, request))
            for node in candidates[:needed]:
                node.assigned_request_ids.append(config.request_id)
                if self._persistence:
                    self._persistence.save_assignment(
                        NodeAssignment(
                            node_id=node.node_id,
                            request_id=config.request_id,
                            assigned_at=utc_now_iso(),
                            reason="rebalance_scale_out",
                        )
                    )
                self._persist_lifecycle(
                    config.request_id,
                    "assigned",
                    self._job_status(config.request_id),
                    reason="rebalance_scale_out",
                    assigned_nodes=[node.node_id],
                )

        elif current_count > desired:
            remove_count = current_count - desired
            # Remove from most loaded nodes first.
            removable = sorted(
                current_nodes,
                key=lambda n: self._node_load_key(n, request),
                reverse=True,
            )
            for node in removable[:remove_count]:
                try:
                    node.assigned_request_ids.remove(config.request_id)
                except ValueError:
                    continue
                if self._persistence:
                    self._persistence.save_unassignment(
                        node_id=node.node_id,
                        request_id=config.request_id,
                        reason="rebalance_scale_in",
                    )
                self._persist_lifecycle(
                    config.request_id,
                    "unassigned",
                    self._job_status(config.request_id),
                    reason="rebalance_scale_in",
                    assigned_nodes=[node.node_id],
                )

    # =========================================================================
    # Polling and Configuration Push
    # =========================================================================

    async def poll_node(self, node: NodeInfo, client: httpx.AsyncClient) -> Optional[dict]:
        """Poll a single node for its current metrics."""
        start_time = time.time()
        try:
            response = await client.get(
                f"{node.endpoint}/metrics",
                headers=self._node_api_headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            metrics = response.json()
            self._observe_node_plan_version(metrics)

            # Update node state
            node.last_metrics = metrics
            node.last_seen = datetime.now(timezone.utc)
            node.health_failures = 0
            node.is_active = metrics.get("is_active", True)
            node.latency_ms = (time.time() - start_time) * 1000
            node.last_error = None

            # Track latency per request (multi-job: report to all assigned)
            for rid in node.assigned_request_ids:
                if rid not in self._request_latencies:
                    self._request_latencies[rid] = []
                self._request_latencies[rid].append(node.latency_ms)

            return metrics

        except Exception as e:
            error_msg = str(e)
            node.health_failures += 1
            node.last_error = error_msg
            node.latency_ms = (time.time() - start_time) * 1000

            if node.health_failures >= self.config.max_health_failures:
                node.is_active = False

            # Log error
            if self._persistence:
                self._persistence.save_error(
                    ErrorType.METRICS_FETCH,
                    f"Failed to fetch metrics from node {node.node_id}: {error_msg}",
                    node_id=node.node_id,
                    request_id=node.assigned_request_id,
                    details={
                        "health_failures": node.health_failures,
                        "latency_ms": node.latency_ms,
                    },
                )

            return None

    async def push_config(
        self,
        node: NodeInfo,
        configs: list[EffectiveConfiguration],
        client: httpx.AsyncClient,
        plan_version: int,
    ) -> bool:
        """Push a per-node plan (one or more analytics configs) to a node."""
        payload = {
            "analytics": [cfg.to_dict() for cfg in configs],
            "plan_version": plan_version,
            "leader_term": self._leader_election.leader_term if self._leader_election else 0,
            "source": self._control_plane_node_id,
        }
        try:
            response = await client.post(
                f"{node.endpoint}/configure",
                json=payload,
                headers=self._node_api_headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            return True

        except Exception as e:
            error_msg = str(e)
            req_ids = ",".join(cfg.request_id for cfg in configs)

            if self._persistence:
                self._persistence.save_error(
                    ErrorType.CONFIGURE_PUSH,
                    f"Failed to push config to node {node.node_id}: {error_msg}",
                    node_id=node.node_id,
                    request_id=req_ids,
                )

            return False

    async def _poll_all_nodes(self) -> dict[str, dict]:
        """Poll all registered nodes in parallel."""
        sem = asyncio.Semaphore(max(1, self.config.max_parallel_polls))

        async def _guarded_poll(node: NodeInfo, client: httpx.AsyncClient):
            async with sem:
                return await self.poll_node(node, client)

        async with httpx.AsyncClient() as client:
            tasks = [_guarded_poll(node, client) for node in self._nodes.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            node_id: result
            for node_id, result in zip(self._nodes.keys(), results)
            if result is not None and not isinstance(result, Exception)
        }

    async def _push_configs(self, node_configs: dict[str, list[EffectiveConfiguration]]) -> dict[str, bool]:
        """Push per-node plans in parallel (single push per node per iteration)."""
        if not node_configs:
            return {}

        sem = asyncio.Semaphore(max(1, self.config.max_parallel_pushes))

        async with httpx.AsyncClient() as client:
            nodes_with_plan = [
                (self._nodes[node_id], configs) for node_id, configs in node_configs.items() if node_id in self._nodes
            ]

            async def _guarded_push(node: NodeInfo, configs: list[EffectiveConfiguration]):
                async with sem:
                    return await self.push_config(node, configs, client, self._plan_version)

            tasks = [_guarded_push(node, configs) for node, configs in nodes_with_plan]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            nodes_with_plan[i][0].node_id: result if isinstance(result, bool) else False
            for i, result in enumerate(results)
        }

    # =========================================================================
    # RL Decision Making
    # =========================================================================

    def _apply_hard_safety_guard(
        self,
        request: AnalyticsRequest,
        proposed_action_idx: int,
        node_metrics: dict[str, dict],
    ) -> int:
        """
        Apply hard guardrail before RL action execution.

        If CPU/memory are near request limits, block actions that would
        likely increase pressure and force a safer action.
        """
        capacity = self._capacity_snapshot(request)
        if int(capacity.get("free", 0) or 0) <= 0 and proposed_action_idx in (1, 3, 6):
            return 0 if proposed_action_idx == 1 else 4

        if not node_metrics:
            return proposed_action_idx

        cpu_values: list[float] = []
        mem_values: list[float] = []
        for metrics in node_metrics.values():
            cpu, mem = self._extract_compute_metrics(metrics)
            if cpu > 0:
                cpu_values.append(cpu)
            if mem > 0:
                mem_values.append(mem)

        if not cpu_values and not mem_values:
            return proposed_action_idx

        avg_cpu = (sum(cpu_values) / len(cpu_values)) if cpu_values else 0.0
        avg_mem = (sum(mem_values) / len(mem_values)) if mem_values else 0.0
        cpu_limit = request.resource_limits.cpu_max_percent
        mem_limit = request.resource_limits.memory_max_percent

        near_limit = (cpu_limit > 0 and avg_cpu >= cpu_limit * 0.95) or (mem_limit > 0 and avg_mem >= mem_limit * 0.95)
        if not near_limit:
            return proposed_action_idx

        # 1 increase_coverage, 3 increase_sample, 6 increase_freshness (more frequent = heavier)
        if proposed_action_idx in (1, 3, 6):
            # Prefer reducing sample first, then coverage.
            return 4 if avg_cpu >= avg_mem else 2
        return proposed_action_idx

    def _safety_masked_action_indices(
        self,
        request: AnalyticsRequest,
        available_action_indices: tuple[int, ...],
        node_metrics: dict[str, dict],
    ) -> tuple[int, ...]:
        """Remove actions that the hard guard would rewrite after selection."""
        safe = tuple(
            action_index
            for action_index in available_action_indices
            if self._apply_hard_safety_guard(request, action_index, node_metrics)
            == action_index
        )
        return safe or (0,)

    def _make_rl_decision(
        self,
        request_id: str,
        request: AnalyticsRequest,
        config: EffectiveConfiguration,
        node_metrics: dict[str, dict],
        current_coverage: float,
    ) -> tuple[Optional[EffectiveConfiguration], Optional[RLDecision]]:
        """
        Use RL agent to decide on configuration adjustment.

        Returns:
            Tuple of (new_config, rl_decision) or (None, None) if no change.
        """
        if not self.config.enable_rl:
            # Log no-op if persistence enabled
            if self._persistence:
                self._persistence.save_noop_decision(
                    request_id, self._request_iterations.get(request_id, 0), "RL disabled"
                )
            return None, None

        env = self._rl_environments.get(request_id)
        agent = self._rl_agents.get(request_id)

        if not env or not agent:
            if self._persistence:
                self._persistence.save_noop_decision(
                    request_id, self._request_iterations.get(request_id, 0), "RL not initialized for request"
                )
            return None, None

        # Update environment with current node metrics
        env.update_node_metrics(node_metrics)
        env.update_fairness_debt(self._tenant_debt(request.tenant_id))
        latencies = self._request_latencies.get(request_id, [])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        capacity = self._capacity_snapshot(request)
        env.update_runtime_context(
            active_jobs=int(capacity.get("used", 0) or 0),
            capacity_used=int(capacity.get("used", 0) or 0),
            capacity_free=int(capacity.get("free", 0) or 0),
            pending_requests=self.pending_request_count(),
            recent_slo_violations=self._recent_slo_violation_count(request_id),
            latency_ms=avg_latency,
        )

        # Get current state
        current_observation = env.observe()
        current_state = current_observation.encoded_state

        # Select action
        available_actions = self._safety_masked_action_indices(
            request,
            env.available_action_indices(),
            node_metrics,
        )
        action = agent.select_action(
            current_state,
            training=self.config.rl_training,
            allowed_action_indices=available_actions,
        )
        action_idx = action.index
        action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else f"action_{action_idx}"

        # Real exploration flag from the agent's last decision
        was_exploration = bool(getattr(agent, "last_was_exploration", False))

        # Apply action to get new state
        if action_idx == 0:  # Hold action
            new_config = config
            next_observation = current_observation
        else:
            # Apply the action
            next_observation = env.step_range(action_idx)
            new_config = env.effective_config

        next_state = next_observation.encoded_state
        next_available_actions = self._safety_masked_action_indices(
            request,
            env.available_action_indices(),
            node_metrics,
        )
        reward = next_observation.reward
        reward_details = next_observation.reward_details

        if self.config.rl_training:
            agent.update(
                current_state,
                action,
                reward,
                next_state,
                done=False,
                allowed_action_indices=available_actions,
                next_allowed_action_indices=next_available_actions,
            )
        else:
            agent.step_complete(reward)

        # Get Q-values for logging
        q_values = agent.get_policy(current_state)
        q_values_dict = {
            ACTION_NAMES[a.index] if a.index < len(ACTION_NAMES) else f"action_{a.index}": v
            for a, v in q_values.items()
        }

        # Create RL decision record
        iteration = self._request_iterations.get(request_id, 0)
        rl_decision = RLDecision(
            request_id=request_id,
            timestamp=utc_now_iso(),
            iteration=iteration,
            state={
                "encoded": str(current_state),
                "cpu_utilization": current_observation.observation.state.compute.cpu_utilization,
                "memory_utilization": current_observation.observation.state.compute.memory_utilization,
                "coverage": config.target_coverage,
                "sample": config.target_sample,
                "freshness": config.target_freshness,
                "input_multiplier": config.input_multiplier,
                "service_duty_percent": getattr(env, "_service_duty_percent", 0.0),
                "active_jobs": int(capacity.get("used", 0) or 0),
                "capacity_used": int(capacity.get("used", 0) or 0),
                "capacity_free": int(capacity.get("free", 0) or 0),
                "pending_requests": self.pending_request_count(),
                "latency_ms": avg_latency,
                "recent_slo_violations": self._recent_slo_violation_count(request_id),
            },
            state_hash=str(hash(current_state)),
            action=action_name,
            action_index=action_idx,
            reward=reward,
            reward_components={
                "requirement_quality": reward_details.requirement_quality,
                "resource_penalty": reward_details.resource_penalty,
                "cost_penalty": reward_details.cost_penalty,
                "range_penalty_coverage": reward_details.range_penalty_coverage,
                "range_penalty_sample": reward_details.range_penalty_sample,
                "range_penalty_freshness": reward_details.range_penalty_freshness,
                "resource_overload_penalty": reward_details.resource_overload_penalty,
                "fairness_penalty": reward_details.fairness_penalty,
                "latency_penalty": reward_details.latency_penalty,
                "capacity_penalty": reward_details.capacity_penalty,
            },
            algorithm=agent.get_metrics().get("algorithm", request.algorithm),
            policy_version=agent.get_metrics().get("policy_version", "qlearning-v1"),
            epsilon=float(getattr(agent, "epsilon", agent.get_metrics().get("exploration_rate", 0.0))),
            was_exploration=was_exploration,
            q_values=q_values_dict,
            reason=self._generate_decision_reason(action_name, reward_details),
        )

        # Save decision to persistence
        if self._persistence:
            self._persistence.save_rl_decision(rl_decision)

        if self.config.rl_training:
            agent.decay_exploration()

        # Log decision to CSV for experiment analysis
        if self.config.enable_logging:
            agent_metrics = agent.get_metrics()
            self._logger.log_decision(
                request_id=request_id,
                algorithm=agent_metrics.get("algorithm", request.algorithm),
                policy_version=agent_metrics.get("policy_version", "qlearning-v1"),
                action_type=action_name,
                action_index=action_idx,
                prev_coverage=config.target_coverage,
                prev_sample=config.target_sample,
                prev_freshness=config.target_freshness,
                new_coverage=new_config.target_coverage,
                new_sample=new_config.target_sample,
                new_freshness=new_config.target_freshness,
                reward=reward,
                reward_components=rl_decision.reward_components,
                epsilon=float(getattr(agent, "epsilon", agent.get_metrics().get("exploration_rate", 0.0))),
                was_exploration=was_exploration,
                state_hash=rl_decision.state_hash,
                reason=rl_decision.reason,
            )

            self._logger.log_rl_learning(
                request_id=request_id,
                algorithm=agent_metrics.get("algorithm", request.algorithm),
                policy_version=agent_metrics.get("policy_version", "qlearning-v1"),
                episodes=agent.episode_count,
                steps=agent.step_count,
                total_reward=agent.total_reward,
                epsilon=float(getattr(agent, "epsilon", agent.get_metrics().get("exploration_rate", 0.0))),
                states_visited=int(agent_metrics.get("states_visited", 0)),
                q_values=q_values_dict,
            )

        # Check if config actually changed
        if (
            new_config.target_coverage == config.target_coverage
            and new_config.target_sample == config.target_sample
            and new_config.target_freshness == config.target_freshness
        ):
            return None, rl_decision

        # Recalculate node count with ceiling and set total_nodes for dynamic user assignment
        total_nodes = len([n for n in self._nodes.values() if n.is_active])
        new_config.assigned_node_count = max(1, math.ceil(total_nodes * new_config.target_coverage))
        if request.placement_limits.max_nodes is not None:
            new_config.assigned_node_count = min(
                new_config.assigned_node_count,
                request.placement_limits.max_nodes,
            )
        new_config.total_nodes = len(self._nodes)  # Total registered nodes for data distribution
        new_config.cpu_max_percent = request.resource_limits.cpu_max_percent
        new_config.memory_max_percent = request.resource_limits.memory_max_percent
        new_config.priority = request.priority
        new_config.tenant_id = request.tenant_id
        new_config.placement_limits = request.placement_limits

        return new_config, rl_decision

    def _generate_decision_reason(self, action_name: str, reward_details) -> str:
        """Generate human-readable reason for RL decision."""
        reasons = []

        if reward_details.resource_overload_penalty > 0:
            reasons.append("node resources overloaded")
        if reward_details.resource_penalty > 0.5:
            reasons.append("high resource pressure")
        if reward_details.cost_penalty > 0.3:
            reasons.append("cost concerns")
        if reward_details.range_penalty_coverage > 0:
            reasons.append("coverage outside user range")
        if reward_details.range_penalty_sample > 0:
            reasons.append("sample outside user range")
        if reward_details.range_penalty_freshness > 0:
            reasons.append("freshness outside user range")
        if getattr(reward_details, "fairness_penalty", 0) > 0:
            reasons.append("tenant fairness debt")
        if getattr(reward_details, "latency_penalty", 0) > 0:
            reasons.append("response-time pressure")
        if getattr(reward_details, "capacity_penalty", 0) > 0:
            reasons.append("placement capacity exhausted")
        if reward_details.requirement_quality < 0.5:
            reasons.append("low requirement quality")

        if not reasons:
            reasons.append("optimizing performance")

        return f"{action_name}: {', '.join(reasons)}"

    # =========================================================================
    # Main Evaluation Loop
    # =========================================================================

    def _evaluate_and_adjust(self, all_metrics: dict[str, dict]) -> tuple[dict[str, list[EffectiveConfiguration]], int]:
        """
        Evaluate current state and decide on adjustments.

        Uses the RL agent for each request to make decisions.

        Returns:
            Tuple of (node_plans, actual rl decisions count).
        """
        node_plans: dict[str, list[EffectiveConfiguration]] = {}
        rl_decisions_count = 0
        tenant_debt_acc: dict[str, list[float]] = {}

        ordered_requests = sorted(
            self._active_requests.items(),
            key=lambda kv: (
                self._priority_rank(kv[1].priority),
                -self._tenant_debt(kv[1].tenant_id),
                kv[0],
            ),
        )

        for request_id, request in ordered_requests:
            if self._job_status(request_id) not in (JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.DEGRADED):
                continue
            config = self._effective_configs.get(request_id)
            if not config:
                continue
            if self._job_status(request_id) == JobStatus.ACCEPTED:
                self._set_job_status(request_id, JobStatus.RUNNING, "orchestration_iteration")
                self._persist_lifecycle(request_id, "running", JobStatus.RUNNING, reason="orchestration_iteration")

            # Get nodes assigned to this request (multi-job aware)
            assigned_nodes = self._request_assigned_nodes(request_id)

            # Calculate current coverage
            active_assigned = len([n for n in assigned_nodes if n.is_active])
            current_coverage = active_assigned / max(self._active_node_count(), 1)
            coverage_debt = max(0.0, request.coverage_range[0] - current_coverage)
            tenant_debt_acc.setdefault(request.tenant_id, []).append(coverage_debt)

            # Increment iteration counter for this request
            if request_id not in self._request_iterations:
                self._request_iterations[request_id] = 0
            self._request_iterations[request_id] += 1
            iteration = self._request_iterations[request_id]

            # Scope node metrics to this request's assigned nodes only.
            request_node_metrics = {
                node.node_id: all_metrics[node.node_id] for node in assigned_nodes if node.node_id in all_metrics
            }

            # Use RL agent if enabled, otherwise use callback
            new_config = None
            rl_decision = None

            if self.config.enable_rl:
                new_config, rl_decision = self._make_rl_decision(
                    request_id, request, config, request_node_metrics, current_coverage
                )
                if rl_decision:
                    rl_decisions_count += 1
            elif self._decision_callback:
                # Optional external decision callback
                new_config = self._decision_callback(
                    request=request,
                    current_config=config,
                    node_metrics=request_node_metrics,
                    current_coverage=current_coverage,
                )

            # Update config if changed
            if new_config and new_config != config:
                # Save configuration change
                if self._persistence:
                    self._persistence.save_config_change(
                        ConfigurationChange(
                            request_id=request_id,
                            timestamp=utc_now_iso(),
                            iteration=iteration,
                            prev_coverage=config.target_coverage,
                            prev_sample=config.target_sample,
                            prev_freshness=config.target_freshness,
                            prev_node_count=config.assigned_node_count,
                            new_coverage=new_config.target_coverage,
                            new_sample=new_config.target_sample,
                            new_freshness=new_config.target_freshness,
                            new_node_count=new_config.assigned_node_count,
                            reason="rl_decision" if rl_decision else "callback",
                            rl_action=rl_decision.action if rl_decision else None,
                        )
                    )

                self._effective_configs[request_id] = new_config
                config = new_config

            # Rebalance assignment when target coverage changes.
            self._rebalance_request_assignments(request, config)
            assigned_nodes = self._request_assigned_nodes(request_id)
            active_assigned = len([n for n in assigned_nodes if n.is_active])
            required_nodes = self._minimum_node_count(request)
            if active_assigned < required_nodes:
                for node in assigned_nodes:
                    with contextlib.suppress(ValueError):
                        node.assigned_request_ids.remove(request_id)
                    if self._persistence:
                        self._persistence.save_unassignment(
                            node_id=node.node_id,
                            request_id=request_id,
                            reason="runtime_capacity_below_minimum",
                        )
                    self._persist_lifecycle(
                        request_id,
                        "unassigned",
                        JobStatus.QUEUED,
                        reason="runtime_capacity_below_minimum",
                        assigned_nodes=[node.node_id],
                    )
                self._mark_queued(request, "runtime_capacity_below_minimum")
                self._record_slo_decision_epoch(request_id, True)
                continue
            current_coverage = active_assigned / max(self._active_node_count(), 1)
            epoch_has_violation = self._persist_coverage_violation(
                request=request,
                config=config,
                current_coverage=current_coverage,
                active_assigned=active_assigned,
                reason="rebalance_capacity",
            )
            latencies = self._request_latencies.get(request_id, [])
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            total_data_volume = 0
            total_processing_time = 0.0
            total_hotspots = 0
            responding_count = 0
            observed_samples: list[float] = []
            observed_freshness_ages: list[float] = []
            observed_spatial_fidelity: list[float] = []
            observed_hotspot_recall: list[float] = []

            for node in assigned_nodes:
                if node.node_id not in all_metrics:
                    continue
                metrics = all_metrics[node.node_id]
                analytics_metrics = self._extract_request_analytics_metrics(metrics, request_id)
                if analytics_metrics and analytics_metrics.get("output_observed"):
                    data_volume_bytes = int(analytics_metrics.get("data_volume_bytes") or 0.0)
                    processing_ms = float(analytics_metrics.get("processing_time_ms") or 0.0)
                    total_data_volume += data_volume_bytes
                    sample_rate = analytics_metrics.get("sample_rate")
                    freshness_age_s = analytics_metrics.get("freshness_age_s")
                    hotspots = int(analytics_metrics.get("hotspots") or 0.0)
                    spatial_fidelity = analytics_metrics.get("spatial_fidelity")
                    hotspot_recall = analytics_metrics.get("hotspot_recall")
                    if sample_rate is not None:
                        observed_samples.append(float(sample_rate))
                    if freshness_age_s is not None:
                        observed_freshness_ages.append(float(freshness_age_s))
                    if spatial_fidelity is not None:
                        observed_spatial_fidelity.append(float(spatial_fidelity))
                    if hotspot_recall is not None:
                        observed_hotspot_recall.append(float(hotspot_recall))
                    total_hotspots += hotspots
                    total_processing_time += processing_ms
                    responding_count += 1

            actual_sample = canonical_contract_mean(observed_samples, request.sample_range)
            observed_freshness_age_max_s = max(observed_freshness_ages) if observed_freshness_ages else None
            mean_spatial_fidelity = (
                sum(observed_spatial_fidelity) / len(observed_spatial_fidelity)
                if observed_spatial_fidelity
                else None
            )
            mean_hotspot_recall = (
                sum(observed_hotspot_recall) / len(observed_hotspot_recall)
                if observed_hotspot_recall
                else None
            )
            avg_processing_time_ms = total_processing_time / responding_count if responding_count > 0 else None
            mean_service_duty_percent = (
                avg_processing_time_ms / (max(config.target_freshness, 1e-6) * 10.0)
                if avg_processing_time_ms is not None
                else None
            )

            epoch_has_violation = self._persist_observed_quality_violations(
                request=request,
                actual_sample=actual_sample,
                freshness_age_s=observed_freshness_age_max_s,
            ) or epoch_has_violation

            cost_estimate = estimate_request_cost(
                assigned_node_count=config.assigned_node_count,
                responding_node_count=responding_count,
                actual_coverage=current_coverage,
                actual_sample=actual_sample if actual_sample is not None else config.target_sample,
                actual_freshness_s=(
                    observed_freshness_age_max_s
                    if observed_freshness_age_max_s is not None
                    else config.target_freshness
                ),
                total_processing_time_ms=total_processing_time,
                total_data_volume_bytes=int(total_data_volume),
                cost_budget=request.cost_budget,
                rates=self.config.cost_model_rates,
            )

            resource_breaches: list[tuple[NodeInfo, str, float, float]] = []
            for node in assigned_nodes:
                if node.node_id not in all_metrics:
                    continue
                cpu, mem = self._extract_compute_metrics(all_metrics[node.node_id])
                if breaches_upper_bound(cpu, request.resource_limits.cpu_max_percent):
                    resource_breaches.append(
                        (node, "cpu_percent", cpu, request.resource_limits.cpu_max_percent)
                    )
                if breaches_upper_bound(mem, request.resource_limits.memory_max_percent):
                    resource_breaches.append(
                        (node, "memory_percent", mem, request.resource_limits.memory_max_percent)
                    )
            epoch_has_violation = bool(resource_breaches) or epoch_has_violation

            # Save per-request metrics
            if self._persistence:
                for node, metric_name, measured, limit in resource_breaches:
                    self._persistence.save_slo_violation(
                        SLOViolation(
                            request_id=request_id,
                            tenant_id=request.tenant_id,
                            node_id=node.node_id,
                            timestamp=utc_now_iso(),
                            metric=metric_name,
                            measured_value=measured,
                            limit_value=limit,
                            severity="critical" if measured > limit + 10 else "warning",
                        )
                    )

                self._persistence.save_request_metrics(
                    RequestMetrics(
                        request_id=request_id,
                        timestamp=utc_now_iso(),
                        iteration=iteration,
                        target_coverage=config.target_coverage,
                        target_sample=config.target_sample,
                        target_freshness=config.target_freshness,
                        actual_coverage=current_coverage,
                        actual_sample=actual_sample,
                        actual_freshness=observed_freshness_age_max_s,
                        observed_freshness_age_max_s=observed_freshness_age_max_s,
                        assigned_node_count=config.assigned_node_count,
                        active_node_count=active_assigned,
                        responding_node_count=responding_count,
                        avg_latency_ms=avg_latency or None,
                        max_latency_ms=max_latency or None,
                        avg_processing_time_ms=avg_processing_time_ms,
                        estimated_cost_units=cost_estimate.score_units,
                        estimated_cost_usd=cost_estimate.estimated_usd,
                        cost_cpu_usd=cost_estimate.cpu_cost_usd,
                        cost_memory_usd=cost_estimate.memory_cost_usd,
                        cost_network_usd=cost_estimate.network_cost_usd,
                        cost_storage_usd=cost_estimate.storage_cost_usd,
                        cost_budget=request.cost_budget,
                        cost_budget_ratio=cost_estimate.budget_ratio,
                        total_data_volume_bytes=int(total_data_volume),
                        total_processing_time_ms=total_processing_time,
                        total_hotspots=total_hotspots,
                        input_multiplier=config.input_multiplier,
                        mean_service_duty_percent=mean_service_duty_percent,
                        mean_spatial_fidelity=mean_spatial_fidelity,
                        mean_hotspot_recall=mean_hotspot_recall,
                        slo_coverage_min=request.coverage_range[0],
                        slo_coverage_max=request.coverage_range[1],
                        slo_sample_min=request.sample_range[0],
                        slo_sample_max=request.sample_range[1],
                        slo_freshness_min=request.freshness_range[0],
                        slo_freshness_max=request.freshness_range[1],
                    )
                )

            self._record_slo_decision_epoch(request_id, epoch_has_violation)
            self._request_latencies[request_id] = []

            if self.config.enable_logging:
                self._logger.log_request_metrics(
                    request_id=request_id,
                    service_type=request.service_type,
                    algorithm=(
                        self._rl_agents.get(request_id).get_metrics().get("algorithm", request.algorithm)
                        if self._rl_agents.get(request_id)
                        else request.algorithm
                    ),
                    policy_version=(
                        self._rl_agents.get(request_id).get_metrics().get("policy_version", "qlearning-v1")
                        if self._rl_agents.get(request_id)
                        else "qlearning-v1"
                    ),
                    target_coverage=config.target_coverage,
                    target_sample=config.target_sample,
                    target_freshness=config.target_freshness,
                    actual_coverage=current_coverage,
                    assigned_nodes=config.assigned_node_count,
                    active_nodes=active_assigned,
                    responding_nodes=responding_count,
                    avg_latency_ms=avg_latency,
                    max_latency_ms=max_latency,
                    actual_sample=actual_sample,
                    actual_freshness=observed_freshness_age_max_s,
                    observed_freshness_age_max_s=observed_freshness_age_max_s,
                    avg_processing_time_ms=avg_processing_time_ms or 0.0,
                    estimated_cost_units=cost_estimate.score_units,
                    estimated_cost_usd=cost_estimate.estimated_usd,
                    cost_cpu_usd=cost_estimate.cpu_cost_usd,
                    cost_memory_usd=cost_estimate.memory_cost_usd,
                    cost_network_usd=cost_estimate.network_cost_usd,
                    cost_storage_usd=cost_estimate.storage_cost_usd,
                    cost_budget=request.cost_budget or 0.0,
                    cost_budget_ratio=cost_estimate.budget_ratio or 0.0,
                    total_data_volume_bytes=int(total_data_volume),
                    total_processing_time_ms=total_processing_time,
                    total_hotspots=total_hotspots,
                    input_multiplier=config.input_multiplier,
                    mean_service_duty_percent=mean_service_duty_percent,
                    mean_spatial_fidelity=mean_spatial_fidelity,
                    mean_hotspot_recall=mean_hotspot_recall,
                )

            # Queue config pushes per node (single plan per node).
            for node in assigned_nodes:
                if node.is_active:
                    node_plans.setdefault(node.node_id, []).append(config)

        # Update and persist tenant fairness debt.
        for tenant_id, debts in tenant_debt_acc.items():
            debt = sum(debts) / len(debts) if debts else 0.0
            self._tenant_fairness_debt[tenant_id] = debt
            if self._persistence:
                active_requests = len(
                    [
                        r
                        for r in self._active_requests.values()
                        if r.tenant_id == tenant_id
                        and self._job_status(r.request_id)
                        in (JobStatus.ACCEPTED, JobStatus.RUNNING, JobStatus.DEGRADED)
                    ]
                )
                self._persistence.save_tenant_fairness(
                    TenantFairnessRecord(
                        timestamp=utc_now_iso(),
                        tenant_id=tenant_id,
                        fairness_debt=debt,
                        active_requests=active_requests,
                    )
                )

        return node_plans, rl_decisions_count

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run_once(self) -> IterationMetrics:
        """Run a single iteration of the orchestration loop."""
        iteration_start = time.time()
        self._iteration_count += 1
        errors_count = 0

        # 0. Refresh leadership before taking actions.
        await self._control_plane_tick()

        # 1. Poll all nodes
        all_metrics = await self._poll_all_nodes()

        # Count errors from failed polls
        for node in self._nodes.values():
            if node.last_error:
                errors_count += 1

        node_plans: dict[str, list[EffectiveConfiguration]] = {}
        rl_decisions_count = 0

        # Enforce iteration budget to avoid control-loop backlog.
        elapsed_ms = (time.time() - iteration_start) * 1000
        if self.config.iteration_budget_ms > 0 and elapsed_ms >= self.config.iteration_budget_ms:
            errors_count += 1
            if self._persistence:
                self._persistence.save_error(
                    ErrorType.INTERNAL,
                    (
                        f"Iteration budget exceeded before decision phase: "
                        f"{elapsed_ms:.1f}ms >= {self.config.iteration_budget_ms:.1f}ms"
                    ),
                )
        elif not self._is_leader:
            # Followers observe but do not mutate control-plane state.
            node_plans = {}
            rl_decisions_count = 0
        else:
            # 2. Admit queued requests before evaluating active workloads.
            self._unassign_inactive_node_workloads()
            self._retry_pending_requests()

            # 3. Evaluate and decide on adjustments (includes RL decisions)
            node_plans, rl_decisions_count = self._evaluate_and_adjust(all_metrics)

        # 4. Push configurations
        push_results = {}
        elapsed_ms = (time.time() - iteration_start) * 1000
        if node_plans and self.config.iteration_budget_ms > 0 and elapsed_ms >= self.config.iteration_budget_ms:
            errors_count += 1
            if self._persistence:
                self._persistence.save_error(
                    ErrorType.INTERNAL,
                    (
                        f"Iteration budget exceeded before push phase: "
                        f"{elapsed_ms:.1f}ms >= {self.config.iteration_budget_ms:.1f}ms"
                    ),
                )
        elif node_plans:
            self._plan_version += 1
            if self._leader_election:
                self._leader_election.register_plan_version(
                    self._plan_version,
                    source_node=self._control_plane_node_id,
                )
            push_results = await self._push_configs(node_plans)
            # Count push failures
            errors_count += sum(1 for success in push_results.values() if not success)
            if self._persistence:
                for node_id, configs in node_plans.items():
                    self._persistence.save_node_plan_applied(
                        NodePlanApplied(
                            timestamp=utc_now_iso(),
                            iteration=self._iteration_count,
                            node_id=node_id,
                            plan_version=self._plan_version,
                            leader_term=self._leader_election.leader_term if self._leader_election else 0,
                            request_ids=[cfg.request_id for cfg in configs],
                            accepted_count=len(configs),
                            rejected_count=0,
                        )
                    )

        # 4. Calculate summary metrics
        total_nodes = len(self._nodes)
        active_nodes = len([n for n in self._nodes.values() if n.is_active])
        responding_nodes = len(all_metrics)

        iteration_time = time.time() - iteration_start

        # 5. Log for experiment analysis
        if self.config.enable_logging:
            self._logger.log_iteration(
                timestamp=datetime.now(timezone.utc),
                total_nodes=total_nodes,
                active_nodes=active_nodes,
                responding_nodes=responding_nodes,
                configs_pushed=len(node_plans),
                iteration_time_ms=iteration_time * 1000,
                node_metrics=all_metrics,
                active_requests=self.running_request_count(),
                rl_decisions_made=rl_decisions_count,
            )

        return IterationMetrics(
            iteration=self._iteration_count,
            timestamp=utc_now_iso(),
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            responding_nodes=responding_nodes,
            configs_pushed=len(node_plans),
            iteration_time_ms=iteration_time * 1000,
            requests_processed=self.running_request_count(),
            rl_decisions_made=rl_decisions_count,
            errors_count=errors_count,
        )

    async def run(self) -> None:
        """Run the orchestration loop continuously."""
        self._running = True

        # Requests transition to RUNNING inside run_once after admission.

        try:
            while self._running:
                metrics = await self.run_once()
                loop_time_s = metrics.iteration_time_ms / 1000.0
                sleep_s = max(0.0, self.config.poll_interval_seconds - loop_time_s)
                await asyncio.sleep(sleep_s)
        finally:
            await self.close_async()

    def stop(self) -> None:
        """Stop the orchestration loop and auto-save RL agents."""
        self._running = False

        self.complete_rl_trajectories()

        # Auto-save RL agents so no learned data is lost
        if self._rl_agents:
            try:
                from argos.common.constants import RL_AGENTS_DIR

                self.save_rl_agents(str(RL_AGENTS_DIR))
                logger.info("RL agents auto-saved to %s", RL_AGENTS_DIR)
            except Exception as e:
                logger.error("Failed to auto-save RL agents: %s", e)

        # Best-effort async cleanup for control-plane bus.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close_async())
        except RuntimeError:
            pass

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def nodes(self) -> dict[str, NodeInfo]:
        """Get all registered nodes."""
        return self._nodes.copy()

    @property
    def active_requests(self) -> dict[str, AnalyticsRequest]:
        """Get all active analytics requests."""
        return self._active_requests.copy()

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running

    def get_rl_metrics(self) -> dict[str, dict[str, Any]]:
        """Get RL agent metrics for all requests."""
        result: dict[str, dict[str, Any]] = {}
        for request_id, agent in self._rl_agents.items():
            metrics = agent.get_metrics()
            expected = self._frozen_policy_fingerprints.get(request_id)
            metrics["frozen_policy_loaded"] = expected is not None
            metrics["policy_unchanged"] = expected is None or agent.policy_fingerprint() == expected
            result[request_id] = metrics
        return result

    def get_rl_policy_fingerprints(self) -> dict[str, str]:
        """Return stable fingerprints for the active decision policies."""
        return {request_id: agent.policy_fingerprint() for request_id, agent in self._rl_agents.items()}

    def set_request_input_multiplier(self, request_id: str, multiplier: int) -> None:
        """Apply one declared exogenous workload level to an active request."""
        if request_id not in self._active_requests or request_id not in self._effective_configs:
            raise KeyError(f"Unknown active request: {request_id}")
        value = max(1, min(100, int(multiplier)))
        self._active_requests[request_id].input_multiplier = value
        self._effective_configs[request_id].input_multiplier = value
        environment = self._rl_environments.get(request_id)
        if environment:
            environment.set_input_multiplier(value)

    def complete_rl_trajectories(self) -> None:
        """Flush pending training rollouts before policy persistence."""
        if not self.config.rl_training:
            return
        for request_id, agent in self._rl_agents.items():
            environment = self._rl_environments.get(request_id)
            final_state = environment.observe().encoded_state if environment else None
            agent.end_trajectory(final_state=final_state, done=True)

    def _try_load_policy(self, agent: RLAgent) -> None:
        """Warm-start an agent from the newest saved artifact of the same algorithm."""
        from pathlib import Path

        try:
            from argos.common.constants import RL_AGENTS_DIR

            agents_dir = Path(str(RL_AGENTS_DIR))
            latest = find_latest_policy(
                str(agents_dir),
                agent.get_metrics().get("algorithm", "qlearning"),
                agent.policy_extension,
            )
            if not latest:
                return
            agent.load(str(latest))
            if hasattr(agent, "set_epsilon"):
                agent.set_epsilon(self.config.rl_epsilon)
            logger.info(
                "Warm-started %s agent from %s (epsilon reset to %.4f)",
                agent.get_metrics().get("algorithm", "qlearning"),
                latest.name,
                self.config.rl_epsilon,
            )
        except Exception as e:
            logger.debug("No saved policy to warm-start from: %s", e)

    def save_rl_agents(self, path: str) -> None:
        """Save all RL agents to disk."""
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        for request_id, agent in self._rl_agents.items():
            algorithm = agent.get_metrics().get("algorithm", "qlearning")
            filename = f"agent_{request_id}_{algorithm}{agent.policy_extension}"
            agent.save(str(save_dir / filename))

    def load_rl_agent(self, request_id: str, path: str) -> Optional[str]:
        """Load a policy now or defer it until a queued request is admitted."""
        if request_id not in self._active_requests:
            raise KeyError(f"Unknown request for policy load: {request_id}")
        policy_path = str(path)
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(policy_path)
        self._pending_policy_loads[request_id] = policy_path
        if request_id in self._rl_agents:
            return self._apply_pending_policy_load(request_id)
        return None

    def assert_rl_policy_unchanged(self, request_id: str) -> None:
        """Fail if a requested frozen policy is pending or has mutated."""
        if request_id in self._pending_policy_loads:
            if self._job_status(request_id) == JobStatus.QUEUED:
                return
            raise RuntimeError(f"Frozen policy was never applied for queued request {request_id}")
        expected = self._frozen_policy_fingerprints.get(request_id)
        if expected is None:
            return
        agent = self._rl_agents.get(request_id)
        if agent is None:
            raise RuntimeError(f"Frozen policy agent missing for request {request_id}")
        observed = agent.policy_fingerprint()
        if observed != expected:
            raise RuntimeError(
                f"Frozen policy mutated for request {request_id}: "
                f"expected {expected}, observed {observed}"
            )

    def reset_rl_run_metrics(self, request_id: str) -> None:
        """Start a fresh accounting window without changing the loaded policy."""
        if request_id in self._rl_agents:
            self._rl_agents[request_id].reset_run_metrics()
