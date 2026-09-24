# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Persistence layer for orchestrator state and history.

This module provides comprehensive persistence for:
- User requests with full payload (ranges, service_type, cost_budget)
- Effective configurations and history of changes
- Node assignments and health tracking
- Per-request metrics (coverage, sample, freshness, latency)
- RL decisions (action, state, reward, reason, parameters)
- Errors from /metrics, /configure, and node failures

Data is stored in JSONL format for easy streaming and analysis.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from argos.common.constants import PERSISTENCE_DIR

# =============================================================================
# Enums and Constants
# =============================================================================


class JobStatus(str, Enum):
    """Status of an analytics job."""

    PENDING = "pending"
    QUEUED = "queued"
    ACCEPTED = "accepted"
    RUNNING = "running"
    DEGRADED = "degraded"
    REJECTED = "rejected"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FINISHED = "finished"
    FAILED = "failed"


class ErrorType(str, Enum):
    """Types of errors that can occur."""

    METRICS_FETCH = "metrics_fetch"
    CONFIGURE_PUSH = "configure_push"
    NODE_HEALTH = "node_health"
    RL_DECISION = "rl_decision"
    VALIDATION = "validation"
    INTERNAL = "internal"


# =============================================================================
# Data Classes for Persistence
# =============================================================================


@dataclass
class PersistedRequest:
    """
    Full representation of a user analytics request.

    Includes all input parameters and tracking metadata.
    """

    request_id: str
    service_type: str
    coverage_min: float
    coverage_max: float
    sample_min: float
    sample_max: float
    freshness_min: float
    freshness_max: float
    cost_budget: Optional[float]
    status: JobStatus
    created_at: str
    updated_at: str
    finished_at: Optional[str] = None
    cancellation_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PersistedRequest:
        """Create from dictionary."""
        data = data.copy()
        data["status"] = JobStatus(data["status"])
        return cls(**data)


@dataclass
class ConfigurationChange:
    """
    Record of a configuration change for a request.

    Tracks before/after values for audit and analysis.
    """

    request_id: str
    timestamp: str
    iteration: int
    # Previous values (None if first config)
    prev_coverage: Optional[float]
    prev_sample: Optional[float]
    prev_freshness: Optional[float]
    prev_node_count: Optional[int]
    # New values
    new_coverage: float
    new_sample: float
    new_freshness: float
    new_node_count: int
    # Context
    reason: str  # "initial", "rl_decision", "manual", "rebalance"
    rl_action: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class NodeAssignment:
    """
    Record of node-to-request assignment.
    """

    node_id: str
    request_id: str
    assigned_at: str
    unassigned_at: Optional[str] = None
    reason: str = "initial"  # "initial", "rebalance", "failure", "cancelled"
    node_health_at_assignment: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RequestLifecycleEvent:
    """Append-only request lifecycle/audit event."""

    request_id: str
    timestamp: str
    event_type: str
    status: str
    reason: str = ""
    assigned_nodes: list[str] = field(default_factory=list)
    pending_position: Optional[int] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestMetrics:
    """
    Per-request metrics snapshot.

    Captures actual achieved values vs. configured targets.
    """

    request_id: str
    timestamp: str
    iteration: int
    # Configured values
    target_coverage: float
    target_sample: float
    target_freshness: float
    # Actual achieved values
    actual_coverage: float
    actual_sample: Optional[float]
    actual_freshness: Optional[float]
    # Performance metrics
    assigned_node_count: int
    active_node_count: int
    responding_node_count: int
    avg_latency_ms: Optional[float]
    max_latency_ms: Optional[float]
    observed_freshness_age_max_s: Optional[float] = None
    avg_processing_time_ms: Optional[float] = None
    estimated_cost_units: float = 0.0
    estimated_cost_usd: Optional[float] = None
    cost_cpu_usd: Optional[float] = None
    cost_memory_usd: Optional[float] = None
    cost_network_usd: Optional[float] = None
    cost_storage_usd: Optional[float] = None
    cost_budget: Optional[float] = None
    cost_budget_ratio: Optional[float] = None
    # Service metrics (aggregated across nodes)
    total_data_volume_bytes: int = 0
    total_processing_time_ms: float = 0.0
    total_hotspots: int = 0
    input_multiplier: int = 1
    mean_service_duty_percent: Optional[float] = None
    mean_spatial_fidelity: Optional[float] = None
    mean_hotspot_recall: Optional[float] = None
    # SLO ranges from the original AnalyticsRequest
    slo_coverage_min: Optional[float] = None
    slo_coverage_max: Optional[float] = None
    slo_sample_min: Optional[float] = None
    slo_sample_max: Optional[float] = None
    slo_freshness_min: Optional[float] = None
    slo_freshness_max: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RLDecision:
    """
    Record of an RL agent decision.

    Provides full transparency into RL learning process.
    """

    request_id: str
    timestamp: str
    iteration: int
    # State representation
    state: dict[str, Any]
    state_hash: str  # For Q-table lookup
    # Action taken
    action: str  # "hold", "increase_coverage", etc.
    action_index: int
    # Outcome
    reward: float
    reward_components: dict[str, float]  # Breakdown of reward
    # Context
    epsilon: float  # Current exploration rate
    was_exploration: bool  # True if random action
    algorithm: str = "qlearning"
    policy_version: str = "qlearning-v1"
    q_values: Optional[dict[str, float]] = None  # Q-values for all actions
    reason: str = ""  # Human-readable explanation

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SLOViolation:
    """Record of a per-request resource/SLO violation."""

    request_id: str
    tenant_id: str
    node_id: str
    timestamp: str
    metric: str
    measured_value: float
    limit_value: float
    severity: str  # warning | critical
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SLODecisionEpoch:
    """Binary SLO outcome for one completed request decision epoch."""

    request_id: str
    timestamp: str
    iteration: int
    violated: bool
    recent_violation_count: int
    window_size: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TenantFairnessRecord:
    """Periodic fairness snapshot per tenant."""

    timestamp: str
    tenant_id: str
    fairness_debt: float
    active_requests: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LeaderElectionEvent:
    """Leader-election and failover audit events."""

    timestamp: str
    node_id: str
    event_type: str  # elected | heartbeat | lost | failover
    score: Optional[float] = None
    term: int = 0
    previous_leader: Optional[str] = None
    reason: Optional[str] = None
    source_node: Optional[str] = None
    plan_version: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NodePlanApplied:
    """Applied per-node runtime plan for one orchestration iteration."""

    timestamp: str
    iteration: int
    node_id: str
    plan_version: int
    leader_term: int
    request_ids: list[str]
    accepted_count: int
    rejected_count: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorRecord:
    """
    Record of an error occurrence.
    """

    error_id: str
    timestamp: str
    error_type: ErrorType
    request_id: Optional[str]
    node_id: Optional[str]
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["error_type"] = self.error_type.value
        return d


# =============================================================================
# Persistence Manager
# =============================================================================


@dataclass
class PersistenceConfig:
    """Configuration for persistence layer."""

    output_dir: Optional[Path] = None
    session_id: Optional[str] = None
    enable_requests: bool = True
    enable_configs: bool = True
    enable_assignments: bool = True
    enable_request_lifecycle: bool = True
    enable_metrics: bool = True
    enable_rl_decisions: bool = True
    enable_errors: bool = True
    enable_slo_violations: bool = True
    enable_tenant_fairness: bool = True
    enable_leader_events: bool = True
    enable_node_plans: bool = True
    flush_immediately: bool = True


class PersistenceManager:
    """
    Manages persistence of all orchestrator state and history.

    Uses JSONL files for efficient append-only storage:
    - requests.jsonl: User request history
    - config_changes.jsonl: Configuration change history
    - node_assignments.jsonl: Node assignment history
    - request_metrics.jsonl: Per-request metrics over time
    - rl_decisions.jsonl: RL agent decision history
    - errors.jsonl: Error log

    Thread-safe for concurrent writes.
    """

    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or PersistenceConfig()
        self._lock = Lock()
        self._initialized = False

        # In-memory indices for quick lookups
        self._requests: dict[str, PersistedRequest] = {}
        self._active_assignments: dict[str, set[str]] = {}  # node_id -> set(request_id)

    def initialize(self) -> None:
        """Initialize persistence layer and create output directory.

        An explicit ``output_dir`` is always honored. Otherwise ``session_id``
        selects ``data/sessions/<session_id>/persistence/`` and the final
        fallback is ``data/persistence/``.
        """
        if self._initialized:
            return

        if self.config.output_dir is None and self.config.session_id:
            from argos.common.constants import SESSIONS_DIR

            self.config.output_dir = SESSIONS_DIR / self.config.session_id / "persistence"
        elif self.config.output_dir is None:
            self.config.output_dir = PERSISTENCE_DIR

        self.config.output_dir = Path(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        streams = {
            "requests": self.config.enable_requests,
            "config_changes": self.config.enable_configs,
            "node_assignments": self.config.enable_assignments,
            "request_lifecycle": self.config.enable_request_lifecycle,
            "request_metrics": self.config.enable_metrics,
            "rl_decisions": self.config.enable_rl_decisions,
            "errors": self.config.enable_errors,
            "slo_violations": self.config.enable_slo_violations,
            "slo_decision_epochs": self.config.enable_slo_violations,
            "tenant_fairness": self.config.enable_tenant_fairness,
            "leader_election_events": self.config.enable_leader_events,
            "node_plan_applied": self.config.enable_node_plans,
        }
        for name, enabled in streams.items():
            if enabled:
                (self.config.output_dir / f"{name}.jsonl").touch(exist_ok=True)
        self._initialized = True

    def _get_file(self, name: str) -> Path:
        """Get path to a persistence file."""
        return self.config.output_dir / f"{name}.jsonl"

    def _append_record(self, filename: str, record: dict) -> None:
        """Append a record to a JSONL file."""
        if not self._initialized:
            self.initialize()

        if self.config.session_id:
            record["session_id"] = self.config.session_id

        filepath = self._get_file(filename)
        # Defensive: ensure parent dir exists (open 'a' creates file, NOT dirs)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.config.flush_immediately:
                f.flush()

    def _read_jsonl(
        self,
        filename: str,
        *,
        filter_key: Optional[str] = None,
        filter_value: Optional[str] = None,
        limit: int = 0,
    ) -> list[dict]:
        """Read records from a JSONL file with optional filtering.

        Args:
            filename: Base name of the JSONL file (without extension).
            filter_key: Only include records where this key matches *filter_value*.
            filter_value: Value to match for *filter_key*.
            limit: Maximum records to return (0 = unlimited).

        Returns:
            List of parsed dicts.
        """
        filepath = self._get_file(filename)
        if not filepath.exists():
            return []

        records: list[dict] = []
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if filter_key and data.get(filter_key) != filter_value:
                        continue
                    records.append(data)
        except FileNotFoundError:
            return []
        return records[-limit:] if limit > 0 else records

    # =========================================================================
    # Request Persistence
    # =========================================================================

    def save_request(self, request: PersistedRequest) -> None:
        """Save or update a user request."""
        if not self.config.enable_requests:
            return

        self._requests[request.request_id] = request
        self._append_record("requests", request.to_dict())

    def update_request_status(self, request_id: str, status: JobStatus, reason: Optional[str] = None) -> None:
        """Update the status of a request."""
        if request_id not in self._requests:
            return

        request = self._requests[request_id]
        request.status = status
        request.updated_at = datetime.now(timezone.utc).isoformat()

        if status == JobStatus.CANCELLED:
            request.cancellation_reason = reason
        elif status in (JobStatus.FINISHED, JobStatus.FAILED):
            request.finished_at = datetime.now(timezone.utc).isoformat()

        self.save_request(request)

    # =========================================================================
    # Configuration History
    # =========================================================================

    def save_config_change(self, change: ConfigurationChange) -> None:
        """Save a configuration change record."""
        if not self.config.enable_configs:
            return

        self._append_record("config_changes", change.to_dict())

    # =========================================================================
    # Node Assignments
    # =========================================================================

    def save_assignment(self, assignment: NodeAssignment) -> None:
        """Save a node assignment."""
        if not self.config.enable_assignments:
            return

        if assignment.request_id:
            self._active_assignments.setdefault(assignment.node_id, set()).add(assignment.request_id)
        self._append_record("node_assignments", assignment.to_dict())

    def save_unassignment(self, node_id: str, request_id: str, reason: str = "unassigned") -> None:
        """Record when a node is unassigned from a request."""
        if not self.config.enable_assignments:
            return

        assignment = NodeAssignment(
            node_id=node_id,
            request_id=request_id,
            assigned_at="",  # Will be filled from history
            unassigned_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        if node_id in self._active_assignments:
            self._active_assignments[node_id].discard(request_id)
            if not self._active_assignments[node_id]:
                self._active_assignments.pop(node_id, None)
        self._append_record("node_assignments", assignment.to_dict())

    def save_request_lifecycle(self, event: RequestLifecycleEvent) -> None:
        """Save an append-only request lifecycle event."""
        if not self.config.enable_request_lifecycle:
            return
        self._append_record("request_lifecycle", event.to_dict())

    def get_request_lifecycle(
        self,
        request_id: Optional[str] = None,
        limit: int = 0,
    ) -> list[RequestLifecycleEvent]:
        """Load request lifecycle events in chronological order."""
        raw = self._read_jsonl(
            "request_lifecycle",
            filter_key="request_id" if request_id else None,
            filter_value=request_id,
        )
        if limit > 0:
            raw = raw[-limit:]
        for record in raw:
            record.pop("session_id", None)
        return [RequestLifecycleEvent(**d) for d in raw]

    # =========================================================================
    # Per-Request Metrics
    # =========================================================================

    def save_request_metrics(self, metrics: RequestMetrics) -> None:
        """Save per-request metrics snapshot."""
        if not self.config.enable_metrics:
            return

        self._append_record("request_metrics", metrics.to_dict())

    def get_request_metrics(
        self,
        request_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[RequestMetrics]:
        """Load recent per-request metrics snapshots in chronological order."""
        raw = self._read_jsonl(
            "request_metrics",
            filter_key="request_id" if request_id else None,
            filter_value=request_id,
        )
        if limit > 0:
            raw = raw[-limit:]
        for record in raw:
            record.pop("session_id", None)
        return [RequestMetrics(**d) for d in raw]

    # =========================================================================
    # RL Decisions
    # =========================================================================

    def save_rl_decision(self, decision: RLDecision) -> None:
        """Save an RL decision record."""
        if not self.config.enable_rl_decisions:
            return

        self._append_record("rl_decisions", decision.to_dict())

    def save_noop_decision(self, request_id: str, iteration: int, reason: str) -> None:
        """
        Save a no-op decision record when RL is disabled.

        This ensures we always have a record for each iteration,
        even if RL isn't making decisions.
        """
        decision = RLDecision(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            iteration=iteration,
            state={},
            state_hash="",
            action="noop",
            action_index=-1,
            reward=0.0,
            reward_components={},
            algorithm="qlearning",
            policy_version="qlearning-v1",
            epsilon=0.0,
            was_exploration=False,
            reason=reason,
        )
        self.save_rl_decision(decision)

    # =========================================================================
    # Error Logging
    # =========================================================================

    def save_error(
        self,
        error_type: ErrorType,
        message: str,
        request_id: Optional[str] = None,
        node_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Save an error record.

        Returns:
            Error ID for tracking.
        """
        if not self.config.enable_errors:
            return ""

        error = ErrorRecord(
            error_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            request_id=request_id,
            node_id=node_id,
            message=message,
            details=details or {},
        )
        self._append_record("errors", error.to_dict())
        return error.error_id

    # =========================================================================
    # Additional streams
    # =========================================================================

    def save_slo_violation(self, violation: SLOViolation) -> None:
        """Save a per-request SLO/resource violation."""
        if not self.config.enable_slo_violations:
            return
        self._append_record("slo_violations", violation.to_dict())

    def get_slo_violations(
        self,
        request_id: Optional[str] = None,
        limit: int = 500,
    ) -> list[SLOViolation]:
        raw = self._read_jsonl(
            "slo_violations",
            filter_key="request_id" if request_id else None,
            filter_value=request_id,
            limit=limit,
        )
        for record in raw:
            record.pop("session_id", None)
        return [SLOViolation(**d) for d in raw]

    def save_slo_decision_epoch(self, epoch: SLODecisionEpoch) -> None:
        """Persist one auditable binary outcome for the recent-SLO window."""
        if not self.config.enable_slo_violations:
            return
        self._append_record("slo_decision_epochs", epoch.to_dict())

    def save_tenant_fairness(self, record: TenantFairnessRecord) -> None:
        """Save tenant fairness snapshot."""
        if not self.config.enable_tenant_fairness:
            return
        self._append_record("tenant_fairness", record.to_dict())

    def save_leader_event(self, event: LeaderElectionEvent) -> None:
        """Save leader-election event."""
        if not self.config.enable_leader_events:
            return
        self._append_record("leader_election_events", event.to_dict())

    def save_node_plan_applied(self, event: NodePlanApplied) -> None:
        """Save per-node plan application event."""
        if not self.config.enable_node_plans:
            return
        self._append_record("node_plan_applied", event.to_dict())

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def load_requests(self) -> list[PersistedRequest]:
        """Load all requests from disk."""
        filepath = self._get_file("requests")
        if not filepath.exists():
            return []

        latest_by_id: dict[str, PersistedRequest] = {}

        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data.pop("session_id", None)
                        request = PersistedRequest.from_dict(data)
                        # Keep only the latest version of each request (last write wins)
                        latest_by_id[request.request_id] = request
        except FileNotFoundError:
            return []

        requests = list(latest_by_id.values())

        # Update memory cache
        for req in requests:
            self._requests[req.request_id] = req

        return requests

    def get_request_history(self, request_id: str) -> list[dict]:
        """Get full history of a request (all versions)."""
        return self._read_jsonl("requests", filter_key="request_id", filter_value=request_id)

    def get_config_history(self, request_id: str) -> list[ConfigurationChange]:
        """Get configuration change history for a request."""
        raw = self._read_jsonl("config_changes", filter_key="request_id", filter_value=request_id)
        for record in raw:
            record.pop("session_id", None)
        changes = [ConfigurationChange(**d) for d in raw]
        return sorted(changes, key=lambda c: c.iteration)

    def get_rl_decisions(self, request_id: Optional[str] = None, limit: int = 1000) -> list[RLDecision]:
        """Get RL decision history."""
        raw = self._read_jsonl(
            "rl_decisions",
            filter_key="request_id" if request_id else None,
            filter_value=request_id,
            limit=limit,
        )
        for record in raw:
            record.pop("session_id", None)
        return [RLDecision(**d) for d in raw]

    def get_errors(
        self, error_type: Optional[ErrorType] = None, request_id: Optional[str] = None, limit: int = 100
    ) -> list[ErrorRecord]:
        """Get error records with optional filtering."""
        # Use _read_jsonl for basic filtering, then apply type filter
        raw = self._read_jsonl(
            "errors",
            filter_key="request_id" if request_id else None,
            filter_value=request_id,
        )
        errors: list[ErrorRecord] = []
        for data in raw:
            if error_type and data.get("error_type") != error_type.value:
                continue
            data["error_type"] = ErrorType(data["error_type"])
            errors.append(ErrorRecord(**data))
            if len(errors) >= limit:
                break
        return errors

    def close(self) -> None:
        """No-op: every write opens and closes its own file."""
        pass


# =============================================================================
# Default Instance
# =============================================================================

# Default persistence manager instance
_default_persistence: Optional[PersistenceManager] = None


def get_persistence_manager(session_id: Optional[str] = None) -> PersistenceManager:
    """Get the default persistence manager instance.

    On first call, reads ``ARGOS_SESSION_ID`` and ``ARGOS_PERSISTENCE_DIR``.
    An explicit persistence directory takes precedence over the default
    session path while the session identifier remains embedded in records.
    """
    global _default_persistence
    if _default_persistence is None:
        import os

        sid = session_id or os.environ.get("ARGOS_SESSION_ID")
        output_dir = os.environ.get("ARGOS_PERSISTENCE_DIR")
        config = PersistenceConfig(
            output_dir=Path(output_dir) if output_dir else None,
            session_id=sid,
        )
        _default_persistence = PersistenceManager(config)
        _default_persistence.initialize()
    return _default_persistence


def configure_persistence(config: PersistenceConfig) -> PersistenceManager:
    """Configure and return a new persistence manager."""
    global _default_persistence
    _default_persistence = PersistenceManager(config)
    _default_persistence.initialize()
    return _default_persistence
