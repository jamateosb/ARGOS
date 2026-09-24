# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Orchestrator REST API for job submission and management.

Provides endpoints for users to submit analytics requests and
query orchestration status, including RL metrics and persistence data.

Security:
- If `ORCHESTRATOR_API_TOKEN` is set in the environment, requests must include
  `X-API-Key: <token>` header for all endpoints except /cluster/status (health check).

New features:
- RL agent metrics and decision history
- Per-request metrics tracking
- Configuration change history
- Error logs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from argos.domain.requests import (
    PRIORITY_LEVELS,
    AnalyticsRequest,
    PlacementLimits,
    ResourceLimits,
)
from argos.common.constants import RL_AGENTS_DIR, TUNING_DIR
from argos.config import load_project_config
from argos.domain.cost import cost_model_rates_from_mapping
from argos.orchestrator.logger import ExperimentLogger, ExperimentLoggerConfig
from argos.orchestrator.loop import IterationMetrics, OrchestrationLoop, OrchestrationLoopConfig
from argos.orchestrator.persistence import (
    ErrorType,
    get_persistence_manager,
)
from argos.orchestrator.rl.selection import resolve_algorithm

_RL_AGENTS_DIR = RL_AGENTS_DIR
_TUNING_DIR = TUNING_DIR


logger = logging.getLogger(__name__)

# ============================================================================
# Authentication
# ============================================================================

_ORCHESTRATOR_API_TOKEN = os.getenv("ORCHESTRATOR_API_TOKEN")


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Validate API key if ORCHESTRATOR_API_TOKEN is configured."""
    if _ORCHESTRATOR_API_TOKEN and x_api_key != _ORCHESTRATOR_API_TOKEN:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================================================================
# API Models
# ============================================================================


class SubmitJobRequest(BaseModel):
    """Request body for POST /submit-job."""

    service_type: str = Field(default="geo_heatmap", description="Type of analytics service (heatmap or geo_heatmap)")
    coverage_min: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum coverage")
    coverage_max: float = Field(default=0.8, ge=0.0, le=1.0, description="Maximum coverage")
    sample_min: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum sample rate")
    sample_max: float = Field(default=0.8, ge=0.0, le=1.0, description="Maximum sample rate")
    freshness_min: float = Field(default=30.0, gt=0.0, description="Minimum freshness (seconds)")
    freshness_max: float = Field(default=120.0, gt=0.0, description="Maximum freshness (seconds)")
    response_time_min: Optional[float] = Field(default=None, gt=0.0, description="Minimum response time bound (s)")
    response_time_max: Optional[float] = Field(default=None, gt=0.0, description="Maximum response time bound (s)")
    cost_budget: Optional[float] = Field(default=None, description="Optional cost budget (for future use)")
    profile_name: str = Field(default="", description="Stable workload profile identifier")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    priority: str = Field(default="standard", description="critical | standard | best_effort")
    algorithm: str = Field(default="auto", description="auto | qlearning | dqn | ppo")
    resource_limits: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional resource limits: cpu_max_percent, memory_max_percent",
    )
    placement_limits: Optional[dict[str, int]] = Field(
        default=None,
        description="Optional placement limits: max_nodes, max_jobs_per_node",
    )
    input_multiplier: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Controlled multiplier for trajectory input volume",
    )


class SubmitJobResponse(BaseModel):
    """Response for POST /submit-job."""

    request_id: str
    service_type: str
    status: str
    effective_coverage: float
    effective_sample: float
    effective_freshness: float
    assigned_nodes: int
    profile_name: str = ""
    tenant_id: str = "default"
    priority: str = "standard"
    algorithm: str = "auto"
    policy_version: str = "auto"
    resource_limits: dict[str, float] = Field(default_factory=dict)
    frozen_policy_required: bool = False
    frozen_policy_loaded: bool = False
    frozen_policy_unchanged: bool = True
    rl_steps: int = 0
    reason: str = ""
    queued_position: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Response for GET /job/{request_id}."""

    request_id: str
    service_type: str
    status: str
    effective_config: dict
    assigned_nodes: list[str]
    created_at: str
    iteration_count: int = 0
    profile_name: str = ""
    tenant_id: str = "default"
    priority: str = "standard"
    algorithm: str = "auto"
    policy_version: str = "auto"
    resource_limits: dict[str, float] = Field(default_factory=dict)
    effective_assignments: list[str] = Field(default_factory=list)
    request_metrics: dict[str, Any] = Field(default_factory=dict)
    violations: list[dict] = Field(default_factory=list)
    frozen_policy_required: bool = False
    frozen_policy_loaded: bool = False
    frozen_policy_unchanged: bool = True
    rl_steps: int = 0
    reason: str = ""
    queued_position: Optional[int] = None


class ClusterStatusResponse(BaseModel):
    """Response for GET /cluster/status."""

    total_nodes: int
    active_nodes: int
    sleeping_nodes: int
    active_requests: int
    queued_requests: int = 0
    orchestrator_running: bool
    last_iteration_ms: float
    rl_enabled: bool
    control_plane_enabled: bool = False
    control_plane_node_id: Optional[str] = None
    control_plane_leader_id: Optional[str] = None
    is_control_plane_leader: bool = True
    control_plane_nats_connected: bool = False


class RegisterNodeRequest(BaseModel):
    """Request body for POST /nodes/register."""

    node_id: str
    endpoint: str


# ============================================================================
# Orchestrator API Application
# ============================================================================


class OrchestratorAPI:
    """
    Orchestrator REST API.

    Manages the orchestration loop and exposes endpoints for:
    - Submitting analytics jobs
    - Querying job status
    - Registering/unregistering nodes
    - Cluster status monitoring
    - RL agent metrics and history
    - Persistence data access
    """

    def __init__(
        self,
        loop_config: Optional[OrchestrationLoopConfig] = None,
        logger_config: Optional[ExperimentLoggerConfig] = None,
        project_config: Optional[dict[str, Any]] = None,
    ):
        self.app = FastAPI(
            title="ARGOS Orchestrator API",
            version="0.3.0",
            description="Multidimensional Elasticity Orchestrator - Central coordination API with RL",
        )

        self._logger = ExperimentLogger(logger_config or ExperimentLoggerConfig())
        self._loop = OrchestrationLoop(
            config=loop_config or OrchestrationLoopConfig(),
            logger=self._logger,
        )
        self._loop_task: Optional[asyncio.Task] = None
        self._job_created_at: dict[str, datetime] = {}
        self._last_iteration_metrics: Optional[IterationMetrics] = None
        self._project_config = project_config or {}
        # Frozen-policy live-trial support: optionally force a learned algorithm and
        # load a serialized policy per request, evaluated without further learning.
        self._frozen_policy_path = os.getenv("ARGOS_LIVE_FROZEN_POLICY") or None
        self._frozen_algorithm = os.getenv("ARGOS_LIVE_FROZEN_ALGORITHM") or None
        self._frozen_policy_map = self._load_frozen_policy_map()
        self._frozen_policy_requests: set[str] = set()
        if self._frozen_policy_path and self._frozen_policy_map:
            raise RuntimeError("Configure either ARGOS_LIVE_FROZEN_POLICY or ARGOS_LIVE_FROZEN_POLICY_MAP, not both")
        configured_paths = [
            *(self._frozen_policy_map.values()),
            *([self._frozen_policy_path] if self._frozen_policy_path else []),
        ]
        missing_paths = [path for path in configured_paths if not Path(path).is_file()]
        if missing_paths:
            raise RuntimeError(f"Frozen policy artifacts do not exist: {missing_paths}")
        if configured_paths and not self._frozen_algorithm:
            raise RuntimeError("ARGOS_LIVE_FROZEN_ALGORITHM is required with a frozen policy")

        self._register_routes()

    @staticmethod
    def _load_frozen_policy_map() -> dict[str, str]:
        """Load a profile-to-policy mapping from JSON text or a JSON file."""
        raw = (os.getenv("ARGOS_LIVE_FROZEN_POLICY_MAP") or "").strip()
        if not raw:
            return {}
        source = Path(raw)
        try:
            payload = json.loads(source.read_text(encoding="utf-8")) if source.is_file() else json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid ARGOS_LIVE_FROZEN_POLICY_MAP: {exc}") from exc
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("ARGOS_LIVE_FROZEN_POLICY_MAP must be a non-empty JSON object")
        mapping = {str(name).strip(): str(path).strip() for name, path in payload.items()}
        if any(not name or not path for name, path in mapping.items()):
            raise RuntimeError("Frozen policy map contains an empty profile or path")
        return mapping

    def _frozen_policy_for_profile(self, profile_name: str) -> Optional[str]:
        """Resolve the exact frozen policy required by one submitted profile."""
        if self._frozen_policy_path:
            return self._frozen_policy_path
        if not self._frozen_policy_map:
            return None
        normalized = profile_name.strip()
        if not normalized:
            raise HTTPException(400, "profile_name is required for profile-specific frozen evaluation")
        path = self._frozen_policy_map.get(normalized)
        if path is None:
            raise HTTPException(400, f"No frozen policy configured for profile_name={normalized!r}")
        return path

    def _register_routes(self) -> None:
        """Register all API routes."""

        # ====================================================================
        # Job Management
        # ====================================================================

        @self.app.post("/submit-job", response_model=SubmitJobResponse)
        async def submit_job(
            request: SubmitJobRequest,
            _: None = Depends(_require_api_key),
        ):
            """Submit a new analytics job."""
            # Validate ranges
            if request.coverage_min > request.coverage_max:
                raise HTTPException(400, "coverage_min must be <= coverage_max")
            if request.sample_min > request.sample_max:
                raise HTTPException(400, "sample_min must be <= sample_max")
            if request.freshness_min > request.freshness_max:
                raise HTTPException(400, "freshness_min must be <= freshness_max")
            if (request.response_time_min is None) != (request.response_time_max is None):
                raise HTTPException(400, "response_time_min and response_time_max must be provided together")
            if (
                request.response_time_min is not None
                and request.response_time_max is not None
                and request.response_time_min > request.response_time_max
            ):
                raise HTTPException(400, "response_time_min must be <= response_time_max")
            if request.priority not in PRIORITY_LEVELS:
                raise HTTPException(400, f"priority must be one of {list(PRIORITY_LEVELS)}")
            try:
                normalized_algorithm = resolve_algorithm(request.algorithm, config=self._project_config)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if self._frozen_algorithm:
                normalized_algorithm = self._frozen_algorithm
            frozen_policy_path = self._frozen_policy_for_profile(request.profile_name)

            resource_limits = ResourceLimits.from_dict(request.resource_limits)
            placement_limits = PlacementLimits.from_dict(request.placement_limits)

            analytics_request = AnalyticsRequest(
                service_type=request.service_type,
                coverage_range=(request.coverage_min, request.coverage_max),
                sample_range=(request.sample_min, request.sample_max),
                freshness_range=(request.freshness_min, request.freshness_max),
                response_time_range=(
                    (request.response_time_min, request.response_time_max)
                    if request.response_time_min is not None and request.response_time_max is not None
                    else None
                ),
                cost_budget=request.cost_budget,
                profile_name=request.profile_name,
                tenant_id=request.tenant_id,
                priority=request.priority,
                algorithm=normalized_algorithm,
                resource_limits=resource_limits,
                placement_limits=placement_limits,
                input_multiplier=request.input_multiplier,
            )

            effective = self._loop.submit_request(analytics_request)
            if frozen_policy_path:
                self._loop.load_rl_agent(analytics_request.request_id, frozen_policy_path)
                self._frozen_policy_requests.add(analytics_request.request_id)
            rl_metrics = self._loop.get_rl_metrics().get(analytics_request.request_id, {})
            assigned = [
                n.node_id for n in self._loop.nodes.values() if analytics_request.request_id in n.assigned_request_ids
            ]
            self._job_created_at[analytics_request.request_id] = datetime.now(timezone.utc)
            status = self._loop.get_job_status(analytics_request.request_id)

            logger.info(f"Job submitted: {analytics_request.request_id} ({request.service_type})")

            return SubmitJobResponse(
                request_id=analytics_request.request_id,
                service_type=request.service_type,
                status=status.value,
                effective_coverage=effective.target_coverage,
                effective_sample=effective.target_sample,
                effective_freshness=effective.target_freshness,
                assigned_nodes=len(assigned),
                profile_name=analytics_request.profile_name,
                tenant_id=analytics_request.tenant_id,
                priority=analytics_request.priority,
                algorithm=rl_metrics.get("algorithm", analytics_request.algorithm),
                policy_version=rl_metrics.get("policy_version", "qlearning-v1"),
                resource_limits=analytics_request.resource_limits.to_dict(),
                frozen_policy_required=analytics_request.request_id in self._frozen_policy_requests,
                frozen_policy_loaded=bool(rl_metrics.get("frozen_policy_loaded", False)),
                frozen_policy_unchanged=bool(rl_metrics.get("policy_unchanged", True)),
                rl_steps=int(rl_metrics.get("steps", 0) or 0),
                reason=self._loop.get_job_status_reason(analytics_request.request_id),
                queued_position=self._loop._queue_position(analytics_request.request_id),
            )

        @self.app.get("/job/{request_id}", response_model=JobStatusResponse)
        async def get_job_status(
            request_id: str,
            _: None = Depends(_require_api_key),
        ):
            """Get status of a submitted job."""
            if request_id not in self._loop.active_requests:
                raise HTTPException(404, "Job not found")

            request = self._loop.active_requests[request_id]
            config = self._loop._effective_configs.get(request_id)

            assigned = [n.node_id for n in self._loop.nodes.values() if request_id in n.assigned_request_ids]

            iteration_count = self._loop._request_iterations.get(request_id, 0)
            persistence = get_persistence_manager()
            violations = [v.to_dict() for v in persistence.get_slo_violations(request_id=request_id, limit=50)]
            request_metrics = persistence.get_request_metrics(request_id, limit=1)
            rl_metrics = self._loop.get_rl_metrics().get(request_id, {})
            status = self._loop.get_job_status(request_id)

            return JobStatusResponse(
                request_id=request_id,
                service_type=request.service_type,
                status=status.value,
                effective_config=config.to_dict() if config else {},
                assigned_nodes=assigned,
                created_at=self._job_created_at.get(request_id, datetime.now(timezone.utc)).isoformat(),
                iteration_count=iteration_count,
                profile_name=request.profile_name,
                tenant_id=request.tenant_id,
                priority=request.priority,
                algorithm=rl_metrics.get("algorithm", request.algorithm),
                policy_version=rl_metrics.get("policy_version", "qlearning-v1"),
                resource_limits=request.resource_limits.to_dict(),
                effective_assignments=assigned,
                request_metrics=request_metrics[0].to_dict() if request_metrics else {},
                violations=violations,
                frozen_policy_required=request_id in self._frozen_policy_requests,
                frozen_policy_loaded=bool(rl_metrics.get("frozen_policy_loaded", False)),
                frozen_policy_unchanged=bool(rl_metrics.get("policy_unchanged", True)),
                rl_steps=int(rl_metrics.get("steps", 0) or 0),
                reason=self._loop.get_job_status_reason(request_id),
                queued_position=self._loop._queue_position(request_id),
            )

        @self.app.delete("/job/{request_id}")
        async def cancel_job(
            request_id: str,
            reason: str = Query(default="user_cancelled", description="Cancellation reason"),
            _: None = Depends(_require_api_key),
        ):
            """Cancel a running job."""
            if request_id not in self._loop.active_requests:
                raise HTTPException(404, "Job not found")

            if request_id in self._frozen_policy_requests:
                self._loop.assert_rl_policy_unchanged(request_id)
            success = self._loop.cancel_request(request_id, reason)
            if not success:
                raise HTTPException(500, "Failed to cancel job")
            self._frozen_policy_requests.discard(request_id)

            logger.info(f"Job cancelled: {request_id} (reason: {reason})")

            return {"status": "cancelled", "request_id": request_id, "reason": reason}

        @self.app.get("/jobs")
        async def list_jobs(_: None = Depends(_require_api_key)):
            """List all active jobs."""
            jobs = []
            for request_id, request in self._loop.active_requests.items():
                config = self._loop._effective_configs.get(request_id)
                assigned = [n.node_id for n in self._loop.nodes.values() if request_id in n.assigned_request_ids]
                jobs.append(
                    {
                        "request_id": request_id,
                        "service_type": request.service_type,
                        "status": self._loop.get_job_status(request_id).value,
                        "effective_coverage": config.target_coverage if config else 0,
                        "effective_sample": config.target_sample if config else 0,
                        "effective_freshness": config.target_freshness if config else 0,
                        "assigned_nodes": assigned,
                        "iteration_count": self._loop._request_iterations.get(request_id, 0),
                        "tenant_id": request.tenant_id,
                        "priority": request.priority,
                        "resource_limits": request.resource_limits.to_dict(),
                        "reason": self._loop.get_job_status_reason(request_id),
                        "queued_position": self._loop._queue_position(request_id),
                    }
                )
            return {"jobs": jobs, "total": len(jobs)}

        # ====================================================================
        # Cluster Status
        # ====================================================================

        @self.app.get("/cluster/status", response_model=ClusterStatusResponse)
        async def get_cluster_status():
            """Get overall cluster status. No authentication required (health check)."""
            nodes = self._loop.nodes
            total = len(nodes)
            active = len([n for n in nodes.values() if n.is_active])
            cp_state = self._loop.get_control_plane_state()

            last_iteration_ms = 0.0
            if self._last_iteration_metrics:
                last_iteration_ms = self._last_iteration_metrics.iteration_time_ms

            return ClusterStatusResponse(
                total_nodes=total,
                active_nodes=active,
                sleeping_nodes=total - active,
                active_requests=self._loop.running_request_count(),
                queued_requests=self._loop.pending_request_count(),
                orchestrator_running=self._loop.is_running,
                last_iteration_ms=last_iteration_ms,
                rl_enabled=self._loop.config.enable_rl,
                control_plane_enabled=bool(cp_state.get("enabled")),
                control_plane_node_id=cp_state.get("node_id"),
                control_plane_leader_id=cp_state.get("leader_id"),
                is_control_plane_leader=bool(cp_state.get("is_leader", True)),
                control_plane_nats_connected=bool(cp_state.get("nats_connected", False)),
            )

        @self.app.get("/cluster/control-plane")
        async def get_control_plane_status(_: None = Depends(_require_api_key)):
            """Get detailed control-plane state (leader election + NATS)."""
            return self._loop.get_control_plane_state()

        @self.app.get("/cluster/nodes")
        async def list_nodes(_: None = Depends(_require_api_key)):
            """List all registered nodes."""
            return {
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "endpoint": n.endpoint,
                        "is_active": n.is_active,
                        "last_seen": n.last_seen.isoformat() if n.last_seen else None,
                        "assigned_requests": n.assigned_request_ids,
                        "assigned_count": len(n.assigned_request_ids),
                        "active_workers": (n.last_metrics or {}).get("active_workers", 0),
                        "max_analytics_per_node": (n.last_metrics or {}).get("max_analytics_per_node"),
                        "quota_usage": (n.last_metrics or {}).get("quota_usage", 0.0),
                        "health_failures": n.health_failures,
                        "latency_ms": n.latency_ms,
                        "last_error": n.last_error,
                        "analytics": (n.last_metrics or {}).get("analytics", []),
                        "violations": (n.last_metrics or {}).get("violations", {}),
                        "violations_recent": ((n.last_metrics or {}).get("violations", {}) or {}).get(
                            "recent_total", 0
                        ),
                        "runtime_mode": (n.last_metrics or {}).get("runtime_mode", "unknown"),
                        "current_plan_version": (n.last_metrics or {}).get("current_plan_version", 0),
                    }
                    for n in self._loop.nodes.values()
                ]
            }

        @self.app.post("/nodes/register")
        async def register_node(
            request: RegisterNodeRequest,
            _: None = Depends(_require_api_key),
        ):
            """Register a new node with the orchestrator."""
            self._loop.register_node(request.node_id, request.endpoint)
            logger.info(f"Node registered: {request.node_id} at {request.endpoint}")
            return {"status": "registered", "node_id": request.node_id}

        @self.app.delete("/nodes/{node_id}")
        async def unregister_node(
            node_id: str,
            _: None = Depends(_require_api_key),
        ):
            """Unregister a node."""
            self._loop.unregister_node(node_id)
            logger.info(f"Node unregistered: {node_id}")
            return {"status": "unregistered", "node_id": node_id}

        # ====================================================================
        # Orchestrator Control
        # ====================================================================

        @self.app.post("/orchestrator/start")
        async def start_orchestrator(_: None = Depends(_require_api_key)):
            """Start the orchestration loop."""
            if self._loop_task is None or self._loop_task.done():
                self._loop_task = asyncio.create_task(self._run_loop_with_metrics())
                logger.info("Orchestration loop started")
                return {"status": "started"}
            return {"status": "already_running"}

        @self.app.post("/orchestrator/stop")
        async def stop_orchestrator(_: None = Depends(_require_api_key)):
            """Stop the orchestration loop."""
            self._loop.stop()
            if self._loop_task:
                try:
                    await asyncio.wait_for(self._loop_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Orchestration loop stop timed out")
            await self._loop.close_async()
            logger.info("Orchestration loop stopped")
            return {"status": "stopped"}

        @self.app.get("/orchestrator/config")
        async def get_orchestrator_config(_: None = Depends(_require_api_key)):
            """Get current orchestrator configuration."""
            config = self._loop.config
            return {
                "poll_interval_seconds": config.poll_interval_seconds,
                "timeout_seconds": config.timeout_seconds,
                "max_health_failures": config.max_health_failures,
                "enable_logging": config.enable_logging,
                "enable_rl": config.enable_rl,
                "enable_persistence": config.enable_persistence,
                "rl_alpha": config.rl_alpha,
                "rl_gamma": config.rl_gamma,
                "rl_epsilon": config.rl_epsilon,
                "rl_seed": config.rl_seed,
                "iteration_budget_ms": config.iteration_budget_ms,
                "enable_control_plane": config.enable_control_plane,
                "control_plane_node_id": config.control_plane_node_id,
                "control_plane_lease_seconds": config.control_plane_lease_seconds,
                "control_plane_heartbeat_interval_seconds": config.control_plane_heartbeat_interval_seconds,
                "enable_nats_bus": config.enable_nats_bus,
                "nats_url": config.nats_url,
                "deployment_profile": config.deployment_profile,
            }

        # ====================================================================
        # RL Metrics and History
        # ====================================================================

        @self.app.get("/rl/metrics")
        async def get_rl_metrics(_: None = Depends(_require_api_key)):
            """Get RL agent metrics for all requests."""
            metrics = self._loop.get_rl_metrics()
            return {
                "rl_enabled": self._loop.config.enable_rl,
                "agents": metrics,
            }

        @self.app.get("/rl/metrics/{request_id}")
        async def get_rl_metrics_for_request(
            request_id: str,
            _: None = Depends(_require_api_key),
        ):
            """Get RL agent metrics for a specific request."""
            metrics = self._loop.get_rl_metrics()
            if request_id not in metrics:
                raise HTTPException(404, "RL agent not found for request")
            return metrics[request_id]

        @self.app.get("/rl/decisions")
        async def get_rl_decisions(
            request_id: Optional[str] = None,
            limit: int = Query(default=100, le=1000),
            _: None = Depends(_require_api_key),
        ):
            """Get RL decision history."""
            persistence = get_persistence_manager()
            decisions = persistence.get_rl_decisions(request_id=request_id, limit=limit)
            return {
                "decisions": [d.to_dict() for d in decisions],
                "total": len(decisions),
            }

        @self.app.post("/rl/save")
        async def save_rl_agents(
            path: str = Query(default=str(_RL_AGENTS_DIR), description="Save path"),
            _: None = Depends(_require_api_key),
        ):
            """Save all RL agents to disk."""
            self._loop.save_rl_agents(path)
            return {"status": "saved", "path": path}

        # ====================================================================
        # Persistence / History
        # ====================================================================

        @self.app.get("/history/requests")
        async def get_request_history(
            request_id: Optional[str] = None,
            _: None = Depends(_require_api_key),
        ):
            """Get request history."""
            persistence = get_persistence_manager()
            if request_id:
                history = persistence.get_request_history(request_id)
                return {"history": history}
            requests = persistence.load_requests()
            return {"requests": [r.to_dict() for r in requests]}

        @self.app.get("/history/configs/{request_id}")
        async def get_config_history(
            request_id: str,
            _: None = Depends(_require_api_key),
        ):
            """Get configuration change history for a request."""
            persistence = get_persistence_manager()
            changes = persistence.get_config_history(request_id)
            return {
                "request_id": request_id,
                "changes": [c.to_dict() for c in changes],
                "total": len(changes),
            }

        @self.app.get("/history/errors")
        async def get_error_history(
            error_type: Optional[str] = None,
            request_id: Optional[str] = None,
            limit: int = Query(default=100, le=1000),
            _: None = Depends(_require_api_key),
        ):
            """Get error history."""
            persistence = get_persistence_manager()
            et = ErrorType(error_type) if error_type else None
            errors = persistence.get_errors(error_type=et, request_id=request_id, limit=limit)
            return {
                "errors": [e.to_dict() for e in errors],
                "total": len(errors),
            }

        @self.app.get("/history/slo-violations")
        async def get_slo_violations(
            request_id: Optional[str] = None,
            limit: int = Query(default=200, le=2000),
            _: None = Depends(_require_api_key),
        ):
            """Get SLO/resource violations history."""
            persistence = get_persistence_manager()
            violations = persistence.get_slo_violations(request_id=request_id, limit=limit)
            return {
                "violations": [v.to_dict() for v in violations],
                "total": len(violations),
            }

        @self.app.get("/history/request-lifecycle")
        async def get_request_lifecycle(
            request_id: Optional[str] = None,
            limit: int = Query(default=500, le=5000),
            _: None = Depends(_require_api_key),
        ):
            """Get request lifecycle/admission history."""
            persistence = get_persistence_manager()
            events = persistence.get_request_lifecycle(request_id=request_id, limit=limit)
            return {
                "events": [e.to_dict() for e in events],
                "total": len(events),
            }

        # ====================================================================
        # Experiment Info
        # ====================================================================

        @self.app.get("/experiments")
        async def get_experiment_info(_: None = Depends(_require_api_key)):
            """Get experiment logging info."""
            return {
                "output_path": str(self._logger.output_path) if self._logger.output_path else None,
                "iteration_count": self._logger.iteration_count,
                "loop_iteration_count": self._loop.iteration_count,
            }

    async def _run_loop_with_metrics(self) -> None:
        """Run the orchestration loop and capture metrics."""
        self._loop._running = True

        # Requests transition to RUNNING inside the loop only after admission.
        get_persistence_manager()

        try:
            while self._loop._running:
                self._last_iteration_metrics = await self._loop.run_once()
                loop_time_s = self._last_iteration_metrics.iteration_time_ms / 1000.0
                sleep_s = max(0.0, self._loop.config.poll_interval_seconds - loop_time_s)
                await asyncio.sleep(sleep_s)
        finally:
            await self._loop.close_async()


# ===========================================================================
# Hyperparameter loading utility
# ===========================================================================


def _load_best_rl_params(
    algorithm: str = "auto",
    *,
    project_config: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, float]]:
    """
    Load best RL hyperparameters for the default API algorithm if available.

    Returns:
        Dict with RL params, or None if file doesn't exist.
    """
    try:
        normalized = resolve_algorithm(algorithm, config=project_config)
    except ValueError:
        normalized = "qlearning"
    for best_params_file in (_TUNING_DIR / f"best_params_{normalized}.yaml", _TUNING_DIR / "best_params.yaml"):
        if not best_params_file.exists():
            continue
        try:
            with open(best_params_file, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            params = raw.get("params", raw) if isinstance(raw, dict) else {}
            stored_algorithm = raw.get("algorithm") if isinstance(raw, dict) else None
            if best_params_file.name == "best_params.yaml" and stored_algorithm not in (None, normalized):
                continue
            if best_params_file.name == "best_params.yaml" and stored_algorithm is None and normalized != "qlearning":
                continue
            params = {k: v for k, v in params.items() if k != "algorithm"}
            logger.info("Loaded tuned RL params from %s: %s", best_params_file, params)
            return params
        except Exception as e:
            logger.warning("Failed to load %s: %s", best_params_file, e)
    return None


def _load_project_config() -> dict[str, Any]:
    """Load config.yaml for API defaults when available."""
    try:
        return load_project_config()
    except Exception as exc:
        logger.warning("Failed to load project config: %s", exc)
        return {}


# Default API instance for direct use with uvicorn
def create_app() -> FastAPI:
    """
    Create the orchestrator API application.

    Automatically loads tuned RL hyperparameters from data/tuning/best_params.yaml
    if it exists. Otherwise uses default values.
    """
    project_config = _load_project_config()
    orchestration_cfg = project_config.get("orchestration", {}) or {}
    control_plane_cfg = orchestration_cfg.get("control_plane", {}) or {}
    cost_model_cfg = project_config.get("cost_model", {}) or {}
    rl_cfg = project_config.get("rl", {}) or {}
    best_params = _load_best_rl_params(str(rl_cfg.get("algorithm", "auto")), project_config=project_config)

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        return float(raw) if raw is not None else float(default)

    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw is not None else int(default)

    cp_kwargs = {
        "poll_interval_seconds": float(orchestration_cfg.get("poll_interval_seconds", 5.0)),
        "timeout_seconds": float(orchestration_cfg.get("timeout_seconds", 5.0)),
        "max_health_failures": int(orchestration_cfg.get("max_health_failures", 3)),
        "parallel_requests": int(orchestration_cfg.get("parallel_requests", 10)),
        "max_jobs_per_node": int(orchestration_cfg.get("max_jobs_per_node", 3)),
        "max_parallel_polls": int(orchestration_cfg.get("max_parallel_polls", 16)),
        "max_parallel_pushes": int(orchestration_cfg.get("max_parallel_pushes", 16)),
        "enable_control_plane": _env_bool("ARGOS_CONTROL_PLANE_ENABLED", bool(control_plane_cfg.get("enabled", True))),
        "control_plane_node_id": os.getenv("ARGOS_CONTROL_PLANE_NODE_ID") or None,
        "control_plane_lease_seconds": _env_int(
            "ARGOS_CONTROL_PLANE_LEASE_SECONDS", int(control_plane_cfg.get("lease_seconds", 15))
        ),
        "control_plane_heartbeat_interval_seconds": _env_float(
            "ARGOS_CONTROL_PLANE_HEARTBEAT_SECONDS", float(control_plane_cfg.get("heartbeat_interval_seconds", 5.0))
        ),
        "enable_nats_bus": _env_bool(
            "ARGOS_CONTROL_PLANE_NATS_ENABLED", bool(control_plane_cfg.get("nats_enabled", False))
        ),
        "nats_url": os.getenv(
            "ARGOS_CONTROL_PLANE_NATS_URL", str(control_plane_cfg.get("nats_url", "nats://127.0.0.1:4222"))
        ),
        "nats_subject_prefix": os.getenv(
            "ARGOS_CONTROL_PLANE_SUBJECT_PREFIX", str(control_plane_cfg.get("nats_subject_prefix", "argos.control"))
        ),
        "iteration_budget_ms": _env_float(
            "ARGOS_LOOP_ITERATION_BUDGET_MS", float(orchestration_cfg.get("iteration_budget_ms", 2000.0))
        ),
        "deployment_profile": str(
            os.getenv("ARGOS_DEPLOYMENT_PROFILE", str(orchestration_cfg.get("deployment_profile", "local")))
        )
        .strip()
        .lower()
        or "local",
        "cost_model_rates": cost_model_rates_from_mapping(cost_model_cfg),
    }

    if best_params:
        # Map Optuna parameter names to OrchestrationLoopConfig field names
        loop_config = OrchestrationLoopConfig(
            rl_epsilon=best_params.get("epsilon", float(rl_cfg.get("exploration_rate", 0.15))),
            rl_alpha=best_params.get("learning_rate", float(rl_cfg.get("learning_rate", 0.1))),
            rl_gamma=best_params.get("discount_factor", float(rl_cfg.get("discount_factor", 0.95))),
            rl_epsilon_decay=best_params.get("epsilon_decay", float(rl_cfg.get("exploration_decay", 0.995))),
            rl_min_epsilon=float(rl_cfg.get("exploration_min", 0.01)),
            **cp_kwargs,
        )
        logger.info("Using tuned RL hyperparameters")
    else:
        loop_config = OrchestrationLoopConfig(
            rl_epsilon=float(rl_cfg.get("exploration_rate", 0.15)),
            rl_alpha=float(rl_cfg.get("learning_rate", 0.1)),
            rl_gamma=float(rl_cfg.get("discount_factor", 0.95)),
            rl_epsilon_decay=float(rl_cfg.get("exploration_decay", 0.995)),
            rl_min_epsilon=float(rl_cfg.get("exploration_min", 0.01)),
            **cp_kwargs,
        )
        logger.info("Using default RL hyperparameters")

    # Live-trial overrides: run the loop as a frozen evaluator (no learning) or as a
    # non-learning static controller, selected by environment for each trial variant.
    loop_config.enable_rl = _env_bool("ARGOS_LIVE_ENABLE_RL", loop_config.enable_rl)
    loop_config.rl_training = _env_bool("ARGOS_LIVE_RL_TRAINING", loop_config.rl_training)
    _live_quantiles = os.getenv("ARGOS_LIVE_INITIAL_QUANTILES")
    if _live_quantiles:
        _qc, _qs, _qf = (float(x) for x in _live_quantiles.split(","))
        loop_config.initial_coverage_quantile = _qc
        loop_config.initial_sample_quantile = _qs
        loop_config.initial_freshness_quantile = _qf

    api = OrchestratorAPI(loop_config=loop_config, project_config=project_config)
    return api.app


app = create_app()

# To run: uvicorn argos.orchestrator.api:app --reload --port 8001
