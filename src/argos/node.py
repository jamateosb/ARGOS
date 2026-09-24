# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Worker-node HTTP API.

- One analytics runtime per request assigned to the node.
- Per-request CPU and memory limits with violation tracking.
- ``POST /configure`` accepts a node plan or a single configuration.
- ``GET /metrics`` returns a structured payload (``state``) and a flat one
  (``state_flat``) with the same host and runtime metrics.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, Optional

import psutil
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from argos.settings import SamplingConfig
from argos.domain.architecture import ServiceMetrics
from argos.domain.monitor import ArchitectureMonitor
from argos.common.constants import DEFAULT_MACHINE_ID
from argos.services.geo_heatmap import GeoHeatmapResult, GeoHeatmapService

__all__ = ["app", "node_state"]


PRIORITY_RANK = {
    "critical": 0,
    "standard": 1,
    "best_effort": 2,
}

_PROCESS_SERVICES: dict[tuple[str, int, int, int], GeoHeatmapService] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_node_id(machine_id: str) -> int:
    env_node = os.getenv("NODE_ID")
    if env_node:
        try:
            return max(1, int(env_node))
        except ValueError:
            pass

    m = re.search(r"(\d+)$", machine_id or "")
    if m:
        try:
            return max(1, int(m.group(1)))
        except ValueError:
            pass
    return 1


def _compute_geo_heatmap_once(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Compute one geo-heatmap cycle in a separate process.

    This mode bypasses the GIL for CPU-bound paths when
    `ARGOS_RUNTIME_EXECUTION_MODE=process`.
    """
    node_id = int(payload.get("node_id", 1))
    total_nodes = int(payload.get("total_nodes", 3))
    resolution = int(payload.get("resolution", 4))
    dataset_path = str(payload.get("dataset_path") or "")
    cache_key = (dataset_path, node_id, total_nodes, resolution)
    service = _PROCESS_SERVICES.get(cache_key)
    if service is None:
        kwargs: dict[str, Any] = {
            "node_id": node_id,
            "total_nodes": total_nodes,
            "sample_rate": float(payload.get("target_sample", 0.5)),
            "resolution": resolution,
        }
        if dataset_path:
            kwargs["dataset_path"] = dataset_path
        service = GeoHeatmapService(**kwargs)
        _PROCESS_SERVICES[cache_key] = service

    service.configure(
        sample_rate=float(payload.get("target_sample", 0.5)),
        resolution=resolution,
        query_params=dict(payload.get("query_params") or {}),
        node_id=node_id,
        total_nodes=total_nodes,
        input_multiplier=max(1, int(payload.get("input_multiplier", 1))),
    )
    result = service.compute_heatmap(str(payload.get("compute_request_id", payload.get("request_id", "unknown"))))
    return result.to_dict()


@dataclass
class AnalyticsRuntime:
    """One analytics worker runtime bound to one request_id."""

    request_id: str
    service_type: str
    node_id: int
    tenant_id: str = "default"
    priority: str = "standard"
    target_coverage: float = 0.5
    target_sample: float = 0.5
    target_freshness: float = 60.0
    total_nodes: int = 3
    cpu_max_percent: float = 80.0
    memory_max_percent: float = 85.0
    assigned_node_count: int = 1
    input_multiplier: int = 1
    plan_version: int = 0
    leader_term: int = 0
    source: str = "http"
    execution_mode: str = "thread"
    process_pool: Optional[ProcessPoolExecutor] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    service: GeoHeatmapService = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _stop_event: Event = field(default_factory=Event, init=False)
    _config_event: Event = field(default_factory=Event, init=False)
    _thread: Optional[Thread] = field(default=None, init=False)
    _is_running: bool = field(default=False, init=False)
    _iterations: int = field(default=0, init=False)
    _last_result: Optional[GeoHeatmapResult] = field(default=None, init=False)
    _violations: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.service = GeoHeatmapService(
            node_id=self.node_id,
            total_nodes=max(1, self.total_nodes),
            sample_rate=self.target_sample,
            input_multiplier=self.input_multiplier,
        )

    def update_from_config(self, cfg: dict[str, Any]) -> None:
        """Update runtime parameters from a node-plan config."""
        with self._lock:
            previous = (
                self.target_sample,
                self.target_freshness,
                self.total_nodes,
                self.input_multiplier,
            )
            self.service_type = str(cfg.get("service_type", self.service_type))
            self.tenant_id = str(cfg.get("tenant_id", self.tenant_id))
            self.priority = str(cfg.get("priority", self.priority))
            self.target_coverage = float(cfg.get("target_coverage", self.target_coverage))
            self.target_sample = float(cfg.get("target_sample", self.target_sample))
            self.target_freshness = float(cfg.get("target_freshness", self.target_freshness))
            self.total_nodes = int(cfg.get("total_nodes", self.total_nodes))
            self.assigned_node_count = int(cfg.get("assigned_node_count", self.assigned_node_count))
            self.input_multiplier = max(1, int(cfg.get("input_multiplier", self.input_multiplier)))
            self.cpu_max_percent = float(cfg.get("cpu_max_percent", self.cpu_max_percent))
            self.memory_max_percent = float(cfg.get("memory_max_percent", self.memory_max_percent))
            self.plan_version = int(cfg.get("plan_version", self.plan_version))
            self.leader_term = int(cfg.get("leader_term", self.leader_term))
            self.source = str(cfg.get("source", self.source))
            self.updated_at = _now_iso()

            self.service.configure(
                sample_rate=self.target_sample,
                node_id=self.node_id,
                total_nodes=max(1, self.total_nodes),
                input_multiplier=self.input_multiplier,
            )
            current = (
                self.target_sample,
                self.target_freshness,
                self.total_nodes,
                self.input_multiplier,
            )
            if current != previous:
                self._config_event.set()

    def _probe_node_resources(self) -> tuple[float, float]:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return float(cpu), float(mem)

    def _append_violation(self, metric: str, value: float, limit: float) -> None:
        severity = "critical" if value > (limit + 10.0) else "warning"
        self._violations.append(
            {
                "timestamp": _now_iso(),
                "metric": metric,
                "measured": value,
                "limit": limit,
                "severity": severity,
            }
        )
        # Keep the last 1000 records in memory.
        if len(self._violations) > 1000:
            self._violations = self._violations[-1000:]

    def compute_once(self, suffix: Optional[int] = None) -> Optional[GeoHeatmapResult]:
        """Run one computation cycle if limits allow it."""
        cpu, mem = self._probe_node_resources()
        if cpu > self.cpu_max_percent:
            self._append_violation("cpu_percent", cpu, self.cpu_max_percent)
            return None
        if mem > self.memory_max_percent:
            self._append_violation("memory_percent", mem, self.memory_max_percent)
            return None

        rid = self.request_id if suffix is None else f"{self.request_id}_{suffix}"
        if self.execution_mode == "process" and self.process_pool is not None:
            try:
                payload = {
                    "request_id": self.request_id,
                    "compute_request_id": rid,
                    "node_id": self.node_id,
                    "total_nodes": self.total_nodes,
                    "target_sample": self.target_sample,
                    "input_multiplier": self.input_multiplier,
                    "resolution": self.service.resolution,
                    "query_params": self.service.query_params,
                    "dataset_path": str(self.service.dataset_path),
                }
                data = self.process_pool.submit(_compute_geo_heatmap_once, payload).result()
                result = GeoHeatmapResult(
                    request_id=str(data["request_id"]),
                    timestamp=float(data["timestamp"]),
                    node_id=int(data["node_id"]),
                    query_params=dict(data.get("query_params", {})),
                    total_points_processed=int(data.get("total_points_processed", 0)),
                    filtered_points=int(data.get("filtered_points", 0)),
                    heatmap_cells=int(data.get("heatmap_cells", 0)),
                    hotspots_detected=int(data.get("hotspots_detected", 0)),
                    processing_time_ms=float(data.get("processing_time_ms", 0.0)),
                    data_volume_bytes=int(data.get("data_volume_bytes", 0)),
                    sample_rate_applied=float(data.get("sample_rate_applied", self.target_sample)),
                    checksum=str(data.get("checksum", "")),
                    spatial_fidelity=(
                        float(data["spatial_fidelity"]) if data.get("spatial_fidelity") is not None else None
                    ),
                    hotspot_recall=(
                        float(data["hotspot_recall"]) if data.get("hotspot_recall") is not None else None
                    ),
                    heatmap_data=list(data.get("heatmap_data", [])),
                )
                self.service._last_result = result
            except Exception:
                self._append_violation("runtime_process_error", 1.0, 0.0)
                return None
        else:
            result = self.service.compute_heatmap(rid)

        with self._lock:
            self._iterations += 1
            self._last_result = result
            self.updated_at = _now_iso()
        return result

    def _run(self) -> None:
        self._is_running = True
        while not self._stop_event.is_set():
            self.compute_once(self._iterations)
            wait_s = max(1.0, float(self.target_freshness))
            self._config_event.wait(timeout=wait_s)
            self._config_event.clear()
        self._is_running = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._config_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._config_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._is_running = False

    def recent_violations(self, window_seconds: int = 300) -> list[dict[str, Any]]:
        cutoff = time.time() - window_seconds
        recent = []
        for v in self._violations:
            ts = datetime.fromisoformat(v["timestamp"]).timestamp()
            if ts >= cutoff:
                recent.append(v)
        return recent

    def to_metrics(self) -> dict[str, Any]:
        result = self._last_result
        service_metrics = self.service.get_metrics()
        last_result_at = (
            datetime.fromtimestamp(result.timestamp, tz=timezone.utc).isoformat() if result else None
        )
        return {
            "request_id": self.request_id,
            "service_type": self.service_type,
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "is_running": self._is_running,
            "target_coverage": self.target_coverage,
            "target_sample": self.target_sample,
            "target_freshness": self.target_freshness,
            "total_nodes": self.total_nodes,
            "assigned_node_count": self.assigned_node_count,
            "input_multiplier": self.input_multiplier,
            "cpu_max_percent": self.cpu_max_percent,
            "memory_max_percent": self.memory_max_percent,
            "execution_mode": self.execution_mode,
            "plan_version": self.plan_version,
            "leader_term": self.leader_term,
            "source": self.source,
            "iterations": self._iterations,
            "updated_at": self.updated_at,
            "last_result_at": last_result_at,
            "recent_violations": len(self.recent_violations()),
            "last_processing_time_ms": result.processing_time_ms if result else 0.0,
            "last_data_volume_bytes": result.data_volume_bytes if result else 0,
            "last_heatmap_cells": result.heatmap_cells if result else 0,
            "last_hotspots": result.hotspots_detected if result else 0,
            "sample_rate_applied": result.sample_rate_applied if result else None,
            "spatial_fidelity": result.spatial_fidelity if result else None,
            "hotspot_recall": result.hotspot_recall if result else None,
            "service_metrics": service_metrics,
        }

    @property
    def last_result(self) -> Optional[GeoHeatmapResult]:
        return self._last_result


class NodeState:
    """
    Holds runtime state for this node.

    Supports multiple analytics runtimes concurrently (multi-tenant).
    """

    def __init__(self, machine_id: Optional[str] = None):
        self.machine_id = machine_id or os.getenv("ARGOS_MACHINE_ID") or DEFAULT_MACHINE_ID
        self.node_id = _infer_node_id(self.machine_id)
        self.is_active: bool = True
        self.monitor = ArchitectureMonitor(SamplingConfig())
        self.last_configured_at: Optional[datetime] = None
        self.max_analytics_per_node: int = int(
            os.getenv("ARGOS_MAX_ANALYTICS_PER_NODE", str(max(3, os.cpu_count() or 1)))
        )
        self.execution_mode: str = os.getenv("ARGOS_RUNTIME_EXECUTION_MODE", "thread").strip().lower()
        self.current_plan_version: int = 0
        self.leader_term: int = 0
        self.plan_source: str = "local"
        self.plan_applied_at: Optional[str] = None
        max_proc_workers = int(os.getenv("ARGOS_RUNTIME_PROCESS_WORKERS", str(os.cpu_count() or 1)))
        self._process_pool: Optional[ProcessPoolExecutor] = None
        if self.execution_mode == "process":
            self._process_pool = ProcessPoolExecutor(max_workers=max(1, max_proc_workers))

        self._runtimes: dict[str, AnalyticsRuntime] = {}
        self._lock = Lock()

    def _normalize_plan(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if "analytics" in payload and isinstance(payload["analytics"], list):
            raw_items = payload["analytics"]
        elif payload.get("request_id"):
            raw_items = [payload]
        else:
            return []

        normalized: list[dict[str, Any]] = []
        default_plan_version = int(payload.get("plan_version") or 0)
        default_leader_term = int(payload.get("leader_term") or 0)
        default_source = str(payload.get("source") or "http")
        for item in raw_items:
            cfg = dict(item)
            cfg.setdefault("service_type", "geo_heatmap")
            cfg.setdefault("target_coverage", 0.5)
            cfg.setdefault("target_sample", 0.5)
            cfg.setdefault("target_freshness", 60.0)
            cfg.setdefault("assigned_node_count", 1)
            cfg.setdefault("total_nodes", 3)
            cfg.setdefault("tenant_id", "default")
            cfg.setdefault("priority", "standard")
            resource_limits = cfg.get("resource_limits") or {}
            cfg.setdefault("cpu_max_percent", resource_limits.get("cpu_max_percent", 80.0))
            cfg.setdefault("memory_max_percent", resource_limits.get("memory_max_percent", 85.0))
            cfg.setdefault("plan_version", default_plan_version)
            cfg.setdefault("leader_term", default_leader_term)
            cfg.setdefault("source", default_source)
            normalized.append(cfg)

        normalized.sort(key=lambda c: PRIORITY_RANK.get(str(c.get("priority", "standard")), 1))
        return normalized

    def _upsert_runtime(self, cfg: dict[str, Any]) -> None:
        request_id = str(cfg["request_id"])
        runtime = self._runtimes.get(request_id)
        if runtime is None:
            runtime = AnalyticsRuntime(
                request_id=request_id,
                service_type=str(cfg.get("service_type", "geo_heatmap")),
                node_id=self.node_id,
                tenant_id=str(cfg.get("tenant_id", "default")),
                priority=str(cfg.get("priority", "standard")),
                target_coverage=float(cfg.get("target_coverage", 0.5)),
                target_sample=float(cfg.get("target_sample", 0.5)),
                target_freshness=float(cfg.get("target_freshness", 60.0)),
                total_nodes=int(cfg.get("total_nodes", 3)),
                cpu_max_percent=float(cfg.get("cpu_max_percent", 80.0)),
                memory_max_percent=float(cfg.get("memory_max_percent", 85.0)),
                assigned_node_count=int(cfg.get("assigned_node_count", 1)),
                input_multiplier=max(1, int(cfg.get("input_multiplier", 1))),
                execution_mode=self.execution_mode,
                process_pool=self._process_pool,
            )
            self._runtimes[request_id] = runtime
        else:
            runtime.update_from_config(cfg)

        if self.is_active:
            runtime.start()

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Apply a node runtime plan.

        The payload can be:
        - a single configuration (flat fields), or
        - a node plan (`{"analytics": [ ... ]}`) with several request configurations.
        """
        plan = self._normalize_plan(payload)
        if not plan:
            raise ValueError("No request configuration provided")

        incoming_plan_version = max(int(cfg.get("plan_version", 0) or 0) for cfg in plan)
        incoming_term = max(int(cfg.get("leader_term", 0) or 0) for cfg in plan)
        incoming_source = str(plan[0].get("source", payload.get("source", "http")))
        if incoming_plan_version and incoming_plan_version < self.current_plan_version:
            raise RuntimeError(f"Stale plan version {incoming_plan_version} < current {self.current_plan_version}")
        if not incoming_plan_version:
            incoming_plan_version = self.current_plan_version + 1

        self.is_active = True
        self.last_configured_at = datetime.now(timezone.utc)
        self.current_plan_version = incoming_plan_version
        self.leader_term = incoming_term
        self.plan_source = incoming_source
        self.plan_applied_at = _now_iso()

        accepted = plan[: self.max_analytics_per_node]
        rejected = plan[self.max_analytics_per_node :]

        desired_ids = {str(cfg["request_id"]) for cfg in accepted}

        with self._lock:
            # Upsert accepted runtimes.
            for cfg in accepted:
                self._upsert_runtime(cfg)

            # Stop runtimes no longer in desired plan.
            for rid in list(self._runtimes.keys()):
                if rid not in desired_ids:
                    self._runtimes[rid].stop()
                    del self._runtimes[rid]

        # Update monitor service summary.
        self.monitor.set_service_metrics(
            ServiceMetrics(
                service_type="multi_analytics",
                is_active=self.is_active and bool(self._runtimes),
                sample_rate=0.0,
            )
        )

        return {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "active_request_ids": sorted(self._runtimes.keys()),
            "rejected_request_ids": [str(cfg["request_id"]) for cfg in rejected],
            "plan_version": self.current_plan_version,
        }

    def deactivate(self) -> None:
        self.is_active = False
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.stop()
        self.monitor.set_service_metrics(
            ServiceMetrics(
                service_type="multi_analytics",
                is_active=False,
                sample_rate=0.0,
            )
        )

    def activate(self) -> None:
        self.is_active = True
        with self._lock:
            for runtime in self._runtimes.values():
                runtime.start()

    def shutdown(self) -> None:
        """Gracefully stop runtimes and background executors."""
        self.deactivate()
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=False, cancel_futures=True)
            self._process_pool = None

    def stop_analytic(self, analytic_id: str) -> bool:
        with self._lock:
            runtime = self._runtimes.pop(analytic_id, None)
        if runtime is None:
            return False
        runtime.stop()
        return True

    def get_heatmap_result(self, analytic_id: str) -> Optional[GeoHeatmapResult]:
        runtime = self._runtimes.get(analytic_id)
        if runtime is None:
            return None
        return runtime.last_result

    def compute_heatmap_sync(self, request_id: str) -> Optional[GeoHeatmapResult]:
        runtime = self._runtimes.get(request_id)
        if runtime is None:
            return None
        return runtime.compute_once()

    def list_analytics(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r.to_metrics() for r in self._runtimes.values()]

    def get_metrics(self) -> dict:
        sample = self.monitor.sample()
        analytics = self.list_analytics()
        first = analytics[0] if analytics else {}

        state_structured = {
            "compute": sample.state.compute.as_dict(),
            "data": sample.state.data.as_dict(),
            "network": sample.state.network.as_dict(),
            "service": sample.state.service.as_dict() if sample.state.service else {},
        }
        state_flat = sample.state.flatten()

        violations_recent = sum(int(a.get("recent_violations", 0)) for a in analytics)

        return {
            "machine_id": self.machine_id,
            "node_id": self.node_id,
            "timestamp": _now_iso(),
            "is_active": self.is_active,
            "active_analytics_count": len(analytics),
            "active_workers": len(analytics),
            "max_analytics_per_node": self.max_analytics_per_node,
            "quota_usage": len(analytics) / max(self.max_analytics_per_node, 1),
            "runtime_mode": self.execution_mode,
            "current_plan_version": self.current_plan_version,
            "leader_term": self.leader_term,
            "plan_source": self.plan_source,
            "plan_applied_at": self.plan_applied_at,
            # structured payload
            "state": state_structured,
            "state_flat": state_flat,
            "analytics": analytics,
            "violations": {
                "recent_total": violations_recent,
            },
            # flat fields (first runtime, if any)
            "request_id": first.get("request_id"),
            "service_type": first.get("service_type", "none"),
            "target_sample": first.get("target_sample", 0.0),
            "target_freshness": first.get("target_freshness", 0.0),
            "target_coverage": first.get("target_coverage", 0.0),
            "total_nodes": first.get("total_nodes", 0),
            "heatmap_service": {
                "last_heatmap_cells": first.get("last_heatmap_cells", 0),
                "last_hotspots": first.get("last_hotspots", 0),
                "last_processing_time_ms": first.get("last_processing_time_ms", 0.0),
                "last_data_volume_bytes": first.get("last_data_volume_bytes", 0),
            },
        }


# Module-level singleton
node_state = NodeState()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        node_state.shutdown()


app = FastAPI(
    title="Node Status API",
    version="0.4.0",
    description="ARGOS Edge Node API with multi-tenant analytics runtimes",
    lifespan=_lifespan,
)

_STATUS_API_TOKEN = os.getenv("STATUS_API_TOKEN")


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if _STATUS_API_TOKEN and x_api_key != _STATUS_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================================================================
# Endpoints
# ============================================================================


class ConfigureAnalytics(BaseModel):
    request_id: str
    service_type: str = "geo_heatmap"
    target_coverage: float = 0.5
    target_sample: float = 0.5
    target_freshness: float = 60.0
    assigned_node_count: int = 1
    total_nodes: int = 3
    tenant_id: str = "default"
    priority: str = "standard"
    cpu_max_percent: float = 80.0
    memory_max_percent: float = 85.0
    input_multiplier: int = 1
    resource_limits: Optional[dict[str, float]] = None
    plan_version: int = 0
    leader_term: int = 0
    source: str = "http"


class ConfigureRequest(BaseModel):
    """Configure payload: a node plan (analytics list) or a single configuration."""

    # node plan
    analytics: Optional[list[ConfigureAnalytics]] = None

    # single configuration
    request_id: Optional[str] = None
    service_type: Optional[str] = None
    target_coverage: Optional[float] = None
    target_sample: Optional[float] = None
    target_freshness: Optional[float] = None
    assigned_node_count: int = 1
    total_nodes: int = 3
    tenant_id: str = "default"
    priority: str = "standard"
    cpu_max_percent: float = 80.0
    memory_max_percent: float = 85.0
    input_multiplier: int = 1
    resource_limits: Optional[dict[str, float]] = None
    plan_version: int = 0
    leader_term: int = 0
    source: str = "http"


@app.get("/metrics")
def get_metrics(_: None = Depends(_require_api_key)):
    """Return current node metrics (multi-tenant aware)."""
    return node_state.get_metrics()


@app.post("/configure")
def post_configure(config: ConfigureRequest, _: None = Depends(_require_api_key)):
    """Receive and apply a node plan from orchestrator."""
    payload = config.model_dump(exclude_none=True)
    try:
        result = node_state.configure(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "status": "configured",
        "is_active": node_state.is_active,
        "accepted": result["accepted"],
        "rejected": result["rejected"],
        "active_request_ids": result["active_request_ids"],
        "rejected_request_ids": result["rejected_request_ids"],
        "plan_version": result["plan_version"],
    }


@app.post("/deactivate")
def post_deactivate(_: None = Depends(_require_api_key)):
    node_state.deactivate()
    return {
        "status": "deactivated",
        "is_active": node_state.is_active,
    }


@app.post("/activate")
def post_activate(_: None = Depends(_require_api_key)):
    node_state.activate()
    return {
        "status": "activated",
        "is_active": node_state.is_active,
    }


# ============================================================================
# Analytics Endpoints
# ============================================================================


@app.get("/analytics/{analytic_id}")
def get_analytics(analytic_id: str, _: None = Depends(_require_api_key)):
    result = node_state.get_heatmap_result(analytic_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analytic not found or no results available")
    return result.to_dict()


@app.post("/analytics/{analytic_id}/compute")
def compute_analytics(analytic_id: str, _: None = Depends(_require_api_key)):
    result = node_state.compute_heatmap_sync(analytic_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analytic not running")
    return result.to_dict()


@app.delete("/analytics/{analytic_id}")
def delete_analytics(analytic_id: str, _: None = Depends(_require_api_key)):
    stopped = node_state.stop_analytic(analytic_id)
    return {
        "status": "stopped" if stopped else "not_running",
        "analytic_id": analytic_id,
    }


@app.get("/analytics")
def list_analytics(_: None = Depends(_require_api_key)):
    return {
        "node_id": node_state.node_id,
        "machine_id": node_state.machine_id,
        "is_active": node_state.is_active,
        "analytics": node_state.list_analytics(),
    }


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "node_id": node_state.node_id,
        "machine_id": node_state.machine_id,
        "is_active": node_state.is_active,
        "timestamp": _now_iso(),
    }
