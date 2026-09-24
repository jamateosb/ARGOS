# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Control-plane helpers for AWS v1:
- Node capability benchmark (CPU/RAM/network proxy)
- Leader election by score + heartbeat lease
- Optional NATS event bus publisher/subscriber
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import psutil

from argos.common.utils import utc_now_iso
from argos.orchestrator.persistence import LeaderElectionEvent, PersistenceManager

logger = logging.getLogger(__name__)


@dataclass
class NodeCapability:
    node_id: str
    hostname: str
    cpu_cores: int
    memory_gb: float
    network_mbps: float
    benchmark_score: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NatsEventBus:
    """Thin wrapper around nats-py (optional runtime dependency)."""

    def __init__(self, url: str):
        self.url = url
        self._nc = None

    async def connect(self) -> None:
        try:
            from nats.aio.client import Client as NatsClient  # type: ignore
        except Exception as exc:
            raise RuntimeError("nats-py is required for NATS event bus") from exc

        self._nc = NatsClient()
        await self._nc.connect(servers=[self.url], name="argos-control-plane")

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.close()

    async def publish(self, subject: str, payload: dict[str, Any]) -> None:
        if self._nc is None:
            return
        await self._nc.publish(subject, json.dumps(payload).encode("utf-8"))

    async def subscribe(self, subject: str, callback: Callable[[dict[str, Any]], Any]) -> None:
        if self._nc is None:
            return

        async def _handler(msg):
            try:
                data = json.loads(msg.data.decode("utf-8"))
            except Exception:
                return
            maybe_coro = callback(data)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

        await self._nc.subscribe(subject, cb=_handler)


class LeaderElectionService:
    """
    Score-based leader election with lease + heartbeat.

    The highest benchmark score wins while its heartbeat is fresh.
    """

    def __init__(
        self,
        node_id: str,
        persistence: Optional[PersistenceManager] = None,
        lease_seconds: int = 15,
    ):
        self.node_id = node_id
        self.lease_seconds = lease_seconds
        self.persistence = persistence
        self._leader_id: Optional[str] = None
        self._leader_term: int = 0
        self._scores: dict[str, float] = {}
        self._heartbeats: dict[str, float] = {}
        self._last_capability: Optional[NodeCapability] = None
        self._peer_sources: dict[str, str] = {}
        self._stale_peers: set[str] = set()
        self._plan_version: int = 0
        self._plan_source_node: Optional[str] = None
        self._capability_sync_status: str = "local_only"

    def benchmark_local_node(self) -> NodeCapability:
        cpu_cores = psutil.cpu_count(logical=True) or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        net = psutil.net_if_stats()
        # Proxy using max NIC speed when available.
        nic_speeds = [max(0, s.speed or 0) for s in net.values()]
        network_mbps = float(max(nic_speeds) if nic_speeds else 100.0)

        # Weighted benchmark score.
        score = (cpu_cores * 0.5) + (memory_gb * 0.4) + (network_mbps / 1000.0 * 0.1)

        cap = NodeCapability(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            cpu_cores=cpu_cores,
            memory_gb=round(memory_gb, 2),
            network_mbps=round(network_mbps, 2),
            benchmark_score=round(score, 4),
            timestamp=utc_now_iso(),
        )
        self.register_score(self.node_id, cap.benchmark_score, heartbeat_time=time.time())
        self._last_capability = cap
        return cap

    def heartbeat(self, node_id: Optional[str] = None) -> None:
        rid = node_id or self.node_id
        self._heartbeats[rid] = time.time()
        if self.persistence:
            self.persistence.save_leader_event(
                LeaderElectionEvent(
                    timestamp=utc_now_iso(),
                    node_id=rid,
                    event_type="heartbeat",
                    score=self._scores.get(rid),
                    term=self._leader_term,
                    source_node=self.node_id,
                    plan_version=self._plan_version,
                )
            )

    def register_score(self, node_id: str, score: float, heartbeat_time: Optional[float] = None) -> None:
        """Register or update a node capability score."""
        self._scores[str(node_id)] = float(score)
        self._heartbeats[str(node_id)] = heartbeat_time if heartbeat_time is not None else time.time()

    def register_remote_heartbeat(
        self,
        node_id: str,
        score: Optional[float] = None,
        heartbeat_time: Optional[float] = None,
    ) -> None:
        """Register a heartbeat from a peer node."""
        rid = str(node_id)
        if score is not None:
            self._scores[rid] = float(score)
        self._heartbeats[rid] = heartbeat_time if heartbeat_time is not None else time.time()
        if rid != self.node_id:
            self._peer_sources[rid] = "nats"
            self._capability_sync_status = "synchronized"

    def register_plan_version(self, version: int, source_node: Optional[str] = None) -> bool:
        """Register the latest accepted plan version."""
        version = int(version)
        if version < self._plan_version:
            return False
        self._plan_version = version
        self._plan_source_node = source_node or self._plan_source_node or self.node_id
        return True

    def elect(self, now: Optional[float] = None) -> Optional[str]:
        now = now if now is not None else time.time()
        previous_leader = self._leader_id

        alive = {
            nid: score
            for nid, score in self._scores.items()
            if (now - self._heartbeats.get(nid, 0.0)) <= self.lease_seconds
        }
        self._stale_peers = {nid for nid in self._scores if nid not in alive and nid != self.node_id}
        if self._stale_peers and self._capability_sync_status == "synchronized":
            self._capability_sync_status = "degraded"

        if previous_leader and previous_leader not in alive and self.persistence:
            self.persistence.save_leader_event(
                LeaderElectionEvent(
                    timestamp=utc_now_iso(),
                    node_id=previous_leader,
                    event_type="lost",
                    score=self._scores.get(previous_leader),
                    term=self._leader_term,
                    previous_leader=previous_leader,
                    reason="heartbeat_expired",
                    source_node=self.node_id,
                    plan_version=self._plan_version,
                )
            )

        if not alive:
            self._leader_id = None
            return None

        leader = sorted(alive.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if leader != previous_leader and self.persistence:
            self._leader_term += 1
            event_type = "failover" if previous_leader else "elected"
            self.persistence.save_leader_event(
                LeaderElectionEvent(
                    timestamp=utc_now_iso(),
                    node_id=leader,
                    event_type=event_type,
                    score=alive[leader],
                    term=self._leader_term,
                    previous_leader=previous_leader,
                    reason="higher_score_or_tie_break",
                    source_node=self.node_id,
                    plan_version=self._plan_version,
                    details={"previous_leader": previous_leader},
                )
            )
        elif leader != previous_leader:
            self._leader_term += 1
        self._leader_id = leader
        return leader

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id

    @property
    def local_capability(self) -> Optional[NodeCapability]:
        return self._last_capability

    @property
    def leader_term(self) -> int:
        return self._leader_term

    def get_score(self, node_id: str) -> float:
        return float(self._scores.get(node_id, 0.0))

    def snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        """Get current leader-election state for diagnostics."""
        now = now if now is not None else time.time()
        alive = {
            nid: {
                "score": score,
                "last_heartbeat_age_s": max(0.0, now - self._heartbeats.get(nid, 0.0)),
                "alive": (now - self._heartbeats.get(nid, 0.0)) <= self.lease_seconds,
            }
            for nid, score in self._scores.items()
        }
        return {
            "node_id": self.node_id,
            "leader_id": self._leader_id,
            "leader_term": self._leader_term,
            "lease_seconds": self.lease_seconds,
            "plan_version": self._plan_version,
            "source": self._plan_source_node or self.node_id,
            "stale_peers": sorted(self._stale_peers),
            "capability_sync_status": self._capability_sync_status,
            "nodes": alive,
        }


def build_control_plane(node_id: str, persistence: Optional[PersistenceManager] = None) -> LeaderElectionService:
    """Factory for the score+heartbeat leader election service."""
    svc = LeaderElectionService(node_id=node_id, persistence=persistence)
    svc.benchmark_local_node()
    svc.heartbeat(node_id)
    svc.elect()
    return svc
