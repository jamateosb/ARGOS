"""Tests for control-plane leader election helpers."""

from argos.orchestrator.control_plane import LeaderElectionService


def test_leader_election_failover_on_stale_heartbeat():
    """Leader should fail over to the highest-score alive node."""
    svc = LeaderElectionService(node_id="node-a", persistence=None, lease_seconds=5)

    svc.register_score("node-a", 10.0, heartbeat_time=100.0)
    svc.register_score("node-b", 20.0, heartbeat_time=100.0)
    assert svc.elect(now=101.0) == "node-b"

    # node-b heartbeat expires; node-a is still alive.
    svc.register_remote_heartbeat("node-a", score=10.0, heartbeat_time=108.0)
    assert svc.elect(now=108.0) == "node-a"


def test_snapshot_reports_liveness_and_scores():
    """Snapshot should expose alive flags and heartbeat ages."""
    svc = LeaderElectionService(node_id="node-x", persistence=None, lease_seconds=10)
    svc.register_score("node-x", 11.5, heartbeat_time=200.0)
    svc.elect(now=205.0)

    snap = svc.snapshot()
    assert snap["node_id"] == "node-x"
    assert snap["leader_id"] == "node-x"
    assert snap["nodes"]["node-x"]["score"] == 11.5
    assert isinstance(snap["nodes"]["node-x"]["alive"], bool)
    assert snap["nodes"]["node-x"]["last_heartbeat_age_s"] >= 0.0


def test_leader_election_tie_break_is_deterministic_and_term_advances_on_change():
    """Equal scores should pick a stable leader and increment term only on leader change."""
    svc = LeaderElectionService(node_id="node-b", persistence=None, lease_seconds=5)

    svc.register_score("node-b", 10.0, heartbeat_time=100.0)
    svc.register_score("node-a", 10.0, heartbeat_time=100.0)
    assert svc.elect(now=101.0) == "node-a"
    assert svc.snapshot()["leader_term"] == 1

    svc.register_remote_heartbeat("node-a", score=10.0, heartbeat_time=102.0)
    assert svc.elect(now=102.0) == "node-a"
    assert svc.snapshot()["leader_term"] == 1

    svc.register_score("node-c", 11.0, heartbeat_time=103.0)
    assert svc.elect(now=103.0) == "node-c"
    assert svc.snapshot()["leader_term"] == 2


def test_snapshot_reports_stale_peers_and_plan_version():
    """Snapshot should expose stale peers, capability sync status, and current plan version."""
    svc = LeaderElectionService(node_id="node-a", persistence=None, lease_seconds=5)
    svc.register_score("node-a", 10.0, heartbeat_time=102.0)
    svc.register_score("node-b", 9.0, heartbeat_time=90.0)
    svc.register_plan_version(7, source_node="node-a")
    svc.elect(now=106.0)

    snap = svc.snapshot(now=106.0)
    assert snap["leader_term"] >= 1
    assert snap["plan_version"] == 7
    assert snap["source"] == "node-a"
    assert "node-b" in snap["stale_peers"]
    assert snap["capability_sync_status"] in {"synchronized", "degraded", "local_only"}
