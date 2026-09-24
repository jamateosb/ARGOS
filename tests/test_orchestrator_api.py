# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Integration tests for the Orchestrator REST API."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestOrchestratorAPI:
    """Tests for the orchestrator API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        # Import here to avoid module-level side effects
        from argos.orchestrator.api import create_app

        app = create_app()
        return TestClient(app)

    def test_cluster_status_no_auth_required(self, client):
        """GET /cluster/status should work without authentication."""
        response = client.get("/cluster/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "active_nodes" in data
        assert "sleeping_nodes" in data
        assert "active_requests" in data
        assert "orchestrator_running" in data
        assert "control_plane_enabled" in data
        assert "is_control_plane_leader" in data

    def test_submit_job_valid(self, client):
        """POST /submit-job should accept valid job requests."""
        response = client.post(
            "/submit-job",
            json={
                "service_type": "heatmap",
                "coverage_min": 0.3,
                "coverage_max": 0.7,
                "sample_min": 0.2,
                "sample_max": 0.6,
                "freshness_min": 30.0,
                "freshness_max": 120.0,
                "algorithm": "qlearning",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["reason"] == "insufficient_capacity"
        assert data["service_type"] == "heatmap"
        assert data["algorithm"] == "qlearning"
        assert "request_id" in data
        assert 0.3 <= data["effective_coverage"] <= 0.7
        assert 0.2 <= data["effective_sample"] <= 0.6
        assert 30.0 <= data["effective_freshness"] <= 120.0

    def test_submit_job_defaults_to_benchmark_selected_algorithm(self, tmp_path, monkeypatch):
        """POST /submit-job should resolve auto to the current benchmark winner."""
        ranking = tmp_path / "model_ranking_ci.csv"
        ranking.write_text(
            "runtime,mode,algorithm,n,reward_mean\nprocess,tuned,ppo,5,25.9\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(ranking))

        from argos.orchestrator.api import create_app

        response = TestClient(create_app()).post("/submit-job", json={"service_type": "heatmap"})

        assert response.status_code == 200
        data = response.json()
        assert data["algorithm"] == "ppo"

    def test_submit_job_auto_reads_latest_benchmark_ranking(self, tmp_path, monkeypatch):
        """POST /submit-job should not hardcode one algorithm when auto is used."""
        ranking = tmp_path / "model_ranking_ci.csv"
        ranking.write_text(
            "runtime,mode,algorithm,n,reward_mean\nprocess,tuned,dqn,5,31.0\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(ranking))

        from argos.orchestrator.api import create_app

        response = TestClient(create_app()).post("/submit-job", json={"service_type": "heatmap"})

        assert response.status_code == 200
        assert response.json()["algorithm"] == "dqn"

    def test_submit_job_invalid_algorithm(self, client):
        """POST /submit-job should reject unsupported RL algorithms."""
        response = client.post(
            "/submit-job",
            json={
                "service_type": "heatmap",
                "algorithm": "bogus",
            },
        )
        assert response.status_code == 400
        assert "algorithm" in response.json()["detail"]

    def test_profile_policy_map_requires_known_profile_and_tracks_queued_load(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Live frozen evaluation must bind each request to its profile policy."""
        import json

        from argos.orchestrator.api import OrchestratorAPI

        policy = tmp_path / "policy.json"
        policy.write_text("{}", encoding="utf-8")
        monkeypatch.setenv(
            "ARGOS_LIVE_FROZEN_POLICY_MAP",
            json.dumps({"known-profile": str(policy)}),
        )
        monkeypatch.setenv("ARGOS_LIVE_FROZEN_ALGORITHM", "qlearning")
        api = OrchestratorAPI()
        mapped_client = TestClient(api.app)

        missing = mapped_client.post("/submit-job", json={"service_type": "heatmap"})
        assert missing.status_code == 400
        assert "profile_name" in missing.json()["detail"]

        queued = mapped_client.post(
            "/submit-job",
            json={"service_type": "heatmap", "profile_name": "known-profile"},
        )
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert queued.json()["profile_name"] == "known-profile"
        assert queued.json()["frozen_policy_required"] is True
        assert queued.json()["frozen_policy_loaded"] is False

        cancelled = mapped_client.delete(f"/job/{queued.json()['request_id']}")
        assert cancelled.status_code == 200

    def test_submit_job_invalid_coverage_range(self, client):
        """POST /submit-job should reject invalid coverage range."""
        response = client.post(
            "/submit-job",
            json={
                "coverage_min": 0.8,
                "coverage_max": 0.3,  # min > max
            },
        )
        assert response.status_code == 400
        assert "coverage_min" in response.json()["detail"]

    def test_submit_job_invalid_sample_range(self, client):
        """POST /submit-job should reject invalid sample range."""
        response = client.post(
            "/submit-job",
            json={
                "sample_min": 0.9,
                "sample_max": 0.1,  # min > max
            },
        )
        assert response.status_code == 400
        assert "sample_min" in response.json()["detail"]

    def test_submit_job_invalid_freshness_range(self, client):
        """POST /submit-job should reject invalid freshness range."""
        response = client.post(
            "/submit-job",
            json={
                "freshness_min": 120.0,
                "freshness_max": 30.0,  # min > max
            },
        )
        assert response.status_code == 400
        assert "freshness_min" in response.json()["detail"]

    def test_get_job_not_found(self, client):
        """GET /job/{id} should return 404 for unknown jobs."""
        response = client.get("/job/nonexistent-id")
        assert response.status_code == 404

    def test_get_job_status(self, client):
        """GET /job/{id} should return job status after submission."""
        # First submit a job
        submit_response = client.post(
            "/submit-job",
            json={"service_type": "heatmap", "algorithm": "qlearning"},
        )
        request_id = submit_response.json()["request_id"]

        # Then get its status
        response = client.get(f"/job/{request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == request_id
        assert data["service_type"] == "heatmap"
        assert data["algorithm"] == "qlearning"
        assert data["policy_version"].startswith("qlearning-v")
        assert isinstance(data["effective_assignments"], list)
        assert isinstance(data["request_metrics"], dict)
        # No registered nodes in this fixture, so the job is queued until capacity appears.
        assert data["status"] == "queued"

    def test_cancel_job(self, client):
        """DELETE /job/{id} should cancel a job."""
        # First submit a job
        submit_response = client.post(
            "/submit-job",
            json={"service_type": "heatmap"},
        )
        request_id = submit_response.json()["request_id"]

        # Then cancel it
        response = client.delete(f"/job/{request_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

        # Verify it's gone
        response = client.get(f"/job/{request_id}")
        assert response.status_code == 404

    def test_cancel_job_not_found(self, client):
        """DELETE /job/{id} should return 404 for unknown jobs."""
        response = client.delete("/job/nonexistent-id")
        assert response.status_code == 404

    def test_register_node(self, client):
        """POST /nodes/register should register a new node."""
        response = client.post(
            "/nodes/register",
            json={
                "node_id": "test-node-1",
                "endpoint": "http://node-1.example:8000",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "registered"
        assert response.json()["node_id"] == "test-node-1"

    def test_list_nodes(self, client):
        """GET /cluster/nodes should list registered nodes."""
        # Register a node first
        client.post(
            "/nodes/register",
            json={
                "node_id": "test-node-2",
                "endpoint": "http://node-2.example:8000",
            },
        )

        response = client.get("/cluster/nodes")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        node_ids = [n["node_id"] for n in data["nodes"]]
        assert "test-node-2" in node_ids
        node = next(n for n in data["nodes"] if n["node_id"] == "test-node-2")
        assert "quota_usage" in node
        assert "active_workers" in node
        assert "max_analytics_per_node" in node
        assert "violations_recent" in node
        assert "runtime_mode" in node
        assert "current_plan_version" in node

    def test_control_plane_endpoint(self, client):
        """GET /cluster/control-plane should return leader-election state."""
        response = client.get("/cluster/control-plane")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "node_id" in data
        assert "is_leader" in data
        assert "leader_term" in data
        assert "plan_version" in data
        assert "source" in data
        assert "stale_peers" in data
        assert "capability_sync_status" in data

    def test_unregister_node(self, client):
        """DELETE /nodes/{id} should unregister a node."""
        # Register a node first
        client.post(
            "/nodes/register",
            json={
                "node_id": "test-node-3",
                "endpoint": "http://node-3.example:8000",
            },
        )

        # Unregister it
        response = client.delete("/nodes/test-node-3")
        assert response.status_code == 200
        assert response.json()["status"] == "unregistered"

    def test_experiments_endpoint(self, client):
        """GET /experiments should return experiment info."""
        response = client.get("/experiments")
        assert response.status_code == 200
        data = response.json()
        assert "output_path" in data
        assert "iteration_count" in data


class TestOrchestratorAPIWithAuth:
    """Tests for API authentication."""

    @pytest.fixture
    def client_with_auth(self):
        """Create a test client with authentication enabled."""
        with patch.dict(os.environ, {"ORCHESTRATOR_API_TOKEN": "test-secret-token"}):
            # Re-import to pick up the new env var
            import importlib

            import argos.orchestrator.api

            importlib.reload(argos.orchestrator.api)
            app = argos.orchestrator.api.create_app()
            yield TestClient(app)

    @pytest.fixture
    def auth_header(self):
        """Return the correct auth header."""
        return {"X-API-Key": "test-secret-token"}

    def test_unauthorized_without_token(self, client_with_auth):
        """Endpoints should reject requests without token when auth is enabled."""
        response = client_with_auth.post(
            "/submit-job",
            json={"service_type": "heatmap"},
        )
        assert response.status_code == 401

    def test_authorized_with_correct_token(self, client_with_auth, auth_header):
        """Endpoints should accept requests with correct token."""
        response = client_with_auth.post(
            "/submit-job",
            json={"service_type": "heatmap"},
            headers=auth_header,
        )
        assert response.status_code == 200

    def test_cluster_status_no_auth_even_when_enabled(self, client_with_auth):
        """GET /cluster/status should work without auth (health check)."""
        response = client_with_auth.get("/cluster/status")
        assert response.status_code == 200


class TestJobWithNodes:
    """Tests for job submission with registered nodes."""

    @pytest.fixture
    def client_with_nodes(self):
        """Create a client and register some nodes."""
        # Ensure auth is disabled (previous tests may have enabled it)
        with patch.dict(os.environ, {"ORCHESTRATOR_API_TOKEN": ""}, clear=False):
            import importlib

            import argos.orchestrator.api

            importlib.reload(argos.orchestrator.api)
            app = argos.orchestrator.api.create_app()
            client = TestClient(app)

            # Register 3 nodes
            for i in range(3):
                client.post(
                    "/nodes/register",
                    json={
                        "node_id": f"node-{i}",
                        "endpoint": f"http://192.168.1.{10+i}:8000",
                    },
                )

            yield client

    def test_job_assigns_nodes_based_on_coverage(self, client_with_nodes):
        """Jobs should assign nodes based on coverage percentage."""
        response = client_with_nodes.post(
            "/submit-job",
            json={
                "service_type": "heatmap",
                "coverage_min": 0.6,
                "coverage_max": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # With 3 nodes and ~65% coverage, should assign ~2 nodes
        assert data["assigned_nodes"] >= 1
        assert data["assigned_nodes"] <= 3

    def test_job_status_shows_assigned_nodes(self, client_with_nodes):
        """Job status should show which nodes are assigned."""
        submit_response = client_with_nodes.post(
            "/submit-job",
            json={"service_type": "heatmap", "coverage_min": 0.5, "coverage_max": 0.9},
        )
        request_id = submit_response.json()["request_id"]

        status_response = client_with_nodes.get(f"/job/{request_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert isinstance(data["assigned_nodes"], list)

    def test_submit_response_reports_actual_assignment_when_capacity_exhausted(self):
        """POST /submit-job should not report desired nodes as actual placement."""
        with patch.dict(os.environ, {"ORCHESTRATOR_API_TOKEN": ""}, clear=False):
            import importlib

            import argos.orchestrator.api

            importlib.reload(argos.orchestrator.api)
            app = argos.orchestrator.api.create_app()
            client = TestClient(app)

            client.post("/nodes/register", json={"node_id": "node-1", "endpoint": "http://node-1.example:8000"})

            payload = {
                "service_type": "heatmap",
                "coverage_min": 1.0,
                "coverage_max": 1.0,
                "placement_limits": {"max_jobs_per_node": 1},
            }
            first = client.post("/submit-job", json=payload)
            second = client.post("/submit-job", json=payload)

            assert first.status_code == 200
            assert first.json()["assigned_nodes"] == 1
            assert second.status_code == 200
            assert second.json()["assigned_nodes"] == 0
            assert second.json()["status"] == "queued"
            assert second.json()["reason"] == "insufficient_capacity"

            violations = client.get(f"/history/slo-violations?request_id={second.json()['request_id']}")
            assert violations.status_code == 200
            assert violations.json()["violations"] == []
