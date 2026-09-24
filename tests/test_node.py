"""Tests for the status API endpoints."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


from fastapi.testclient import TestClient  # noqa: E402


class TestStatusAPI(unittest.TestCase):
    """Tests for status API endpoints."""

    def setUp(self):
        """Set up test client."""
        # Clear any token requirement for tests
        with patch.dict(os.environ, {"STATUS_API_TOKEN": ""}, clear=False):
            from argos.node import app

            self.client = TestClient(app)

    def test_configure_rejects_stale_plan_version(self):
        """POST /configure should reject an older plan version once a newer one was applied."""
        with patch.dict(os.environ, {"STATUS_API_TOKEN": ""}, clear=False):
            from argos import node

            importlib.reload(node)
            client = TestClient(node.app)

            ok = client.post(
                "/configure",
                json={
                    "analytics": [
                        {
                            "request_id": "req-1",
                            "service_type": "geo_heatmap",
                            "plan_version": 3,
                            "leader_term": 2,
                            "source": "orch-a",
                        }
                    ]
                },
            )
            self.assertEqual(ok.status_code, 200)

            stale = client.post(
                "/configure",
                json={
                    "analytics": [
                        {
                            "request_id": "req-1",
                            "service_type": "geo_heatmap",
                            "plan_version": 2,
                            "leader_term": 2,
                            "source": "orch-a",
                        }
                    ]
                },
            )
            self.assertEqual(stale.status_code, 409)

    def test_metrics_expose_plan_and_runtime_metadata(self):
        """GET /metrics should expose runtime mode and plan-version metadata."""
        with patch.dict(os.environ, {"STATUS_API_TOKEN": ""}, clear=False):
            from argos import node

            importlib.reload(node)
            client = TestClient(node.app)

            response = client.post(
                "/configure",
                json={
                    "analytics": [
                        {
                            "request_id": "req-meta",
                            "service_type": "geo_heatmap",
                            "plan_version": 5,
                            "leader_term": 4,
                            "source": "orch-b",
                        }
                    ]
                },
            )
            self.assertEqual(response.status_code, 200)

            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            data = metrics.json()
            self.assertIn(data["runtime_mode"], {"thread", "process"})
            self.assertGreaterEqual(data["max_analytics_per_node"], 1)
            self.assertEqual(data["current_plan_version"], 5)
            self.assertEqual(data["leader_term"], 4)
            self.assertEqual(data["plan_source"], "orch-b")

    def test_configure_preserves_input_multiplier_for_new_runtime(self):
        with patch.dict(os.environ, {"STATUS_API_TOKEN": ""}, clear=False):
            from argos import node

            importlib.reload(node)
            client = TestClient(node.app)
            response = client.post(
                "/configure",
                json={
                    "analytics": [
                        {
                            "request_id": "req-input-scale",
                            "service_type": "geo_heatmap",
                            "input_multiplier": 8,
                        }
                    ]
                },
            )

            self.assertEqual(response.status_code, 200)
            metrics = client.get("/metrics").json()
            runtime = next(item for item in metrics["analytics"] if item["request_id"] == "req-input-scale")
            self.assertEqual(runtime["input_multiplier"], 8)
            self.assertEqual(runtime["service_metrics"]["input_multiplier"], 8)

    def test_process_heatmap_compute_uses_payload_config_and_cache(self):
        """Process-mode worker computation should preserve runtime config and cache services."""
        from argos import node

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "SB_User0.json").write_text(
                json.dumps(
                    [
                        {"lat": 37.378833, "lng": -5.976987, "timestamp": "2026-04-29T00:00:00+00:00"},
                        {"lat": 37.378834, "lng": -5.976988, "timestamp": "2026-04-29T00:00:01+00:00"},
                    ]
                ),
                encoding="utf-8",
            )

            node._PROCESS_SERVICES.clear()
            payload = {
                "request_id": "req-process",
                "compute_request_id": "req-process_0",
                "node_id": 1,
                "total_nodes": 1,
                "target_sample": 0.5,
                "resolution": 3,
                "dataset_path": str(dataset),
                "query_params": {"latitude": 37.378833, "longitude": -5.976987, "radius": 100},
            }

            first = node._compute_geo_heatmap_once(payload)
            second_payload = dict(payload)
            second_payload["compute_request_id"] = "req-process_1"
            second_payload["target_sample"] = 1.0
            second = node._compute_geo_heatmap_once(second_payload)

            self.assertEqual(len(node._PROCESS_SERVICES), 1)
            self.assertEqual(first["request_id"], "req-process_0")
            self.assertEqual(first["query_params"]["radius"], 100)
            self.assertEqual(first["sample_rate_applied"], 0.5)
            self.assertEqual(second["sample_rate_applied"], 1.0)

    def test_runtime_wakes_only_when_workload_configuration_changes(self):
        from argos import node

        runtime = node.AnalyticsRuntime(
            request_id="wake-test",
            service_type="geo_heatmap",
            node_id=1,
            target_sample=0.5,
            target_freshness=30.0,
            input_multiplier=2,
        )
        runtime.update_from_config(
            {
                "target_sample": 0.5,
                "target_freshness": 30.0,
                "total_nodes": 3,
                "input_multiplier": 2,
            }
        )
        self.assertFalse(runtime._config_event.is_set())

        runtime.update_from_config({"input_multiplier": 8})

        self.assertTrue(runtime._config_event.is_set())
        self.assertEqual(runtime.service.input_multiplier, 8)


class TestStatusAPIWithAuth(unittest.TestCase):
    """Tests for status API with authentication."""

    def test_unauthorized_without_token(self):
        """Request without token should be rejected when token is required."""
        with patch.dict(os.environ, {"STATUS_API_TOKEN": "secret-token"}):
            # Need to reimport to pick up new token
            import importlib

            from argos import node

            importlib.reload(node)

            client = TestClient(node.app)
            response = client.get("/metrics")

            self.assertEqual(response.status_code, 401)

    def test_authorized_with_correct_token(self):
        """Request with correct token should be processed."""
        with patch.dict(os.environ, {"STATUS_API_TOKEN": "secret-token"}):
            import importlib

            from argos import node

            importlib.reload(node)

            client = TestClient(node.app)
            response = client.get("/metrics", headers={"X-API-Key": "secret-token"})

            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
