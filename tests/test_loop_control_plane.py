"""Tests for orchestration loop control-plane and budget behavior."""

import asyncio

import pytest

from argos.domain.requests import AnalyticsRequest
from argos.orchestrator.loop import (
    NodeInfo,
    OrchestrationLoop,
    OrchestrationLoopConfig,
    breaches_lower_bound,
    breaches_upper_bound,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_rl_decision_uses_only_assigned_node_metrics():
    """Each request should pass only its assigned-node metrics to RL."""
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
            enable_rl=True,
        )
    )

    request = AnalyticsRequest(request_id="req-1")
    config = request.midpoint_config()
    config.assigned_node_count = 1

    loop._active_requests[request.request_id] = request
    loop._effective_configs[request.request_id] = config
    loop._nodes["node-1"] = NodeInfo(
        node_id="node-1",
        endpoint="http://node-1",
        is_active=True,
        assigned_request_ids=[request.request_id],
    )
    loop._nodes["node-2"] = NodeInfo(
        node_id="node-2",
        endpoint="http://node-2",
        is_active=True,
        assigned_request_ids=[],
    )

    captured = {}

    def fake_make_decision(request_id, req, cfg, node_metrics, current_coverage):
        captured["node_ids"] = set(node_metrics.keys())
        return None, None

    loop._make_rl_decision = fake_make_decision

    node_metrics = {
        "node-1": {"state": {"compute": {"cpu_utilization": 55.0, "memory_utilization": 60.0}}},
        "node-2": {"state": {"compute": {"cpu_utilization": 99.0, "memory_utilization": 99.0}}},
    }

    loop._evaluate_and_adjust(node_metrics)
    assert captured["node_ids"] == {"node-1"}


def test_observed_node_plan_version_advances_local_counter():
    """A replacement leader must not push stale plan versions after polling nodes."""
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=True,
        )
    )

    loop._observe_node_plan_version(
        {
            "node_id": "node-a",
            "current_plan_version": 7,
            "plan_source": "previous-leader",
        }
    )
    assert loop._plan_version == 7
    assert loop._leader_election.snapshot()["plan_version"] == 7

    loop._observe_node_plan_version({"node_id": "node-b", "current_plan_version": 5})
    assert loop._plan_version == 7


def test_node_advertised_capacity_overrides_orchestrator_fallback_limit():
    """Placement should honor node runtime capacity exposed by /metrics."""
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
            enable_rl=False,
            max_jobs_per_node=1,
        )
    )
    loop.register_node("node-1", "http://node-1")
    loop._nodes["node-1"].last_metrics = {"max_analytics_per_node": 2}

    first = AnalyticsRequest(request_id="req-1", coverage_range=(1.0, 1.0))
    second = AnalyticsRequest(request_id="req-2", coverage_range=(1.0, 1.0))

    loop.submit_request(first)
    loop.submit_request(second)

    assert loop._nodes["node-1"].assigned_request_ids == ["req-1", "req-2"]


@pytest.mark.anyio
async def test_iteration_budget_skips_decision_phase():
    """If polling consumes the budget, decision/push should be skipped."""
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
            iteration_budget_ms=5.0,
        )
    )

    async def fake_poll():
        await asyncio.sleep(0.02)  # 20ms > 5ms budget
        return {}

    called = {"evaluate": False}

    def fake_evaluate(_metrics):
        called["evaluate"] = True
        return {}, 0

    loop._poll_all_nodes = fake_poll
    loop._evaluate_and_adjust = fake_evaluate

    metrics = await loop.run_once()
    assert called["evaluate"] is False
    assert metrics.errors_count >= 1


def test_contract_bounds_ignore_floating_point_representation_error():
    """Values one ULP outside a bound are representation error, not breaches.

    Regression test: controllers saturate at the extremes of their action space,
    so effective configurations land exactly on contract bounds. Arithmetic then
    yields values such as 0.6999999999999998 against a 0.7 bound. Comparing those
    strictly recorded thousands of phantom SLO violations, which fed both the
    reported telemetry and the recent-violation state dimension.
    """
    assert not breaches_lower_bound(0.6999999999999998, 0.7)
    assert not breaches_lower_bound(0.3499999999999999, 0.35)
    assert not breaches_upper_bound(0.7000000000000002, 0.7)
    assert not breaches_lower_bound(0.7, 0.7)
    assert not breaches_upper_bound(0.7, 0.7)
    # A zero bound must still get a usable tolerance instead of collapsing.
    assert not breaches_lower_bound(0.0, 0.0)


def test_contract_bounds_still_detect_genuine_breaches():
    """Real breaches are orders of magnitude above the tolerance and must fire."""
    assert breaches_lower_bound(0.69, 0.70)
    assert breaches_lower_bound(0.0, 0.3)
    assert breaches_upper_bound(45.0, 30.0)


def test_safety_guard_is_applied_as_action_mask_before_selection():
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
        )
    )
    request = AnalyticsRequest(request_id="guard-mask")
    node_metrics = {
        "node-1": {
            "state": {
                "compute": {
                    "cpu_utilization": request.resource_limits.cpu_max_percent,
                    "memory_utilization": 10.0,
                }
            }
        }
    }

    masked = loop._safety_masked_action_indices(
        request,
        tuple(range(7)),
        node_metrics,
    )

    assert masked == (0, 2, 4, 5)


def test_observed_quality_violations_skip_boundary_saturation():
    """The detector itself must not persist a violation for a saturated bound."""
    recorded = []

    class _RecordingPersistence:
        def save_slo_violation(self, violation):
            recorded.append(violation)

    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
        )
    )
    loop._persistence = _RecordingPersistence()

    request = AnalyticsRequest(request_id="req-boundary")
    lower_bound = request.sample_range[0]

    loop._persist_observed_quality_violations(
        request=request,
        actual_sample=lower_bound - 5.551115123125783e-17,
        freshness_age_s=None,
    )
    assert recorded == []

    loop._persist_observed_quality_violations(
        request=request,
        actual_sample=lower_bound - 0.1,
        freshness_age_s=None,
    )
    assert len(recorded) == 1
    assert recorded[0].metric == "sample_rate"


def test_recent_slo_window_decrements_after_clean_decisions():
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
            slo_window_decisions=20,
        )
    )

    loop._record_slo_decision_epoch("req-window", True)
    for _ in range(19):
        loop._record_slo_decision_epoch("req-window", False)
    assert loop._recent_slo_violation_count("req-window") == 1

    loop._record_slo_decision_epoch("req-window", False)
    assert loop._recent_slo_violation_count("req-window") == 0


def test_coverage_violation_records_upper_contract_breach():
    recorded = []

    class _RecordingPersistence:
        def save_slo_violation(self, violation):
            recorded.append(violation)

    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_logging=False,
            enable_persistence=False,
            enable_control_plane=False,
        )
    )
    loop._persistence = _RecordingPersistence()
    request = AnalyticsRequest(
        request_id="coverage-upper",
        coverage_range=(0.33, 0.67),
    )
    config = request.midpoint_config()

    assert loop._persist_coverage_violation(
        request=request,
        config=config,
        current_coverage=1.0,
        active_assigned=3,
        reason="test",
    )
    assert recorded[0].metric == "coverage"
    assert recorded[0].limit_value == 0.67
    assert recorded[0].details["bound"] == "maximum"
