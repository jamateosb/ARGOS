"""Admission-control and enriched RL-state tests."""

from pathlib import Path

from argos.domain.requests import AnalyticsRequest, PlacementLimits
from argos.orchestrator.loop import OrchestrationLoop, OrchestrationLoopConfig
from argos.orchestrator.persistence import JobStatus


class _ViolationSink:
    def __init__(self):
        self.violations = []

    def save_slo_violation(self, violation):
        self.violations.append(violation)


def _loop(enable_rl: bool = False) -> OrchestrationLoop:
    return OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_persistence=False,
            enable_logging=False,
            enable_rl=enable_rl,
            enable_control_plane=False,
            max_jobs_per_node=1,
        )
    )


def test_queued_requests_are_admitted_by_priority_when_capacity_frees():
    """Critical queued requests should be admitted before standard ones."""
    loop = _loop()
    loop.register_node("node-1", "http://node-1")

    first = AnalyticsRequest(
        request_id="first",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
        priority="standard",
    )
    standard = AnalyticsRequest(
        request_id="standard",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
        priority="standard",
    )
    critical = AnalyticsRequest(
        request_id="critical",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
        priority="critical",
    )

    loop.submit_request(first)
    loop.submit_request(standard)
    loop.submit_request(critical)

    assert loop.get_job_status(first.request_id) == JobStatus.ACCEPTED
    assert loop.get_job_status(standard.request_id) == JobStatus.QUEUED
    assert loop.get_job_status(critical.request_id) == JobStatus.QUEUED

    loop.cancel_request(first.request_id, reason="test_capacity_release")
    loop._retry_pending_requests()

    assert loop.get_job_status(critical.request_id) == JobStatus.ACCEPTED
    assert loop.get_job_status(standard.request_id) == JobStatus.QUEUED
    assert loop.nodes["node-1"].assigned_request_ids == ["critical"]


def test_same_priority_queue_preserves_arrival_order():
    """Same-priority queued requests should be retried FIFO."""
    loop = _loop()
    loop.register_node("node-1", "http://node-1")

    first = AnalyticsRequest(
        request_id="first",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )
    second = AnalyticsRequest(
        request_id="second",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )
    third = AnalyticsRequest(
        request_id="third",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )

    loop.submit_request(first)
    loop.submit_request(second)
    loop.submit_request(third)

    loop.cancel_request(first.request_id, reason="test_capacity_release")
    loop._retry_pending_requests()

    assert loop.get_job_status(second.request_id) == JobStatus.ACCEPTED
    assert loop.get_job_status(third.request_id) == JobStatus.QUEUED
    assert loop.nodes["node-1"].assigned_request_ids == ["second"]


def test_queued_request_loads_frozen_policy_when_admitted(tmp_path: Path):
    loop = _loop(enable_rl=True)
    loop.config.rl_seed = 17
    loop.register_node("node-1", "http://node-1")
    placement = PlacementLimits(max_jobs_per_node=1)
    first = AnalyticsRequest(
        request_id="policy-source",
        coverage_range=(1.0, 1.0),
        placement_limits=placement,
    )
    queued = AnalyticsRequest(
        request_id="policy-queued",
        coverage_range=(1.0, 1.0),
        placement_limits=placement,
    )

    loop.submit_request(first)
    source_agent = loop._rl_agents[first.request_id]
    source_agent.update(
        ("s0",),
        source_agent._actions[1],
        reward=0.8,
        next_state=("s1",),
        done=True,
    )
    expected_fingerprint = source_agent.policy_fingerprint()
    policy_path = tmp_path / "policy.json"
    source_agent.save(str(policy_path))

    loop.submit_request(queued)
    assert loop.get_job_status(queued.request_id) == JobStatus.QUEUED
    assert loop.load_rl_agent(queued.request_id, str(policy_path)) is None

    loop.cancel_request(first.request_id, reason="release")
    loop._retry_pending_requests()

    assert loop.get_job_status(queued.request_id) == JobStatus.ACCEPTED
    assert loop._rl_agents[queued.request_id].policy_fingerprint() == expected_fingerprint
    assert queued.request_id not in loop._pending_policy_loads
    loop.assert_rl_policy_unchanged(queued.request_id)


def test_readmission_preserves_existing_agent_policy():
    loop = _loop(enable_rl=True)
    loop.config.rl_seed = 23
    loop.register_node("node-1", "http://node-1")
    request = AnalyticsRequest(
        request_id="readmit-policy",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )
    config = loop.submit_request(request)
    agent = loop._rl_agents[request.request_id]
    agent.update(
        ("s0",),
        agent._actions[2],
        reward=0.5,
        next_state=("s1",),
        done=True,
    )
    fingerprint = agent.policy_fingerprint()

    loop._nodes["node-1"].assigned_request_ids.remove(request.request_id)
    loop._mark_queued(request, "test_readmission")
    loop._try_admit_or_queue(request, config, initial=False)

    assert loop._rl_agents[request.request_id] is agent
    assert loop._rl_agents[request.request_id].policy_fingerprint() == fingerprint


def test_request_agents_receive_distinct_reproducible_private_seeds():
    def seeded_loop():
        candidate = _loop(enable_rl=True)
        candidate.config.rl_seed = 31
        candidate.config.max_jobs_per_node = 2
        candidate.register_node("node-1", "http://node-1")
        for request_id in ("seed-a", "seed-b"):
            candidate.submit_request(
                AnalyticsRequest(
                    request_id=request_id,
                    coverage_range=(1.0, 1.0),
                    placement_limits=PlacementLimits(max_jobs_per_node=2),
                )
            )
        return candidate

    first = seeded_loop()
    second = seeded_loop()

    assert first._agent_seeds["seed-a"] != first._agent_seeds["seed-b"]
    assert first._agent_seeds == second._agent_seeds


def test_inactive_node_assignments_are_removed_before_rebalance():
    """Inactive nodes must not remain logical assignees for running jobs."""
    loop = _loop()
    loop.register_node("node-1", "http://node-1")
    loop.register_node("node-2", "http://node-2")
    loop.register_node("node-3", "http://node-3")
    request = AnalyticsRequest(
        request_id="node-failure",
        coverage_range=(0.34, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )

    loop.submit_request(request)
    assert loop.get_job_status(request.request_id) == JobStatus.ACCEPTED
    assert len(loop._request_assigned_nodes(request.request_id)) == 3

    failed_node = next(node for node in loop.nodes.values() if request.request_id in node.assigned_request_ids)
    loop._nodes[failed_node.node_id].is_active = False

    removed = loop._unassign_inactive_node_workloads()

    assert removed == 1
    assert request.request_id not in loop.nodes[failed_node.node_id].assigned_request_ids
    assert all(node.is_active for node in loop._request_assigned_nodes(request.request_id))


def test_rl_decision_state_includes_capacity_queue_latency_and_slo_context():
    """RL decisions should persist the richer operational state features."""
    loop = _loop(enable_rl=True)
    loop.register_node("node-1", "http://node-1")
    request = AnalyticsRequest(
        request_id="rl-rich-state",
        coverage_range=(1.0, 1.0),
        placement_limits=PlacementLimits(max_jobs_per_node=1),
    )

    config = loop.submit_request(request)
    loop._request_latencies[request.request_id] = [120.0, 80.0]

    _, decision = loop._make_rl_decision(
        request.request_id,
        request,
        config,
        node_metrics={},
        current_coverage=1.0,
    )

    assert decision is not None
    assert decision.state["active_jobs"] == 1
    assert decision.state["capacity_used"] == 1
    assert decision.state["capacity_free"] == 0
    assert decision.state["pending_requests"] == 0
    assert decision.state["latency_ms"] == 100.0
    assert decision.state["recent_slo_violations"] == 0
    assert "capacity_penalty" in decision.reward_components
    assert "latency_penalty" in decision.reward_components


def test_analytics_extraction_uses_completed_output_timestamp_not_config_timestamp():
    metrics = {
        "analytics": [
            {
                "request_id": "req-output",
                "updated_at": "2030-01-01T00:00:00+00:00",
                "last_result_at": "2020-01-01T00:00:00+00:00",
                "sample_rate_applied": 0.42,
                "service_metrics": {"sample_rate": 0.99},
            }
        ]
    }

    extracted = OrchestrationLoop._extract_request_analytics_metrics(metrics, "req-output")

    assert extracted["sample_rate"] == 0.42
    assert extracted["output_observed"] is True
    assert extracted["freshness_age_s"] is not None
    assert extracted["freshness_age_s"] > 60.0


def test_analytics_extraction_does_not_treat_config_as_observed_output():
    metrics = {
        "analytics": [
            {
                "request_id": "req-pending",
                "target_sample": 0.8,
                "service_metrics": {"sample_rate": 0.8},
                "last_result_at": None,
            }
        ]
    }

    extracted = OrchestrationLoop._extract_request_analytics_metrics(metrics, "req-pending")

    assert extracted["output_observed"] is False
    assert extracted["sample_rate"] is None
    assert extracted["freshness_age_s"] is None


def test_observed_sample_and_freshness_violations_are_persisted():
    loop = _loop()
    sink = _ViolationSink()
    loop._persistence = sink
    request = AnalyticsRequest(
        request_id="req-slo",
        sample_range=(0.4, 0.8),
        freshness_range=(30.0, 90.0),
    )

    loop._persist_observed_quality_violations(
        request=request,
        actual_sample=0.2,
        freshness_age_s=120.0,
    )

    assert [violation.metric for violation in sink.violations] == [
        "sample_rate",
        "freshness_age_seconds",
    ]


def test_initial_operating_point_quantiles_are_applied_to_config_and_environment():
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_persistence=False,
            enable_logging=False,
            enable_control_plane=False,
            enable_rl=True,
            initial_coverage_quantile=1.0,
            initial_sample_quantile=0.0,
            initial_freshness_quantile=0.25,
        )
    )
    loop.register_node("node-1", "http://node-1")
    request = AnalyticsRequest(
        request_id="fixed-point",
        coverage_range=(0.4, 0.8),
        sample_range=(0.2, 0.6),
        freshness_range=(20.0, 100.0),
        algorithm="static",
    )

    config = loop.submit_request(request)

    assert config.target_coverage == 0.8
    assert config.target_sample == 0.2
    assert config.target_freshness == 40.0
    assert loop._rl_environments[request.request_id].effective_config is config

    loop.set_request_input_multiplier(request.request_id, 16)
    assert request.input_multiplier == 16
    assert config.input_multiplier == 16
    assert loop._rl_environments[request.request_id].effective_config.input_multiplier == 16
