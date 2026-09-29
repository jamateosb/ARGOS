"""Contract-feasible placement: realized coverage k/N must stay inside the accepted range."""

import pytest

from argos.domain.numeric_contract import feasible_node_range
from argos.domain.requests import AnalyticsRequest, PlacementLimits
from argos.orchestrator.loop import OrchestrationLoop, OrchestrationLoopConfig


@pytest.mark.parametrize(
    ("coverage_range", "expected"),
    [
        ((0.25, 0.45), (1, 1)),  # lax-background
        ((0.60, 0.95), (2, 2)),  # short-burst
        ((0.75, 1.00), (3, 3)),  # aggressive-incident
        ((0.45, 0.75), (2, 2)),  # standard-operations
        ((0.33, 0.67), (1, 2)),  # cost-sensitive: 1/3 and 2/3 within tolerance
        ((0.40, 0.60), None),  # between two reachable values
    ],
)
def test_feasible_node_range_on_three_nodes(coverage_range, expected):
    assert feasible_node_range(coverage_range, 3) == expected


def test_feasible_node_range_respects_max_nodes():
    assert feasible_node_range((0.30, 1.00), 3, max_nodes=2) == (1, 2)
    assert feasible_node_range((0.75, 1.00), 3, max_nodes=2) is None
    assert feasible_node_range((0.25, 0.45), 3, max_nodes=1) == (1, 1)


def test_feasible_node_range_without_active_nodes():
    assert feasible_node_range((0.25, 0.45), 0) is None


def _loop(nodes: int = 3) -> OrchestrationLoop:
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            enable_persistence=False,
            enable_logging=False,
            enable_control_plane=False,
            enable_rl=False,
        )
    )
    for index in range(1, nodes + 1):
        loop.register_node(f"node-{index}", f"http://node-{index}")
    return loop


@pytest.mark.parametrize(
    "coverage_range",
    [(0.25, 0.45), (0.60, 0.95), (0.75, 1.00), (0.45, 0.75), (0.33, 0.67)],
)
def test_desired_node_count_keeps_realized_coverage_inside_contract(coverage_range):
    loop = _loop()
    request = AnalyticsRequest(request_id="r", coverage_range=coverage_range)
    config = request.midpoint_config()
    low, high = coverage_range
    for step in range(101):
        config.target_coverage = low + (high - low) * step / 100
        count = loop._desired_node_count(request, config)
        assert low - 1e-9 <= count / 3 <= high + 1e-9, (config.target_coverage, count)
        # The target itself is never moved outside the client range.
        assert low <= config.target_coverage <= high


def test_desired_node_count_follows_target_when_several_counts_are_feasible():
    loop = _loop()
    request = AnalyticsRequest(request_id="r", coverage_range=(0.33, 0.67))
    config = request.midpoint_config()
    config.target_coverage = 0.33
    assert loop._desired_node_count(request, config) == 1
    config.target_coverage = 0.60
    assert loop._desired_node_count(request, config) == 2


def test_desired_node_count_respects_max_nodes():
    loop = _loop()
    request = AnalyticsRequest(
        request_id="r",
        coverage_range=(0.30, 1.00),
        placement_limits=PlacementLimits(max_nodes=2),
    )
    config = request.midpoint_config()
    config.target_coverage = 1.0
    assert loop._desired_node_count(request, config) == 2


def test_unreachable_coverage_range_is_rejected_at_admission():
    loop = _loop()
    request = AnalyticsRequest(request_id="gap", coverage_range=(0.40, 0.60))
    loop.submit_request(request)
    assert loop.get_job_status("gap").value == "rejected"
    assert loop.get_job_status_reason("gap") == "coverage_range_unreachable_with_active_nodes"
    assert not loop._request_assigned_nodes("gap")


def test_max_nodes_below_coverage_minimum_keeps_its_rejection_reason():
    loop = _loop()
    request = AnalyticsRequest(
        request_id="limited",
        coverage_range=(0.75, 1.00),
        placement_limits=PlacementLimits(max_nodes=2),
    )
    loop.submit_request(request)
    assert loop.get_job_status("limited").value == "rejected"
    assert loop.get_job_status_reason("limited") == "placement_limits_below_coverage_min"


def test_feasible_request_is_admitted_with_contract_safe_node_count():
    loop = _loop()
    request = AnalyticsRequest(request_id="lax", coverage_range=(0.25, 0.45))
    config = loop.submit_request(request)
    assert loop.get_job_status("lax").value == "accepted"
    assert config.assigned_node_count == 1
    assert len(loop._request_assigned_nodes("lax")) == 1
