import pytest

from argos.domain.cost import cost_model_rates_from_mapping, estimate_request_cost


def test_estimate_request_cost_returns_budget_ratio_when_budget_is_present():
    estimate = estimate_request_cost(
        assigned_node_count=3,
        responding_node_count=2,
        actual_coverage=2 / 3,
        actual_sample=0.8,
        actual_freshness_s=30.0,
        total_processing_time_ms=6000.0,
        total_data_volume_bytes=50 * 1024 * 1024,
        cost_budget=0.05,
    )

    assert estimate.score_units > 0
    assert estimate.estimated_usd > 0
    assert estimate.cpu_cost_usd > 0
    assert estimate.memory_cost_usd > 0
    assert estimate.cpu_hours > 0
    assert estimate.memory_gb_hours > 0
    assert estimate.network_gb > 0
    assert estimate.budget_ratio is not None
    assert estimate.budget_ratio == estimate.estimated_usd / 0.05
    assert estimate.estimated_usd == pytest.approx(
        estimate.cpu_cost_usd + estimate.memory_cost_usd + estimate.network_cost_usd + estimate.storage_cost_usd
    )


def test_estimate_request_cost_uses_assigned_nodes_when_no_nodes_respond():
    estimate = estimate_request_cost(
        assigned_node_count=2,
        responding_node_count=0,
        actual_coverage=0.0,
        actual_sample=0.4,
        actual_freshness_s=120.0,
        total_processing_time_ms=0.0,
        total_data_volume_bytes=0,
    )

    assert estimate.score_units == 0.0
    assert estimate.estimated_usd > 0.0


def test_cost_model_rates_can_be_loaded_from_config_mapping():
    rates = cost_model_rates_from_mapping(
        {
            "cpu_cost_per_hour": 0.1,
            "memory_cost_per_gb_hour": 0.02,
            "network_cost_per_gb": 0.03,
            "storage_cost_per_gb_month": 0.2,
            "baseline_memory_gb_per_node": 1.5,
            "baseline_storage_gb_per_request": 0.1,
        }
    )

    assert rates.cpu_cost_per_hour == 0.1
    assert rates.memory_cost_per_gb_hour == 0.02
    assert rates.network_cost_per_gb == 0.03
    assert rates.storage_cost_per_gb_month == 0.2
    assert rates.baseline_memory_gb_per_node == 1.5
    assert rates.baseline_storage_gb_per_request == 0.1
