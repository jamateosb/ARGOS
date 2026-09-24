# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Operational cost estimation helpers for live orchestration metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class CostModelRates:
    """Configurable proxy rates used to estimate per-request operational cost.

    Defaults are intentionally static and provider-neutral so experiments remain
    reproducible. Override them from ``config.yaml -> cost_model`` when using
    region-specific billing or a different FinOps model.
    """

    cpu_cost_per_hour: float = 0.0464
    memory_cost_per_gb_hour: float = 0.0050
    network_cost_per_gb: float = 0.0200
    storage_cost_per_gb_month: float = 0.1000
    baseline_memory_gb_per_node: float = 0.75
    baseline_storage_gb_per_request: float = 0.05


@dataclass(frozen=True)
class CostEstimate:
    """Machine-readable cost estimate for one request snapshot."""

    score_units: float
    estimated_usd: float
    cpu_cost_usd: float
    memory_cost_usd: float
    network_cost_usd: float
    storage_cost_usd: float
    cpu_hours: float
    memory_gb_hours: float
    network_gb: float
    storage_gb_month: float
    budget_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_COST_MODEL_RATES = CostModelRates()


def cost_model_rates_from_mapping(raw: Optional[dict]) -> CostModelRates:
    """Build ``CostModelRates`` from a config mapping, preserving defaults."""
    raw = raw or {}
    defaults = DEFAULT_COST_MODEL_RATES
    return CostModelRates(
        cpu_cost_per_hour=float(raw.get("cpu_cost_per_hour", defaults.cpu_cost_per_hour)),
        memory_cost_per_gb_hour=float(raw.get("memory_cost_per_gb_hour", defaults.memory_cost_per_gb_hour)),
        network_cost_per_gb=float(raw.get("network_cost_per_gb", defaults.network_cost_per_gb)),
        storage_cost_per_gb_month=float(raw.get("storage_cost_per_gb_month", defaults.storage_cost_per_gb_month)),
        baseline_memory_gb_per_node=float(raw.get("baseline_memory_gb_per_node", defaults.baseline_memory_gb_per_node)),
        baseline_storage_gb_per_request=float(
            raw.get("baseline_storage_gb_per_request", defaults.baseline_storage_gb_per_request)
        ),
    )


def estimate_request_cost(
    *,
    assigned_node_count: int,
    responding_node_count: int,
    actual_coverage: float,
    actual_sample: float,
    actual_freshness_s: float,
    total_processing_time_ms: float,
    total_data_volume_bytes: int,
    cost_budget: Optional[float] = None,
    rates: CostModelRates = DEFAULT_COST_MODEL_RATES,
) -> CostEstimate:
    """Estimate live operational cost from request metrics.

    The model intentionally mixes two views:
    - ``score_units`` keeps a simple elasticity-oriented pressure score used for
      comparisons in reports.
    - ``estimated_usd`` approximates runtime cost using CPU time, memory holding
      time, transferred data, and a tiny storage footprint.

    The estimate is not meant to mirror AWS billing exactly; it provides a
    stable, explainable operational proxy until provider-specific billing data is
    integrated.
    """

    nodes = max(responding_node_count, assigned_node_count, 0)
    sample = max(actual_sample, 0.0)
    coverage = max(actual_coverage, 0.0)
    freshness_s = max(actual_freshness_s, 1.0)
    freshness_hours = freshness_s / 3600.0

    cpu_hours = max(nodes, 0) * max(total_processing_time_ms, 0.0) / 3_600_000.0
    memory_gb_hours = max(nodes, 0) * rates.baseline_memory_gb_per_node * max(sample, 0.05) * freshness_hours
    network_gb = max(total_data_volume_bytes, 0) / float(1024**3)
    storage_gb_month = rates.baseline_storage_gb_per_request * (freshness_hours / (24.0 * 30.0))

    cpu_cost_usd = cpu_hours * rates.cpu_cost_per_hour
    memory_cost_usd = memory_gb_hours * rates.memory_cost_per_gb_hour
    network_cost_usd = network_gb * rates.network_cost_per_gb
    storage_cost_usd = storage_gb_month * rates.storage_cost_per_gb_month
    estimated_usd = cpu_cost_usd + memory_cost_usd + network_cost_usd + storage_cost_usd

    score_units = nodes * coverage * sample * (1.0 + 1.0 / freshness_s)
    budget_ratio = None
    if cost_budget is not None and cost_budget > 0:
        budget_ratio = estimated_usd / cost_budget

    return CostEstimate(
        score_units=score_units,
        estimated_usd=estimated_usd,
        cpu_cost_usd=cpu_cost_usd,
        memory_cost_usd=memory_cost_usd,
        network_cost_usd=network_cost_usd,
        storage_cost_usd=storage_cost_usd,
        cpu_hours=cpu_hours,
        memory_gb_hours=memory_gb_hours,
        network_gb=network_gb,
        storage_gb_month=storage_gb_month,
        budget_ratio=budget_ratio,
    )
