# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
RL environment bridging the architecture monitor with the Q-learning agent.

Provides :class:`OrchestrationEnvironment` which translates raw node metrics
into discretised RL states and maps :class:`RangeAdjustmentAction` deltas
back into updated :class:`EffectiveConfiguration` values, enforcing the
user-specified requirement ranges.

Reward signal balances quality (coverage × sample), resource pressure,
and constraint-violation penalties through configurable weights.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from argos.settings import (
    DEFAULT_BOUNDS,
    DEFAULT_PRESSURE_WEIGHTS,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_STEPS,
    DEFAULT_THRESHOLDS,
    STATE_BINS,
    ActionSteps,
    PressureWeights,
    RequirementBounds,
    ResourceThresholds,
    RewardWeights,
)
from argos.domain.architecture import (
    ArchitectureState,
    ComputeMetrics,
    DataMetrics,
    NetworkMetrics,
    ServiceMetrics,
)
from argos.domain.monitor import ArchitectureMonitor, ArchitectureSample
from argos.domain.requests import AnalyticsRequest, EffectiveConfiguration
from argos.domain.requirements import AnalyticsRequirements, RequirementAction, bucketize
from argos.domain.numeric_contract import (
    canonical_contract_value,
    contract_position,
)
from argos.orchestrator.rl.base import STANDARD_ACTIONS


@dataclass
class RewardBreakdown:
    """Components of the reward signal.

    Components are bounded before aggregation. Their raw sum can fall below
    -1 when several penalties coincide, so ``total`` explicitly clips the
    final score to [-1, +1].

    Range penalties are split per dimension (coverage, sample, freshness)
    so the agent and evaluation graphs can identify which requirement
    dimension is causing violations.  Each is capped at ~0.167 so their
    sum stays in [0, 0.5], preserving the original penalty budget.
    """

    requirement_quality: float
    resource_penalty: float
    cost_penalty: float
    range_penalty_coverage: float = 0.0
    range_penalty_sample: float = 0.0
    range_penalty_freshness: float = 0.0
    resource_overload_penalty: float = 0.0
    fairness_penalty: float = 0.0
    latency_penalty: float = 0.0
    capacity_penalty: float = 0.0

    @property
    def total(self) -> float:
        """Return the aggregated reward clamped to [-1, +1]."""
        raw = (
            self.requirement_quality
            - self.resource_penalty
            - self.cost_penalty
            - self.range_penalty_coverage
            - self.range_penalty_sample
            - self.range_penalty_freshness
            - self.resource_overload_penalty
            - self.fairness_penalty
            - self.latency_penalty
            - self.capacity_penalty
        )
        return max(-1.0, min(1.0, raw))


@dataclass
class StepResult:
    """Result of a single environment step or observation."""

    encoded_state: Hashable
    reward: float
    reward_details: RewardBreakdown
    observation: ArchitectureSample
    effective_config: Optional[EffectiveConfiguration] = None


@dataclass
class RangeAdjustmentAction:
    """
    Action for adjusting configuration within user-specified ranges.

    Unlike RequirementAction which modifies fixed values,
    this action navigates within the allowed ranges.
    """

    coverage_delta: float = 0.0  # Positive = more nodes, negative = fewer
    sample_delta: float = 0.0  # Positive = more data, negative = less
    freshness_delta: float = 0.0  # Positive = less fresh, negative = more fresh


# Define discrete actions for range-based navigation
RANGE_ACTIONS = [
    RangeAdjustmentAction(
        coverage_delta=action.coverage_delta,
        sample_delta=action.sample_delta,
        freshness_delta=action.freshness_delta,
    )
    for action in STANDARD_ACTIONS
]


class OrchestrationEnvironment:
    """
    Connects the monitor (architecture state) with the Q-learning agent by
    exposing a discrete state and computing rewards.
    """

    def __init__(
        self,
        monitor: ArchitectureMonitor,
        requirements: Optional[AnalyticsRequirements] = None,
        thresholds: ResourceThresholds = DEFAULT_THRESHOLDS,
        bounds: RequirementBounds = DEFAULT_BOUNDS,
        steps: ActionSteps = DEFAULT_STEPS,
        reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
        pressure_weights: PressureWeights = DEFAULT_PRESSURE_WEIGHTS,
    ):
        self.monitor = monitor
        self.requirements = requirements or AnalyticsRequirements()
        self.thresholds = thresholds
        self.bounds = bounds
        self.steps = steps
        self.reward_weights = reward_weights
        self.pressure_weights = pressure_weights

        # New: Range-based orchestration state
        self._current_request: Optional[AnalyticsRequest] = None
        self._effective_config: Optional[EffectiveConfiguration] = None
        self._node_metrics: dict[str, dict] = {}
        self._fairness_debt: float = 0.0
        self._active_jobs: int = 0
        self._capacity_used: int = 0
        self._capacity_free: int = 0
        self._pending_requests: int = 0
        self._recent_slo_violations: int = 0
        self._latency_ms: float = 0.0
        self._service_duty_percent: float = 0.0

    def set_request(
        self,
        request: AnalyticsRequest,
        effective_config: Optional[EffectiveConfiguration] = None,
    ) -> EffectiveConfiguration:
        """
        Set the current analytics request.

        Initializes the effective configuration at the midpoint of ranges.
        """
        self._current_request = request
        self._effective_config = effective_config or request.midpoint_config()
        self._effective_config.target_coverage = canonical_contract_value(
            self._effective_config.target_coverage,
            request.coverage_range,
        )
        self._effective_config.target_sample = canonical_contract_value(
            self._effective_config.target_sample,
            request.sample_range,
        )
        self._effective_config.target_freshness = canonical_contract_value(
            self._effective_config.target_freshness,
            request.freshness_range,
        )

        # Keep the requirements view in sync with the effective configuration
        self.requirements.coverage = self._effective_config.target_coverage
        self.requirements.sample = self._effective_config.target_sample
        self.requirements.freshness_seconds = self._effective_config.target_freshness

        return self._effective_config

    def update_node_metrics(self, metrics: dict[str, dict]) -> None:
        """Update the node metrics from the orchestration loop."""
        self._node_metrics = metrics
        duty_values: list[float] = []
        request_id = self._current_request.request_id if self._current_request else None
        if request_id:
            for payload in metrics.values():
                analytics = payload.get("analytics")
                if not isinstance(analytics, list):
                    continue
                runtime = next(
                    (
                        item
                        for item in analytics
                        if isinstance(item, dict) and str(item.get("request_id", "")) == request_id
                    ),
                    None,
                )
                if not runtime or not runtime.get("last_result_at"):
                    continue
                processing_ms = float(runtime.get("last_processing_time_ms") or 0.0)
                freshness_s = max(float(runtime.get("target_freshness") or 0.0), 1e-6)
                if processing_ms > 0:
                    duty_values.append(processing_ms / (freshness_s * 10.0))
        self._service_duty_percent = max(duty_values, default=0.0)

    def set_input_multiplier(self, multiplier: int) -> None:
        """Update the exogenous input-volume level without changing quality bounds."""
        value = max(1, min(100, int(multiplier)))
        if self._current_request:
            self._current_request.input_multiplier = value
        if self._effective_config:
            self._effective_config.input_multiplier = value

    def update_fairness_debt(self, debt: float) -> None:
        """Set external fairness debt signal (0 = fair, 1 = severe unfairness)."""
        self._fairness_debt = max(0.0, min(1.0, float(debt)))

    def update_runtime_context(
        self,
        *,
        active_jobs: int = 0,
        capacity_used: int = 0,
        capacity_free: int = 0,
        pending_requests: int = 0,
        recent_slo_violations: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Update orchestration context that is not present in node telemetry."""
        self._active_jobs = max(0, int(active_jobs))
        self._capacity_used = max(0, int(capacity_used))
        self._capacity_free = max(0, int(capacity_free))
        self._pending_requests = max(0, int(pending_requests))
        self._recent_slo_violations = max(0, int(recent_slo_violations))
        self._latency_ms = max(0.0, float(latency_ms or 0.0))

    def _clamp_to_range(self, value: float, range_tuple: tuple[float, float]) -> float:
        """Clamp a value to stay within the specified range."""
        return canonical_contract_value(value, range_tuple)

    def available_action_indices(self) -> tuple[int, ...]:
        """Return actions that can change the current bounded configuration."""
        if not self._current_request or not self._effective_config:
            return (0,)
        current = self._effective_config
        available = [0]
        for index, action in enumerate(RANGE_ACTIONS[1:], start=1):
            coverage = self._clamp_to_range(
                current.target_coverage + action.coverage_delta,
                self._current_request.coverage_range,
            )
            sample = self._clamp_to_range(
                current.target_sample + action.sample_delta,
                self._current_request.sample_range,
            )
            freshness = self._clamp_to_range(
                current.target_freshness + action.freshness_delta,
                self._current_request.freshness_range,
            )
            if (
                abs(coverage - current.target_coverage) > 1e-9
                or abs(sample - current.target_sample) > 1e-9
                or abs(freshness - current.target_freshness) > 1e-9
            ):
                available.append(index)
        return tuple(available)

    def _encode(self, state: ArchitectureState) -> tuple[int, ...]:
        """Discretize architecture + requirement state for the Q-table."""
        arch_tuple = (
            bucketize(state.compute.cpu_utilization, STATE_BINS["cpu"]),
            bucketize(state.compute.memory_utilization, STATE_BINS["memory"]),
            bucketize(state.network.bandwidth_mbps or 0.0, STATE_BINS["bandwidth"]),
        )
        if self._current_request and self._effective_config:
            request = self._current_request
            config = self._effective_config

            req_tuple = (
                bucketize(
                    contract_position(config.target_coverage, request.coverage_range),
                    STATE_BINS["contract_position"],
                ),
                bucketize(
                    contract_position(config.target_sample, request.sample_range),
                    STATE_BINS["contract_position"],
                ),
                bucketize(
                    contract_position(config.target_freshness, request.freshness_range),
                    STATE_BINS["contract_position"],
                ),
            )
        else:
            req_tuple = self.requirements.as_state_tuple()
        runtime_tuple = (
            bucketize(self._service_duty_percent, STATE_BINS["service_duty"]),
            bucketize(
                self._effective_config.input_multiplier if self._effective_config else 1,
                STATE_BINS["input_multiplier"],
            ),
            bucketize(self._active_jobs, STATE_BINS["active_jobs"]),
            bucketize(self._capacity_free, STATE_BINS["free_capacity"]),
            bucketize(self._pending_requests, STATE_BINS["pending_requests"]),
            bucketize(state.network.latency_ms or self._latency_ms, STATE_BINS["latency"]),
            bucketize(self._recent_slo_violations, STATE_BINS["recent_slo_violations"]),
        )
        return arch_tuple + req_tuple + runtime_tuple

    def _requirement_pressure(self) -> dict:
        """
        Map coverage/sample/freshness to an estimated pressure contribution for each resource.
        This heuristic follows the multidimensional-elasticity model: higher
        coverage/sample/freshness increases CPU, memory and bandwidth pressure.
        """
        freshness_pressure = self.requirements.freshness_pressure()
        pw = self.pressure_weights
        return {
            "cpu": (
                pw.cpu_coverage * self.requirements.coverage
                + pw.cpu_sample * self.requirements.sample
                + pw.cpu_freshness * freshness_pressure
            ),
            "memory": (
                pw.memory_coverage * self.requirements.coverage
                + pw.memory_sample * self.requirements.sample
                + pw.memory_freshness * freshness_pressure
            ),
            "bandwidth": (
                pw.bandwidth_coverage * self.requirements.coverage
                + pw.bandwidth_sample * self.requirements.sample
                + pw.bandwidth_freshness * freshness_pressure
            ),
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp a value to [0, 1]."""
        return max(0.0, min(1.0, value))

    @staticmethod
    def _range_score_higher_is_better(value: float, lo: float, hi: float) -> float:
        """Score a value inside a range where the upper bound is best."""
        return OrchestrationEnvironment._clamp01((value - lo) / max(hi - lo, 1e-6))

    @staticmethod
    def _range_score_lower_is_better(value: float, lo: float, hi: float) -> float:
        """Score a value inside a range where the lower bound is best."""
        return OrchestrationEnvironment._clamp01((hi - value) / max(hi - lo, 1e-6))

    def _request_requirement_quality(self) -> float:
        """Return requirement quality normalised to the active request ranges."""
        if not self._current_request:
            return self.requirements.requirement_quality(self.bounds)

        req = self._current_request
        cfg = self._effective_config
        coverage = cfg.target_coverage if cfg else self.requirements.coverage
        sample = cfg.target_sample if cfg else self.requirements.sample
        freshness = cfg.target_freshness if cfg else self.requirements.freshness_seconds

        coverage_score = self._range_score_higher_is_better(coverage, *req.coverage_range)
        sample_score = self._range_score_higher_is_better(sample, *req.sample_range)
        freshness_score = self._range_score_lower_is_better(freshness, *req.freshness_range)

        return (coverage_score + sample_score + freshness_score) / 3.0

    def _max_cost_for_current_context(self) -> float:
        """Return the maximum expected cost for the active request or global bounds."""
        if self._current_request:
            req = self._current_request
            return req.coverage_range[1] * req.sample_range[1] * (1.0 + 1.0 / max(req.freshness_range[0], 1.0))

        return (
            self.bounds.coverage[1] * self.bounds.sample[1] * (1.0 + 1.0 / max(self.bounds.freshness_seconds[0], 1.0))
        )

    def _observed_resource_pressure(self, state: ArchitectureState) -> Optional[float]:
        """Return pressure from observed CPU/memory/network metrics when available."""
        values: list[float] = []
        if state.compute.cpu_utilization > 0:
            values.append(self._clamp01(state.compute.cpu_utilization / max(self.thresholds.cpu_percent, 1e-6)))
        if state.compute.memory_utilization > 0:
            values.append(self._clamp01(state.compute.memory_utilization / max(self.thresholds.memory_percent, 1e-6)))
        if state.network.bandwidth_mbps and state.network.bandwidth_mbps > 0:
            values.append(self._clamp01(state.network.bandwidth_mbps / max(self.thresholds.bandwidth_mbps, 1e-6)))
        host_pressure = sum(values) / len(values) if values else None
        duty_pressure = self._clamp01(self._service_duty_percent / 100.0)
        if host_pressure is None:
            return duty_pressure if self._service_duty_percent > 0 else None
        return max(host_pressure, duty_pressure)

    @staticmethod
    def _dim_penalty(current: float, lo: float, hi: float) -> float:
        """Linear penalty for a single dimension outside [lo, hi].

        Returns 0 inside the range and grows linearly with distance
        outside, capped at 0.5.
        """
        if lo <= current <= hi:
            return 0.0
        distance = (lo - current) if current < lo else (current - hi)
        return min(distance * 2.0, 0.5)

    def _range_penalties(self) -> tuple[float, float, float]:
        """Per-dimension penalties when outside the user's requested range.

        Returns ``(coverage_penalty, sample_penalty, freshness_penalty)``.
        Each value is ``_dim_penalty() / 3``, keeping it in [0, ~0.167]
        so the sum of all three stays in [0, 0.5] — the same total budget
        as the previous averaged implementation.
        """
        if not self._current_request or not self._effective_config:
            return 0.0, 0.0, 0.0

        cfg = self._effective_config
        req = self._current_request

        cov_p = (
            self._dim_penalty(
                cfg.target_coverage,
                *req.coverage_range,
            )
            / 3.0
        )
        smp_p = (
            self._dim_penalty(
                cfg.target_sample,
                *req.sample_range,
            )
            / 3.0
        )
        frs_p = (
            self._dim_penalty(
                cfg.target_freshness,
                *req.freshness_range,
            )
            / 3.0
        )

        return cov_p, smp_p, frs_p

    @staticmethod
    def _extract_compute_metrics(metrics: dict) -> tuple[float, float]:
        """Extract cpu/memory from structured or flattened state payloads."""
        state = metrics.get("state", {})

        cpu = 0.0
        mem = 0.0

        compute = state.get("compute")
        if isinstance(compute, dict):
            cpu = float(compute.get("cpu_utilization", 0.0) or 0.0)
            mem = float(compute.get("memory_utilization", 0.0) or 0.0)
        else:
            cpu = float(state.get("compute.cpu_utilization", state.get("cpu_utilization", 0.0)) or 0.0)
            mem = float(state.get("compute.memory_utilization", state.get("memory_utilization", 0.0)) or 0.0)

        # Fallback to the flattened payload reported by the node API.
        if cpu <= 0.0 or mem <= 0.0:
            state_flat = metrics.get("state_flat", {})
            if isinstance(state_flat, dict):
                cpu = cpu if cpu > 0.0 else float(state_flat.get("compute.cpu_utilization", 0.0) or 0.0)
                mem = mem if mem > 0.0 else float(state_flat.get("compute.memory_utilization", 0.0) or 0.0)

        return cpu, mem

    def _extract_request_limits(self, metrics: dict) -> tuple[float, float]:
        """Extract per-request CPU/memory limits from node analytics payload."""
        cpu_limit = self.thresholds.cpu_hard_limit
        mem_limit = self.thresholds.memory_hard_limit

        if not self._current_request:
            return cpu_limit, mem_limit

        analytics = metrics.get("analytics", [])
        if not isinstance(analytics, list):
            return cpu_limit, mem_limit

        for item in analytics:
            if not isinstance(item, dict):
                continue
            if item.get("request_id") != self._current_request.request_id:
                continue
            cpu_limit = float(item.get("cpu_max_percent", cpu_limit) or cpu_limit)
            mem_limit = float(item.get("memory_max_percent", mem_limit) or mem_limit)
            break

        return cpu_limit, mem_limit

    def _resource_overload_penalty(self) -> float:
        """Penalty when actual node CPU/memory exceeds hard limits.

        Reads real metrics from ``_node_metrics`` (fed by the loop via
        :meth:`update_node_metrics`).  Returns 0 when no metrics are
        available or when usage is below the configured thresholds.

        The penalty scales linearly with the excess above the limit and
        is steep (×5) to strongly discourage overloading nodes.
        """
        if not self._node_metrics:
            return 0.0

        cpu_values: list[float] = []
        mem_values: list[float] = []
        cpu_limits: list[float] = []
        mem_limits: list[float] = []
        for metrics in self._node_metrics.values():
            cpu, mem = self._extract_compute_metrics(metrics)
            cpu_lim, mem_lim = self._extract_request_limits(metrics)
            if cpu > 0:
                cpu_values.append(cpu)
                cpu_limits.append(cpu_lim)
            if mem > 0:
                mem_values.append(mem)
                mem_limits.append(mem_lim)

        if not cpu_values and not mem_values:
            return 0.0

        penalty = 0.0
        weight = self.thresholds.overload_penalty_weight

        if cpu_values:
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_cpu_limit = sum(cpu_limits) / len(cpu_limits) if cpu_limits else self.thresholds.cpu_hard_limit
            if avg_cpu > avg_cpu_limit:
                excess = (avg_cpu - avg_cpu_limit) / 100.0
                penalty += min(excess * 5.0, 1.0) * weight

        if mem_values:
            avg_mem = sum(mem_values) / len(mem_values)
            avg_mem_limit = sum(mem_limits) / len(mem_limits) if mem_limits else self.thresholds.memory_hard_limit
            if avg_mem > avg_mem_limit:
                excess = (avg_mem - avg_mem_limit) / 100.0
                penalty += min(excess * 5.0, 1.0) * weight

        return penalty

    def _observation_from_node_metrics(self) -> Optional[ArchitectureSample]:
        """Build a synthetic observation from real polled node metrics."""
        if not self._node_metrics:
            return None

        cpu_vals: list[float] = []
        mem_vals: list[float] = []
        bw_vals: list[float] = []
        lat_vals: list[float] = []

        for metrics in self._node_metrics.values():
            cpu, mem = self._extract_compute_metrics(metrics)
            if cpu > 0:
                cpu_vals.append(cpu)
            if mem > 0:
                mem_vals.append(mem)

            state = metrics.get("state", {})
            net = state.get("network")
            if isinstance(net, dict):
                bw = float(net.get("bandwidth_mbps", 0.0) or 0.0)
                lat = float(net.get("latency_ms", 0.0) or 0.0)
            else:
                bw = float(state.get("network.bandwidth_mbps", 0.0) or 0.0)
                lat = float(state.get("network.latency_ms", 0.0) or 0.0)
            if bw > 0:
                bw_vals.append(bw)
            if lat > 0:
                lat_vals.append(lat)

        avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
        avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else 0.0
        avg_bw = sum(bw_vals) / len(bw_vals) if bw_vals else 0.0
        avg_lat = sum(lat_vals) / len(lat_vals) if lat_vals else (self._latency_ms or None)

        state = ArchitectureState(
            compute=ComputeMetrics(cpu_utilization=avg_cpu, memory_utilization=avg_mem),
            data=DataMetrics(volume_mb=None, refresh_rate_s=None, processing_time_ms=None),
            network=NetworkMetrics(latency_ms=avg_lat, bandwidth_mbps=avg_bw),
            service=ServiceMetrics(
                service_type=self._current_request.service_type if self._current_request else "unknown",
                is_active=True,
                response_time_ms=avg_lat or 0.0,
                sample_rate=self.requirements.sample,
            ),
        )

        return ArchitectureSample(
            timestamp=datetime.now(timezone.utc),
            state=state,
            bandwidth_window_s=getattr(getattr(self.monitor, "_sampling", None), "network_window_seconds", 0.0),
        )

    def _latency_penalty(self, state: ArchitectureState) -> float:
        """Penalty for violating request response-time bounds or the operational tail budget."""
        latency_ms = state.network.latency_ms or self._latency_ms
        if latency_ms <= 0:
            return 0.0

        threshold_ms = 1000.0
        if self._current_request and self._current_request.response_time_range:
            threshold_ms = max(self._current_request.response_time_range[1] * 1000.0, 1.0)

        if latency_ms <= threshold_ms:
            return 0.0
        return min(((latency_ms - threshold_ms) / threshold_ms) * 0.35, 0.35)

    def _capacity_penalty(self) -> float:
        """Penalty when the cluster has no placement headroom while demand is waiting."""
        if self._capacity_free <= 0 and self._pending_requests > 0:
            return 0.30
        if self._capacity_free <= 0:
            return 0.15
        if self._capacity_free == 1 and self._pending_requests > 0:
            return 0.08
        return 0.0

    def _reward(self, state: ArchitectureState) -> RewardBreakdown:
        """Compute a bounded reward that is finally clipped to [-1, +1].

        Design decisions
        ----------------
        * **Actual host metrics** — a steep overload penalty fires when
          real CPU or memory readings exceed configurable hard limits,
          teaching the agent to reduce load before nodes saturate.
        * **Heuristic pressure** — requirement-induced pressure is still
          used as a softer signal for resource awareness.
        * **Linear penalties** — ``pressure`` and ``cost`` scale linearly,
          creating clear negative/positive reward regions that help the agent
          distinguish between high-quality and low-quality decisions.
        * **Normalised weights** — the configured ``resource_pressure`` and
          ``cost`` weights are converted to blend ratios that sum to 1.0,
          so ``resource_penalty + cost_penalty`` stays in [0, 1]. Additional
          bounded penalties can push the raw aggregate below -1.
        """
        req_pressures = self._requirement_pressure()

        # Prefer real telemetry. Fall back to heuristic requirement pressure
        # when no usable node/monitor metrics exist.
        observed_pressure = self._observed_resource_pressure(state)
        pressure_avg = (
            observed_pressure
            if observed_pressure is not None
            else (req_pressures["cpu"] + req_pressures["memory"] + req_pressures["bandwidth"]) / 3.0
        )

        # Quality ∈ [0, 1], scaled by its config weight.
        quality = self._request_requirement_quality() * self.reward_weights.requirement_quality

        # Cost normalised to [0, 1].
        raw_cost = self.requirements.estimated_cost()
        max_cost = self._max_cost_for_current_context()
        cost_norm = min(raw_cost / max(max_cost, 1e-6), 1.0)

        # Blend weights normalised so penalty budget = 1.0.
        w_r = self.reward_weights.resource_pressure
        w_c = self.reward_weights.cost
        w_total = w_r + w_c

        # Linear scaling for clear negative/positive regions.
        resource_penalty = (w_r / w_total) * pressure_avg
        cost_penalty = (w_c / w_total) * cost_norm

        cov_rp, smp_rp, frs_rp = self._range_penalties()
        overload_penalty = self._resource_overload_penalty()
        fairness_penalty = self._fairness_debt * 0.2
        latency_penalty = self._latency_penalty(state)
        capacity_penalty = self._capacity_penalty()

        return RewardBreakdown(
            requirement_quality=quality,
            resource_penalty=resource_penalty,
            cost_penalty=cost_penalty,
            range_penalty_coverage=cov_rp,
            range_penalty_sample=smp_rp,
            range_penalty_freshness=frs_rp,
            resource_overload_penalty=overload_penalty,
            fairness_penalty=fairness_penalty,
            latency_penalty=latency_penalty,
            capacity_penalty=capacity_penalty,
        )

    def observe(self) -> StepResult:
        """Observe the environment without applying an action."""
        observation = self._observation_from_node_metrics() or self.monitor.sample()
        encoded_state = self._encode(observation.state)
        reward_details = self._reward(observation.state)
        return StepResult(
            encoded_state=encoded_state,
            reward=reward_details.total,
            reward_details=reward_details,
            observation=observation,
            effective_config=self._effective_config,
        )

    def step(self, action: RequirementAction) -> StepResult:
        """Apply an action (requirement tweak) and observe the next state."""
        self.requirements.apply_action(action, self.bounds, self.steps)
        return self.observe()

    def step_range(self, action_idx: int) -> StepResult:
        """
        Apply a range-based action and observe the next state.

        This is the new method for active orchestration where actions
        navigate within user-specified ranges.
        """
        if not self._current_request or not self._effective_config:
            raise ValueError("No request set. Call set_request() first.")

        action = RANGE_ACTIONS[action_idx % len(RANGE_ACTIONS)]

        # Apply deltas, clamped to user's ranges
        new_coverage = self._clamp_to_range(
            self._effective_config.target_coverage + action.coverage_delta,
            self._current_request.coverage_range,
        )
        new_sample = self._clamp_to_range(
            self._effective_config.target_sample + action.sample_delta,
            self._current_request.sample_range,
        )
        new_freshness = self._clamp_to_range(
            self._effective_config.target_freshness + action.freshness_delta,
            self._current_request.freshness_range,
        )

        # Update effective config
        self._effective_config = EffectiveConfiguration(
            request_id=self._effective_config.request_id,
            service_type=self._effective_config.service_type,
            target_coverage=new_coverage,
            target_sample=new_sample,
            target_freshness=new_freshness,
            assigned_node_count=self._effective_config.assigned_node_count,
            total_nodes=self._effective_config.total_nodes,
            tenant_id=self._effective_config.tenant_id,
            priority=self._effective_config.priority,
            algorithm=self._effective_config.algorithm,
            cpu_max_percent=self._effective_config.cpu_max_percent,
            memory_max_percent=self._effective_config.memory_max_percent,
            placement_limits=self._effective_config.placement_limits,
            input_multiplier=self._effective_config.input_multiplier,
        )

        # Keep the requirements view in sync with the effective configuration
        self.requirements.coverage = new_coverage
        self.requirements.sample = new_sample
        self.requirements.freshness_seconds = new_freshness

        return self.observe()

    @property
    def current_request(self) -> Optional[AnalyticsRequest]:
        """Get the current analytics request."""
        return self._current_request

    @property
    def effective_config(self) -> Optional[EffectiveConfiguration]:
        """Get the current effective configuration."""
        return self._effective_config

    @property
    def num_range_actions(self) -> int:
        """Get the number of available range-based actions."""
        return len(RANGE_ACTIONS)
