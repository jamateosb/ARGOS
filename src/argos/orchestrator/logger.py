# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Experiment logger for generating evaluation-ready CSV artifacts.

Logs orchestration data to CSV files used by the evaluation pipeline:
coverage over time, resource usage, QoS metrics, RL learning curves, etc.

Generates:
- iterations.csv: Summary metrics per orchestration iteration
- node_metrics.csv: Per-node metrics over time
- decisions.csv: RL decisions with full state/action/reward details
- rl_learning.csv: RL agent learning curves (Q-values, exploration, etc.)
- request_metrics.csv: Per-request aggregated metrics
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from argos.common.constants import RUNS_DIR


def _to_csv_dict(obj) -> dict:
    """Convert a dataclass to a flat dict suitable for CSV writing.

    datetime fields are converted to ISO strings automatically.
    """
    d = asdict(obj)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


@dataclass
class ExperimentLoggerConfig:
    """Configuration for experiment logging."""

    output_dir: str = str(RUNS_DIR)
    experiment_name: str = "default"
    log_iterations: bool = True
    log_node_metrics: bool = True
    log_decisions: bool = True
    log_rl_learning: bool = True
    log_request_metrics: bool = True
    flush_interval: int = 10  # Flush every N iterations


@dataclass
class IterationLog:
    """Log entry for a single orchestration iteration."""

    timestamp: datetime
    iteration_id: int
    total_nodes: int
    active_nodes: int
    responding_nodes: int
    configs_pushed: int
    iteration_time_ms: float
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_response_time_ms: float = 0.0
    coverage_achieved: float = 0.0
    active_requests: int = 0
    rl_decisions_made: int = 0
    errors_count: int = 0

    def to_dict(self) -> dict:
        return _to_csv_dict(self)


@dataclass
class NodeMetricLog:
    """Log entry for individual node metrics."""

    timestamp: datetime
    iteration_id: int
    node_id: str
    is_active: bool
    cpu_utilization: float
    memory_utilization: float
    bandwidth_mbps: float
    service_type: str
    sample_rate: float
    request_id: Optional[str]
    latency_ms: float = 0.0
    health_failures: int = 0
    heatmap_cells: int = 0
    hotspots_detected: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        d = _to_csv_dict(self)
        if d.get("request_id") is None:
            d["request_id"] = ""
        return d


@dataclass
class DecisionLog:
    """Log entry for RL/orchestrator decisions with full details."""

    timestamp: datetime
    iteration_id: int
    request_id: str
    algorithm: str
    policy_version: str
    action_type: str  # e.g., "hold", "increase_coverage", "decrease_sample"
    action_index: int
    # Configuration changes
    prev_coverage: float
    prev_sample: float
    prev_freshness: float
    new_coverage: float
    new_sample: float
    new_freshness: float
    # RL metrics
    reward: float
    reward_quality: float = 0.0
    reward_resource_penalty: float = 0.0
    reward_cost_penalty: float = 0.0
    reward_range_penalty_coverage: float = 0.0
    reward_range_penalty_sample: float = 0.0
    reward_range_penalty_freshness: float = 0.0
    reward_overload_penalty: float = 0.0
    reward_latency_penalty: float = 0.0
    reward_capacity_penalty: float = 0.0
    # Agent state
    epsilon: float = 0.0
    was_exploration: bool = False
    state_hash: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return _to_csv_dict(self)


@dataclass
class RLLearningLog:
    """Log entry for RL agent learning progress."""

    timestamp: datetime
    iteration_id: int
    request_id: str
    algorithm: str
    policy_version: str
    episodes: int
    steps: int
    total_reward: float
    avg_reward: float
    epsilon: float
    states_visited: int
    # Q-value statistics
    avg_q_value: float = 0.0
    max_q_value: float = 0.0
    min_q_value: float = 0.0

    def to_dict(self) -> dict:
        return _to_csv_dict(self)


@dataclass
class RequestMetricLog:
    """Log entry for per-request aggregated metrics."""

    timestamp: datetime
    iteration_id: int
    request_id: str
    service_type: str
    algorithm: str
    policy_version: str
    # Configuration
    target_coverage: float
    target_sample: float
    target_freshness: float
    # Actual achieved
    actual_coverage: float
    assigned_nodes: int
    active_nodes: int
    responding_nodes: int
    # Performance
    avg_latency_ms: float
    max_latency_ms: float
    actual_sample: Optional[float] = None
    actual_freshness: Optional[float] = None
    observed_freshness_age_max_s: Optional[float] = None
    avg_processing_time_ms: float = 0.0
    estimated_cost_units: float = 0.0
    estimated_cost_usd: float = 0.0
    cost_cpu_usd: float = 0.0
    cost_memory_usd: float = 0.0
    cost_network_usd: float = 0.0
    cost_storage_usd: float = 0.0
    cost_budget: float = 0.0
    cost_budget_ratio: float = 0.0
    total_data_volume_bytes: int = 0
    total_processing_time_ms: float = 0.0
    total_hotspots: int = 0
    input_multiplier: int = 1
    mean_service_duty_percent: Optional[float] = None
    mean_spatial_fidelity: Optional[float] = None
    mean_hotspot_recall: Optional[float] = None

    def to_dict(self) -> dict:
        return _to_csv_dict(self)


class ExperimentLogger:
    """
    Logger that produces CSV files for evaluation graphs.

    Generates five files per experiment:
    - iterations.csv: Summary metrics per orchestration iteration
    - node_metrics.csv: Per-node metrics over time
    - decisions.csv: RL/orchestrator decisions with full state/action/reward
    - rl_learning.csv: RL agent learning curves
    - request_metrics.csv: Per-request aggregated metrics
    """

    def __init__(self, config: ExperimentLoggerConfig = None, **kwargs):
        if config is None:
            config = ExperimentLoggerConfig(**kwargs) if kwargs else ExperimentLoggerConfig()
        self.config = config
        self._iteration_id = 0
        self._iterations: list[IterationLog] = []
        self._node_metrics: list[NodeMetricLog] = []
        self._decisions: list[DecisionLog] = []
        self._rl_learning: list[RLLearningLog] = []
        self._request_metrics: list[RequestMetricLog] = []
        self._initialized = False
        self._output_path: Optional[Path] = None

    @property
    def output_dir(self) -> str:
        """Return the resolved experiment output directory."""
        if self._output_path:
            return str(self._output_path)
        return self.config.output_dir

    def _ensure_initialized(self) -> None:
        """Create output directory and CSV headers if needed."""
        if self._initialized:
            return

        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f"{self.config.experiment_name}_{timestamp}"
        self._output_path = Path(self.config.output_dir) / experiment_dir
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Create CSV files with headers
        self._write_headers()
        self._initialized = True

    def _write_headers(self) -> None:
        """Write CSV headers for all log files."""
        header_specs = [
            (self.config.log_iterations, "iterations.csv", IterationLog(datetime.now(), 0, 0, 0, 0, 0, 0.0)),
            (
                self.config.log_node_metrics,
                "node_metrics.csv",
                NodeMetricLog(datetime.now(), 0, "", False, 0.0, 0.0, 0.0, "", 0.0, None),
            ),
            (
                self.config.log_decisions,
                "decisions.csv",
                DecisionLog(datetime.now(), 0, "", "", "", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            (
                self.config.log_rl_learning,
                "rl_learning.csv",
                RLLearningLog(datetime.now(), 0, "", "", "", 0, 0, 0.0, 0.0, 0.0, 0),
            ),
            (
                self.config.log_request_metrics,
                "request_metrics.csv",
                RequestMetricLog(datetime.now(), 0, "", "", "", "", 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0, 0.0, 0),
            ),
        ]
        for enabled, filename, sample in header_specs:
            if enabled:
                with open(self._output_path / filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(sample.to_dict().keys()))
                    writer.writeheader()

    def log_iteration(
        self,
        timestamp: datetime,
        total_nodes: int,
        active_nodes: int,
        responding_nodes: int,
        configs_pushed: int,
        iteration_time_ms: float,
        node_metrics: dict[str, dict] = None,
        active_requests: int = 0,
        rl_decisions_made: int = 0,
        errors_count: int = 0,
    ) -> None:
        """Log a complete orchestration iteration."""
        self._ensure_initialized()
        self._iteration_id += 1

        # Calculate averages from node metrics
        avg_cpu = 0.0
        avg_mem = 0.0
        avg_response = 0.0

        if node_metrics:
            cpu_values = []
            mem_values = []
            response_values = []

            for node_id, metrics in node_metrics.items():
                state = metrics.get("state", {})
                compute = state.get("compute", {})
                cpu_values.append(compute.get("cpu_utilization", 0.0))
                mem_values.append(compute.get("memory_utilization", 0.0))
                response_values.append(state.get("service", {}).get("response_time_ms", 0.0))

                # Get heatmap service metrics if available
                heatmap_service = metrics.get("heatmap_service", {})

                # Log individual node metrics
                if self.config.log_node_metrics:
                    network = state.get("network", {})
                    node_log = NodeMetricLog(
                        timestamp=timestamp,
                        iteration_id=self._iteration_id,
                        node_id=node_id,
                        is_active=metrics.get("is_active", False),
                        cpu_utilization=compute.get("cpu_utilization", 0.0),
                        memory_utilization=compute.get("memory_utilization", 0.0),
                        bandwidth_mbps=network.get("bandwidth_mbps", 0.0),
                        service_type=metrics.get("service_type", "none"),
                        sample_rate=metrics.get("target_sample", 0.0),
                        request_id=metrics.get("request_id"),
                        latency_ms=metrics.get("latency_ms", 0.0),
                        health_failures=metrics.get("health_failures", 0),
                        heatmap_cells=heatmap_service.get("last_heatmap_cells", 0),
                        hotspots_detected=heatmap_service.get("last_hotspots", 0),
                        processing_time_ms=heatmap_service.get("last_processing_time_ms", 0.0),
                    )
                    self._node_metrics.append(node_log)

            if cpu_values:
                avg_cpu = sum(cpu_values) / len(cpu_values)
            if mem_values:
                avg_mem = sum(mem_values) / len(mem_values)
            if response_values:
                avg_response = sum(response_values) / len(response_values)

        coverage = active_nodes / max(total_nodes, 1)

        iteration_log = IterationLog(
            timestamp=timestamp,
            iteration_id=self._iteration_id,
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            responding_nodes=responding_nodes,
            configs_pushed=configs_pushed,
            iteration_time_ms=iteration_time_ms,
            avg_cpu_utilization=avg_cpu,
            avg_memory_utilization=avg_mem,
            avg_response_time_ms=avg_response,
            coverage_achieved=coverage,
            active_requests=active_requests,
            rl_decisions_made=rl_decisions_made,
            errors_count=errors_count,
        )
        self._iterations.append(iteration_log)

        # Flush periodically
        if self._iteration_id % self.config.flush_interval == 0:
            self.flush()

    def log_decision(
        self,
        request_id: str,
        algorithm: str,
        policy_version: str,
        action_type: str,
        action_index: int,
        prev_coverage: float,
        prev_sample: float,
        prev_freshness: float,
        new_coverage: float,
        new_sample: float,
        new_freshness: float,
        reward: float,
        reward_components: dict[str, float] = None,
        epsilon: float = 0.0,
        was_exploration: bool = False,
        state_hash: str = "",
        reason: str = "",
    ) -> None:
        """Log an RL/orchestrator decision with full details."""
        self._ensure_initialized()

        reward_components = reward_components or {}

        decision = DecisionLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=self._iteration_id,
            request_id=request_id,
            algorithm=algorithm,
            policy_version=policy_version,
            action_type=action_type,
            action_index=action_index,
            prev_coverage=prev_coverage,
            prev_sample=prev_sample,
            prev_freshness=prev_freshness,
            new_coverage=new_coverage,
            new_sample=new_sample,
            new_freshness=new_freshness,
            reward=reward,
            reward_quality=reward_components.get("requirement_quality", 0.0),
            reward_resource_penalty=reward_components.get("resource_penalty", 0.0),
            reward_cost_penalty=reward_components.get("cost_penalty", 0.0),
            reward_range_penalty_coverage=reward_components.get("range_penalty_coverage", 0.0),
            reward_range_penalty_sample=reward_components.get("range_penalty_sample", 0.0),
            reward_range_penalty_freshness=reward_components.get("range_penalty_freshness", 0.0),
            reward_overload_penalty=reward_components.get("resource_overload_penalty", 0.0),
            reward_latency_penalty=reward_components.get("latency_penalty", 0.0),
            reward_capacity_penalty=reward_components.get("capacity_penalty", 0.0),
            epsilon=epsilon,
            was_exploration=was_exploration,
            state_hash=state_hash,
            reason=reason,
        )
        self._decisions.append(decision)

    def log_rl_learning(
        self,
        request_id: str,
        algorithm: str,
        policy_version: str,
        episodes: int,
        steps: int,
        total_reward: float,
        epsilon: float,
        states_visited: int,
        q_values: dict[str, float] = None,
    ) -> None:
        """Log RL agent learning progress."""
        self._ensure_initialized()

        avg_reward = total_reward / max(steps, 1)

        # Calculate Q-value statistics
        avg_q = max_q = min_q = 0.0
        if q_values:
            values = list(q_values.values())
            if values:
                avg_q = sum(values) / len(values)
                max_q = max(values)
                min_q = min(values)

        learning_log = RLLearningLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=self._iteration_id,
            request_id=request_id,
            algorithm=algorithm,
            policy_version=policy_version,
            episodes=episodes,
            steps=steps,
            total_reward=total_reward,
            avg_reward=avg_reward,
            epsilon=epsilon,
            states_visited=states_visited,
            avg_q_value=avg_q,
            max_q_value=max_q,
            min_q_value=min_q,
        )
        self._rl_learning.append(learning_log)

    def log_request_metrics(
        self,
        request_id: str,
        service_type: str,
        algorithm: str,
        policy_version: str,
        target_coverage: float,
        target_sample: float,
        target_freshness: float,
        actual_coverage: float,
        assigned_nodes: int,
        active_nodes: int,
        responding_nodes: int,
        avg_latency_ms: float,
        max_latency_ms: float,
        actual_sample: Optional[float] = None,
        actual_freshness: Optional[float] = None,
        observed_freshness_age_max_s: Optional[float] = None,
        avg_processing_time_ms: float = 0.0,
        estimated_cost_units: float = 0.0,
        estimated_cost_usd: float = 0.0,
        cost_cpu_usd: float = 0.0,
        cost_memory_usd: float = 0.0,
        cost_network_usd: float = 0.0,
        cost_storage_usd: float = 0.0,
        cost_budget: float = 0.0,
        cost_budget_ratio: float = 0.0,
        total_data_volume_bytes: int = 0,
        total_processing_time_ms: float = 0.0,
        total_hotspots: int = 0,
        input_multiplier: int = 1,
        mean_service_duty_percent: Optional[float] = None,
        mean_spatial_fidelity: Optional[float] = None,
        mean_hotspot_recall: Optional[float] = None,
    ) -> None:
        """Log per-request aggregated metrics."""
        self._ensure_initialized()

        request_log = RequestMetricLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=self._iteration_id,
            request_id=request_id,
            service_type=service_type,
            algorithm=algorithm,
            policy_version=policy_version,
            target_coverage=target_coverage,
            target_sample=target_sample,
            target_freshness=target_freshness,
            actual_coverage=actual_coverage,
            assigned_nodes=assigned_nodes,
            active_nodes=active_nodes,
            responding_nodes=responding_nodes,
            avg_latency_ms=avg_latency_ms,
            max_latency_ms=max_latency_ms,
            actual_sample=actual_sample,
            actual_freshness=actual_freshness,
            observed_freshness_age_max_s=observed_freshness_age_max_s,
            avg_processing_time_ms=avg_processing_time_ms,
            estimated_cost_units=estimated_cost_units,
            estimated_cost_usd=estimated_cost_usd,
            cost_cpu_usd=cost_cpu_usd,
            cost_memory_usd=cost_memory_usd,
            cost_network_usd=cost_network_usd,
            cost_storage_usd=cost_storage_usd,
            cost_budget=cost_budget,
            cost_budget_ratio=cost_budget_ratio,
            total_data_volume_bytes=total_data_volume_bytes,
            total_processing_time_ms=total_processing_time_ms,
            total_hotspots=total_hotspots,
            input_multiplier=input_multiplier,
            mean_service_duty_percent=mean_service_duty_percent,
            mean_spatial_fidelity=mean_spatial_fidelity,
            mean_hotspot_recall=mean_hotspot_recall,
        )
        self._request_metrics.append(request_log)

    def flush(self) -> None:
        """Write buffered logs to CSV files."""
        if not self._initialized:
            return

        buffers = [
            (self._iterations, "iterations.csv", self.config.log_iterations),
            (self._node_metrics, "node_metrics.csv", self.config.log_node_metrics),
            (self._decisions, "decisions.csv", self.config.log_decisions),
            (self._rl_learning, "rl_learning.csv", self.config.log_rl_learning),
            (self._request_metrics, "request_metrics.csv", self.config.log_request_metrics),
        ]
        for buf, filename, enabled in buffers:
            if buf and enabled:
                with open(self._output_path / filename, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(buf[0].to_dict().keys()))
                    for log in buf:
                        writer.writerow(log.to_dict())
                buf.clear()

    def close(self) -> None:
        """Flush remaining logs and close."""
        self.flush()

    @property
    def output_path(self) -> Optional[Path]:
        """Get the experiment output directory."""
        return self._output_path

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_id
