# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Configuration defaults for requirements, monitoring, and runtime collector options."""

from dataclasses import dataclass
from typing import Optional

from argos.common.constants import DATA_ROOT, DEFAULT_MACHINE_ID, DEFAULT_RANDOM_SEED, OUTPUT_FORMAT_JSONL

# -----------------------------------------------------------------------------
# RL / Requirements Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementBounds:
    """Range limits for analytics requirements."""

    coverage: tuple[float, float] = (0.2, 1.0)
    sample: tuple[float, float] = (0.1, 1.0)
    freshness_seconds: tuple[float, float] = (5.0, 600.0)  # lower = fresher


@dataclass(frozen=True)
class ActionSteps:
    """Step sizes applied when the agent tweaks the requirements."""

    coverage: float = 0.05
    sample: float = 0.05
    freshness_seconds: float = 10.0


@dataclass(frozen=True)
class ResourceThresholds:
    """Upper bounds used to measure pressure on the architecture."""

    cpu_percent: float = 75.0
    memory_percent: float = 80.0
    bandwidth_mbps: float = 150.0
    # Hard limits: steep penalty when actual node metrics exceed these
    cpu_hard_limit: float = 80.0
    memory_hard_limit: float = 85.0
    overload_penalty_weight: float = 0.5


@dataclass(frozen=True)
class RewardWeights:
    """Weights used by the reward model to combine goals."""

    requirement_quality: float = 1.0
    resource_pressure: float = 1.2
    cost: float = 0.5


@dataclass(frozen=True)
class AgentHyperParams:
    """Hyperparameters for the Q-learning agent."""

    alpha: float = 0.15
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = DEFAULT_RANDOM_SEED  # For reproducibility


# -----------------------------------------------------------------------------
# Scoring and Pressure Weights (for orchestrator selection and RL)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for orchestrator capacity scoring (must sum to ~1.0)."""

    cpu_available: float = 0.5
    memory_available: float = 0.4
    bandwidth_headroom: float = 0.1


@dataclass(frozen=True)
class PressureWeights:
    """
    Weights mapping requirements to resource pressure.
    Each row (cpu, memory, bandwidth) should sum to 1.0.
    """

    # CPU pressure weights
    cpu_coverage: float = 0.40
    cpu_sample: float = 0.35
    cpu_freshness: float = 0.25
    # Memory pressure weights
    memory_coverage: float = 0.30
    memory_sample: float = 0.40
    memory_freshness: float = 0.30
    # Bandwidth pressure weights
    bandwidth_coverage: float = 0.45
    bandwidth_sample: float = 0.45
    bandwidth_freshness: float = 0.10


# -----------------------------------------------------------------------------
# Monitoring Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    """Telemetry sampling cadence for the monitor."""

    interval_seconds: float = 1.0
    network_window_seconds: float = 5.0


@dataclass(frozen=True)
class LatencyProbeConfig:
    """Settings for optional latency probing to a target (e.g., orchestrator)."""

    enabled: bool = True
    host: Optional[str] = "127.0.0.1"
    port: int = 80
    timeout_seconds: float = 0.5


@dataclass(frozen=True)
class CollectorConfig:
    """Configuration for the data collector output."""

    data_root: str = str(DATA_ROOT)
    persist: bool = True
    rotation_seconds: int = 1200  # rotate files every 20 minutes
    max_entries_per_file: int = 300  # or when this many entries are written


@dataclass(frozen=True)
class HttpSinkConfig:
    """Configuration for HTTP sink behavior."""

    retry_attempts: int = 3
    retry_wait_seconds: float = 1.0
    default_timeout: float = 5.0


# -----------------------------------------------------------------------------
# Collector Runtime Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectorRuntimeConfig:
    """
    Runtime-adjustable settings for the collector (override via CLI/ENV).
    """

    machine_id: str = DEFAULT_MACHINE_ID
    interval_seconds: float = 1.0
    output_format: str = OUTPUT_FORMAT_JSONL
    sink: str = "local"  # local|http (more sinks can be added)
    http_endpoint: Optional[str] = None
    http_timeout: float = 5.0
    http_verify_ssl: bool = True
    latency_probe_enabled: bool = True
    latency_host: Optional[str] = "127.0.0.1"
    latency_port: int = 80
    latency_timeout: float = 0.5


@dataclass
class CollectorRunArgs:
    """
    All arguments needed to run the collector loop.
    Reduces parameter count from 16 to 1 in run_collector_loop().
    """

    machine_id: str
    interval_seconds: float
    output_format: str
    persist: bool
    sink: str
    http_endpoint: Optional[str]
    http_timeout: float
    http_verify_ssl: bool
    data_root: str
    latency_probe_enabled: bool
    latency_host: Optional[str]
    latency_port: int
    latency_timeout: float
    iterations: Optional[int] = None
    quiet: bool = True


# -----------------------------------------------------------------------------
# State Discretization Bins (for Q-learning)
# -----------------------------------------------------------------------------

STATE_BINS = {
    "cpu": [20, 40, 60, 75, 90],
    "memory": [30, 50, 70, 85, 95],
    "bandwidth": [10, 50, 100, 200, 500],
    "coverage": [0.3, 0.5, 0.7, 0.85, 1.01],
    "sample": [0.15, 0.35, 0.55, 0.75, 1.01],
    "freshness": [30, 60, 120, 240, 600],
    "contract_position": [0.2, 0.4, 0.6, 0.8],
    "active_jobs": [0, 1, 3, 6, 10],
    "free_capacity": [0, 1, 2, 4, 8],
    "pending_requests": [0, 1, 2, 5, 10],
    "latency": [10, 50, 100, 250, 1000],
    "recent_slo_violations": [0, 1, 3, 5, 10],
    "service_duty": [10, 25, 50, 75, 100],
    "input_multiplier": [1, 2, 4, 8, 16, 32],
}

RL_STATE_KEYS = (
    "cpu",
    "memory",
    "bandwidth",
    "coverage",
    "sample",
    "freshness",
    "service_duty",
    "input_multiplier",
    "active_jobs",
    "free_capacity",
    "pending_requests",
    "latency",
    "recent_slo_violations",
)
RL_STATE_SIZE = len(RL_STATE_KEYS)

# -----------------------------------------------------------------------------
# Default Instances
# -----------------------------------------------------------------------------

DEFAULT_BOUNDS = RequirementBounds()
DEFAULT_STEPS = ActionSteps()
DEFAULT_THRESHOLDS = ResourceThresholds()
DEFAULT_REWARD_WEIGHTS = RewardWeights()
DEFAULT_AGENT_PARAMS = AgentHyperParams()
DEFAULT_SAMPLING = SamplingConfig()
DEFAULT_LATENCY_PROBE = LatencyProbeConfig()
DEFAULT_COLLECTOR = CollectorConfig()
DEFAULT_COLLECTOR_RUNTIME = CollectorRuntimeConfig()
DEFAULT_SCORING_WEIGHTS = ScoringWeights()
DEFAULT_PRESSURE_WEIGHTS = PressureWeights()
DEFAULT_HTTP_SINK = HttpSinkConfig()
