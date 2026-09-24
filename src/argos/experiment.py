# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""CLI runner for live experiments and reproducible RL benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import json
import logging
import signal
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

from argos.domain.requests import SUPPORTED_RL_ALGORITHMS, AnalyticsRequest, ResourceLimits
from argos.common.constants import REPO_ROOT, RUNS_DIR, SESSIONS_DIR, TUNING_DIR, generate_session_id
from argos.config import DEFAULT_CONFIG_PATH, load_project_config, node_endpoint_list
from argos.domain.cost import cost_model_rates_from_mapping
from argos.provenance import build_run_manifest, write_run_manifest
from argos.orchestrator.logger import ExperimentLogger
from argos.orchestrator.loop import OrchestrationLoop, OrchestrationLoopConfig
from argos.orchestrator.persistence import PersistenceConfig, configure_persistence
from argos.orchestrator.rl.factory import normalize_algorithm
from argos.orchestrator.rl.selection import resolve_algorithm

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _TeeStream:
    """Write console output to the original stream and one or more log files."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _load_yaml_config(path: Optional[Path]) -> dict:
    """Load YAML config if it exists; return an empty mapping otherwise."""
    return load_project_config(path or DEFAULT_CONFIG_PATH)


def _node_endpoints_from_config(config: dict) -> str:
    return ",".join(node_endpoint_list(config))


def _range_default(job_cfg: dict, name: str, fallback: str) -> str:
    min_key = f"{name}_min"
    max_key = f"{name}_max"
    if min_key in job_cfg and max_key in job_cfg:
        return f"{job_cfg[min_key]},{job_cfg[max_key]}"
    return fallback


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_parser.add_argument("--mode", type=str, default="live", choices=["live", "benchmark"])
    pre_args, _unknown = pre_parser.parse_known_args(argv)

    config = _load_yaml_config(pre_args.config)
    rl_cfg = config.get("rl", {}) or {}
    benchmark_cfg = config.get("benchmark", {}) or {}
    orchestration_cfg = config.get("orchestration", {}) or {}
    paths_cfg = config.get("paths", {}) or {}
    mode = pre_args.mode
    job_cfg = (benchmark_cfg.get("job", {}) if mode == "benchmark" else config.get("default_job", {})) or {}
    resource_limits = job_cfg.get("resource_limits", {}) or {}
    benchmark_algorithms = benchmark_cfg.get("algorithms", ["qlearning", "dqn", "ppo"])
    if isinstance(benchmark_algorithms, list):
        benchmark_algorithm_default = ",".join(str(item) for item in benchmark_algorithms)
    else:
        benchmark_algorithm_default = str(benchmark_algorithms)
    benchmark_seeds = benchmark_cfg.get("seeds", [42])
    if isinstance(benchmark_seeds, list):
        seed_default = ",".join(str(item) for item in benchmark_seeds)
    else:
        seed_default = str(benchmark_seeds)

    parser = argparse.ArgumentParser(
        description="Run ARGOS live experiments or qlearning/dqn/ppo benchmarks",
        parents=[pre_parser],
    )

    parser.add_argument(
        "--nodes",
        type=str,
        default=_node_endpoints_from_config(config),
        help="Comma-separated node endpoints (for example http://localhost:8010,http://localhost:8011)",
    )
    parser.add_argument("--experiment-name", type=str, default="experiment")
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(benchmark_cfg.get("iterations", 0)) if mode == "benchmark" else 0,
        help="0 means run until interrupted",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(
            benchmark_cfg.get("poll_interval_seconds", orchestration_cfg.get("poll_interval_seconds", 5.0))
            if mode == "benchmark"
            else orchestration_cfg.get("poll_interval_seconds", 5.0)
        ),
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=benchmark_algorithm_default if mode == "benchmark" else str(rl_cfg.get("algorithm", "auto")),
        help="auto, qlearning, dqn, ppo, a comma-separated subset, or 'all' in benchmark mode",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=seed_default,
        help="Comma-separated random seeds used by the RL backend(s)",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="",
        help="Optional policy artifact to load after request submission",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["train", "evaluate"],
        default="train",
        help="Train a policy or evaluate a frozen policy without parameter updates",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Load the newest global policy before running. Disabled by default for reproducibility",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional session identifier. If omitted, a new one is generated.",
    )

    parser.add_argument("--rl-enabled", action="store_true", default=True)
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discount-factor", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    parser.add_argument("--epsilon-min", type=float, default=None)
    parser.add_argument(
        "--exploration-decay-steps",
        type=int,
        default=0,
        help="Decision horizon over which epsilon reaches its configured floor",
    )
    parser.add_argument(
        "--rl-params-source",
        choices=["default", "best"],
        default="best",
        help="Use config defaults or best Optuna parameters for missing RL hyperparameters.",
    )

    parser.add_argument(
        "--service-type",
        type=str,
        default=str(job_cfg.get("service_type", "geo_heatmap")),
        choices=["heatmap", "geo_heatmap"],
    )
    parser.add_argument("--coverage-range", type=str, default=_range_default(job_cfg, "coverage", "0.5,0.9"))
    parser.add_argument("--sample-range", type=str, default=_range_default(job_cfg, "sample", "0.3,0.8"))
    parser.add_argument("--freshness-range", type=str, default=_range_default(job_cfg, "freshness", "30,120"))
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument(
        "--input-multiplier",
        type=int,
        default=int(job_cfg.get("input_multiplier", 1)),
        help="Fixed trajectory input-volume multiplier for this run",
    )
    parser.add_argument(
        "--initial-quantiles",
        default="0.5,0.5,0.5",
        help="Initial coverage,sample,freshness quantiles inside their accepted ranges",
    )
    parser.add_argument(
        "--input-schedule",
        default="",
        help="Optional comma-separated input multipliers applied in fixed-length phases",
    )
    parser.add_argument(
        "--schedule-segment-iterations",
        type=int,
        default=24,
        help="Iterations per input-schedule phase",
    )
    parser.add_argument("--tenant-id", type=str, default=str(job_cfg.get("tenant_id", "default")))
    parser.add_argument(
        "--priority",
        type=str,
        default=str(job_cfg.get("priority", "standard")),
        choices=["critical", "standard", "best_effort"],
    )
    parser.add_argument("--cpu-max", type=float, default=float(resource_limits.get("cpu_max_percent", 80.0)))
    parser.add_argument("--memory-max", type=float, default=float(resource_limits.get("memory_max_percent", 85.0)))
    parser.add_argument("--output-dir", type=str, default=str(paths_cfg.get("runs_dir", RUNS_DIR)))
    parser.add_argument("--persistence-dir", type=str, default="")

    args = parser.parse_args(argv)
    if not args.nodes:
        raise ValueError("--nodes is required when config.yaml does not define cluster.nodes endpoints")
    args.config_data = config
    return args


def parse_range(range_str: str) -> tuple:
    """Parse a range string like ``0.5,0.9`` into a tuple."""
    parts = [p.strip() for p in range_str.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Expected 'min,max'")
    return float(parts[0]), float(parts[1])


def parse_quantiles(raw: str) -> tuple[float, float, float]:
    """Parse three operating-point quantiles inside [0, 1]."""
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 3 or any(value < 0.0 or value > 1.0 for value in parts):
        raise ValueError("--initial-quantiles requires three values inside [0, 1]")
    return parts[0], parts[1], parts[2]


def parse_input_schedule(raw: str) -> list[int]:
    """Parse an optional sequence of controlled input-volume multipliers."""
    if not raw.strip():
        return []
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value < 1 or value > 100 for value in values):
        raise ValueError("--input-schedule values must be integers inside [1, 100]")
    return values


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return seeds or [42]


def resolve_algorithms(mode: str, raw: str, config: Optional[dict] = None) -> list[str]:
    """Resolve one or more algorithm identifiers for the selected mode."""
    value = (raw or "auto").strip().lower()
    if value == "all":
        if mode != "benchmark":
            raise ValueError("'all' is only valid with --mode benchmark")
        return list(SUPPORTED_RL_ALGORITHMS)

    return [resolve_algorithm(part, config=config) for part in value.split(",") if part.strip()]


def _load_best_rl_params(algorithm: str) -> Optional[dict[str, float]]:
    """Load offline-tuned RL hyperparameters for one algorithm when available."""
    normalized = normalize_algorithm(algorithm)
    candidates = [TUNING_DIR / f"best_params_{normalized}.yaml", TUNING_DIR / "best_params.yaml"]
    for best_params_file in candidates:
        if not best_params_file.exists():
            continue
        try:
            with open(best_params_file, encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - defensive only
            logger.warning("Failed to load %s: %s", best_params_file, exc)
            continue

        params = raw.get("params", raw) if isinstance(raw, dict) else {}
        stored_algorithm = raw.get("algorithm") if isinstance(raw, dict) else None
        if best_params_file.name == "best_params.yaml" and stored_algorithm not in (None, normalized):
            continue
        if best_params_file.name == "best_params.yaml" and stored_algorithm is None and normalized != "qlearning":
            continue
        params = {k: v for k, v in params.items() if k != "algorithm"}
        if params:
            return params
    return None


def _apply_rl_defaults(args: argparse.Namespace, rl_enabled: bool, algorithm: str) -> None:
    """Fill missing RL CLI values from Optuna output or hardcoded defaults."""
    if not rl_enabled:
        return

    best_params = _load_best_rl_params(algorithm) if args.rl_params_source == "best" else None
    best_params = best_params or {}
    algorithm_defaults = {
        "qlearning": {
            "epsilon": 0.15,
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
        "dqn": {
            "epsilon": 1.0,
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.05,
        },
        "ppo": {
            "epsilon": 0.0,
            "learning_rate": 0.0003,
            "discount_factor": 0.99,
            "epsilon_decay": 1.0,
            "epsilon_min": 0.0,
        },
        "static": {
            "epsilon": 0.0,
            "learning_rate": 0.0,
            "discount_factor": 0.95,
            "epsilon_decay": 1.0,
            "epsilon_min": 0.0,
        },
        "threshold": {
            "epsilon": 0.0,
            "learning_rate": 0.0,
            "discount_factor": 0.95,
            "epsilon_decay": 1.0,
            "epsilon_min": 0.0,
        },
    }
    defaults = algorithm_defaults[normalize_algorithm(algorithm)]
    configured = (getattr(args, "config_data", {}).get("rl", {}).get("algorithms", {}) or {}).get(
        normalize_algorithm(algorithm),
        {},
    )
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, best_params.get(key, configured.get(key, fallback)))


def _cleanup_experiment_request(
    loop: OrchestrationLoop,
    request_id: str,
    reason: str = "experiment_completed",
) -> bool:
    """Remove a benchmark/live experiment request from every assigned node."""
    was_active = request_id in loop.active_requests
    cancelled = bool(loop.cancel_request(request_id, reason=reason)) if was_active else False
    for node in loop.nodes.values():
        loop._teardown_request_on_node(node, request_id)
    return cancelled


def _session_root(session_id: str) -> Path:
    """Return the base directory for one session."""
    root = SESSIONS_DIR / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


async def _run_single_experiment(
    args: argparse.Namespace,
    *,
    algorithm: str,
    seed: int,
    session_id: str,
) -> dict[str, object]:
    """Run one live experiment and return a machine-readable summary."""
    node_endpoints = [node.strip() for node in args.nodes.split(",") if node.strip()]
    if not node_endpoints:
        raise ValueError("No node endpoints provided")

    rl_enabled = not args.no_rl
    _apply_rl_defaults(args, rl_enabled, algorithm)

    session_root = _session_root(session_id)
    runs_root = session_root / "runs"
    experiment_name = f"{args.experiment_name}_{algorithm}_seed{seed}"
    output_root = Path(args.output_dir) if args.output_dir and args.output_dir != str(RUNS_DIR) else runs_root

    print(f"\n{'=' * 60}")
    print(f"Session: {session_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {args.mode}")
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    print(f"Nodes: {len(node_endpoints)}")
    print(f"{'=' * 60}")

    logger_instance = ExperimentLogger(
        experiment_name=experiment_name,
        output_dir=str(output_root),
    )
    logger_instance._ensure_initialized()

    persistence_dir = Path(args.persistence_dir) if args.persistence_dir else session_root / "persistence"
    persistence = configure_persistence(
        PersistenceConfig(
            output_dir=persistence_dir,
            session_id=session_id,
            enable_requests=True,
            enable_configs=True,
            enable_assignments=True,
            enable_metrics=True,
            enable_rl_decisions=True,
            enable_errors=True,
            enable_slo_violations=True,
            enable_tenant_fairness=True,
            enable_leader_events=True,
            enable_node_plans=True,
        )
    )

    initial_quantiles = parse_quantiles(args.initial_quantiles)
    input_schedule = parse_input_schedule(args.input_schedule)
    if input_schedule and args.schedule_segment_iterations <= 0:
        raise ValueError("--schedule-segment-iterations must be positive")
    loop = OrchestrationLoop(
        config=OrchestrationLoopConfig(
            poll_interval_seconds=args.poll_interval,
            enable_rl=rl_enabled,
            rl_alpha=args.learning_rate,
            rl_gamma=args.discount_factor,
            rl_epsilon=args.epsilon,
            rl_epsilon_decay=args.epsilon_decay,
            rl_min_epsilon=args.epsilon_min,
            rl_exploration_decay_steps=(
                args.exploration_decay_steps
                if args.exploration_decay_steps > 0
                else args.iterations
            ),
            rl_seed=seed,
            rl_training=args.policy_mode == "train",
            rl_warm_start=args.warm_start,
            initial_coverage_quantile=initial_quantiles[0],
            initial_sample_quantile=initial_quantiles[1],
            initial_freshness_quantile=initial_quantiles[2],
            cost_model_rates=cost_model_rates_from_mapping(getattr(args, "config_data", {}).get("cost_model", {})),
        ),
        logger=logger_instance,
        persistence=persistence,
    )

    for index, endpoint in enumerate(node_endpoints):
        loop.register_node(f"node-{index + 1}", endpoint)

    coverage_min, coverage_max = parse_range(args.coverage_range)
    sample_min, sample_max = parse_range(args.sample_range)
    freshness_min, freshness_max = parse_range(args.freshness_range)

    request = AnalyticsRequest(
        service_type=args.service_type,
        coverage_range=(coverage_min, coverage_max),
        sample_range=(sample_min, sample_max),
        freshness_range=(freshness_min, freshness_max),
        profile_name=args.profile_name,
        tenant_id=args.tenant_id,
        priority=args.priority,
        algorithm=algorithm,
        resource_limits=ResourceLimits(
            cpu_max_percent=args.cpu_max,
            memory_max_percent=args.memory_max,
        ),
        input_multiplier=args.input_multiplier,
    )

    effective = loop.submit_request(request)
    if args.policy_path:
        loop.load_rl_agent(request.request_id, args.policy_path)
        if args.policy_mode == "evaluate":
            loop.reset_rl_run_metrics(request.request_id)
    policy_fingerprint_before = loop.get_rl_policy_fingerprints().get(request.request_id) if rl_enabled else None

    stop_event = asyncio.Event()

    def _signal_handler(_sig, _frame):
        stop_event.set()

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    iteration = 0
    max_iterations = args.iterations if args.iterations > 0 else float("inf")

    try:
        while not stop_event.is_set() and iteration < max_iterations:
            if input_schedule:
                phase_index = (iteration // args.schedule_segment_iterations) % len(input_schedule)
                loop.set_request_input_multiplier(request.request_id, input_schedule[phase_index])
            iteration += 1
            start_time = time.time()
            metrics = await loop.run_once()
            elapsed_ms = (time.time() - start_time) * 1000.0

            rl_metrics = loop.get_rl_metrics().get(request.request_id, {})
            epsilon = rl_metrics.get("epsilon", rl_metrics.get("exploration_rate", 0.0))
            print(
                f"[{iteration:4d}] "
                f"active={metrics.active_nodes}/{metrics.total_nodes} "
                f"requests={metrics.requests_processed} "
                f"time={elapsed_ms:.0f}ms "
                f"algorithm={rl_metrics.get('algorithm', algorithm)} "
                f"eps={epsilon:.3f}"
            )

            wait_time = max(0.0, args.poll_interval - (time.time() - start_time))
            if wait_time > 0 and not stop_event.is_set():
                await asyncio.sleep(wait_time)
    except BaseException:
        with contextlib.suppress(Exception):
            _cleanup_experiment_request(loop, request.request_id, reason="experiment_aborted")
        with contextlib.suppress(Exception):
            logger_instance.close()
        with contextlib.suppress(Exception):
            persistence.close()
        with contextlib.suppress(Exception):
            await loop.close_async()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    rl_metrics = loop.get_rl_metrics().get(request.request_id, {})
    if rl_enabled and args.policy_mode == "train":
        loop.complete_rl_trajectories()
        rl_metrics = loop.get_rl_metrics().get(request.request_id, {})
    policy_fingerprint_after = loop.get_rl_policy_fingerprints().get(request.request_id) if rl_enabled else None
    model_dir = Path(logger_instance.output_dir) / "rl_models"
    if rl_enabled and args.policy_mode == "train":
        loop.save_rl_agents(str(model_dir))
    saved_policy_artifacts = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in sorted(model_dir.glob("*"))
        if path.is_file()
    ]
    runtime_modes = _runtime_modes_from_loop(loop)
    cleanup_performed = _cleanup_experiment_request(loop, request.request_id)

    summary = {
        "session_id": session_id,
        "experiment_name": experiment_name,
        "algorithm": rl_metrics.get("algorithm", algorithm),
        "requested_algorithm": rl_metrics.get("requested_algorithm", algorithm),
        "runtime_mode": _runtime_mode_label(runtime_modes),
        "runtime_modes": runtime_modes,
        "profile_name": args.profile_name,
        "rl_params_source": args.rl_params_source,
        "learning_rate": args.learning_rate,
        "discount_factor": args.discount_factor,
        "exploration_rate_initial": args.epsilon,
        "exploration_decay": args.epsilon_decay,
        "exploration_min": args.epsilon_min,
        "exploration_decay_steps": loop.config.rl_exploration_decay_steps,
        "slo_window_decisions": loop.config.slo_window_decisions,
        "policy_version": rl_metrics.get("policy_version", "unknown"),
        "policy_mode": args.policy_mode,
        "policy_source": args.policy_path or ("latest-global" if args.warm_start else "fresh"),
        "policy_source_sha256": _sha256_file(Path(args.policy_path)) if args.policy_path else None,
        "policy_fingerprint_before": policy_fingerprint_before,
        "policy_fingerprint_after": policy_fingerprint_after,
        "policy_unchanged": policy_fingerprint_before == policy_fingerprint_after,
        "seed": seed,
        "iterations": iteration,
        "request_id": request.request_id,
        "effective_coverage": effective.target_coverage,
        "effective_sample": effective.target_sample,
        "effective_freshness": effective.target_freshness,
        "assigned_nodes": effective.assigned_node_count,
        "input_multiplier": effective.input_multiplier,
        "input_schedule": input_schedule,
        "schedule_segment_iterations": args.schedule_segment_iterations if input_schedule else None,
        "initial_quantiles": list(initial_quantiles),
        "fallback_reason": rl_metrics.get("fallback_reason"),
        "total_reward": rl_metrics.get("total_reward", 0.0),
        "steps": rl_metrics.get("steps", 0),
        "states_visited": rl_metrics.get("states_visited", 0),
        "epsilon": rl_metrics.get("epsilon", rl_metrics.get("exploration_rate", 0.0)),
        "learning_metrics": {
            key: rl_metrics[key]
            for key in (
                "loss",
                "actor_loss",
                "critic_loss",
                "entropy",
                "ppo_updates",
                "rollout_pending",
                "replay_size",
            )
            if key in rl_metrics
        },
        "cleanup_performed": cleanup_performed,
        "output_dir": logger_instance.output_dir,
        "persistence_dir": str(persistence.config.output_dir),
        "model_dir": str(model_dir),
        "saved_policy_artifacts": saved_policy_artifacts,
    }

    summary_path = Path(logger_instance.output_dir) / "run_summary.json"
    manifest_path = Path(logger_instance.output_dir) / "run_manifest.json"
    node_manifest = []
    for node in loop.nodes.values():
        metrics = node.last_metrics or {}
        node_manifest.append(
            {
                "node_id": node.node_id,
                "endpoint": node.endpoint,
                "machine_id": metrics.get("machine_id"),
                "runtime_mode": metrics.get("runtime_mode"),
                "max_analytics_per_node": metrics.get("max_analytics_per_node"),
            }
        )
    request_contract = {
        "service_type": request.service_type,
        "coverage_range": list(request.coverage_range),
        "sample_range": list(request.sample_range),
        "freshness_range": list(request.freshness_range),
        "input_multiplier": request.input_multiplier,
        "input_schedule": input_schedule,
        "schedule_segment_iterations": args.schedule_segment_iterations if input_schedule else None,
        "resource_limits": request.resource_limits.to_dict(),
        "placement_limits": request.placement_limits.to_dict(),
    }
    repository_root = REPO_ROOT
    manifest = build_run_manifest(
        repository_root=repository_root,
        config_path=Path(args.config).resolve() if args.config else None,
        dataset_path=repository_root / "data" / "seville_bus",
        command=[sys.executable, *sys.argv],
        summary=summary,
        request_contract=request_contract,
        nodes=node_manifest,
    )
    write_run_manifest(manifest_path, manifest)
    summary["manifest_path"] = str(manifest_path)
    summary["code_fingerprint"] = manifest["code"]["sha256"]
    summary["dataset_fingerprint"] = manifest["dataset"]["sha256"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger_instance.close()
    persistence.close()
    await loop.close_async()
    return summary


def _write_benchmark_summary(base_dir: Path, results: list[dict[str, object]]) -> None:
    """Persist benchmark comparison results as JSON and CSV."""
    base_dir.mkdir(parents=True, exist_ok=True)
    json_path = base_dir / "benchmark_summary.json"
    csv_path = base_dir / "benchmark_summary.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    _write_benchmark_figures(base_dir, results)


def _runtime_modes_from_loop(loop: OrchestrationLoop) -> list[str]:
    """Return sorted runtime modes reported by registered nodes."""
    modes = {str((node.last_metrics or {}).get("runtime_mode") or "unknown") for node in loop.nodes.values()}
    return sorted(modes) or ["unknown"]


def _runtime_mode_label(modes: list[str]) -> str:
    """Render a compact runtime-mode label for summaries."""
    clean = sorted({str(mode or "unknown") for mode in modes}) or ["unknown"]
    return clean[0] if len(clean) == 1 else "mixed:" + ",".join(clean)


def _write_benchmark_figures(base_dir: Path, results: list[dict[str, object]]) -> list[str]:
    """Render lightweight benchmark-level comparison figures."""
    completed = [row for row in results if int(row.get("steps", 0) or 0) > 0]
    if not completed:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    by_algorithm: dict[str, list[dict[str, object]]] = {}
    for row in completed:
        by_algorithm.setdefault(str(row.get("algorithm", "unknown")), []).append(row)

    colors = {
        "qlearning": "#3f5f7f",
        "dqn": "#5b7f63",
        "ppo": "#a97845",
        "unknown": "#52606d",
    }
    saved: list[str] = []
    base_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    labels = sorted(by_algorithm)
    reward_means = []
    reward_stds = []
    for label in labels:
        rewards = [float(row.get("total_reward", 0.0) or 0.0) for row in by_algorithm[label]]
        reward_means.append(sum(rewards) / len(rewards))
        if len(rewards) > 1:
            mean = reward_means[-1]
            reward_stds.append((sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)) ** 0.5)
        else:
            reward_stds.append(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(
        labels,
        reward_means,
        yerr=reward_stds,
        capsize=4,
        color=[colors.get(label, colors["unknown"]) for label in labels],
    )
    ax.set_ylabel("Mean Total Reward")
    ax.set_xlabel("Algorithm")
    path = base_dir / "benchmark_reward_mean.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label in labels:
        rows = sorted(by_algorithm[label], key=lambda row: int(row.get("seed", 0) or 0))
        x = [int(row.get("seed", 0) or 0) for row in rows]
        y = [float(row.get("total_reward", 0.0) or 0.0) for row in rows]
        ax.plot(x, y, marker="o", linewidth=1.8, label=label, color=colors.get(label, colors["unknown"]))
    ax.set_xlabel("Seed")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="best")
    path = base_dir / "benchmark_reward_by_seed.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    state_means = [
        sum(float(row.get("states_visited", 0.0) or 0.0) for row in by_algorithm[label]) / len(by_algorithm[label])
        for label in labels
    ]
    ax.bar(labels, state_means, color=[colors.get(label, colors["unknown"]) for label in labels])
    ax.set_ylabel("Mean States Visited")
    ax.set_xlabel("Algorithm")
    path = base_dir / "benchmark_states_visited.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(path))

    return saved


def _console_log_paths(args: argparse.Namespace) -> list[Path]:
    """Return automatic console log destinations for CLI execution."""
    if not args.session_id:
        args.session_id = generate_session_id()

    if args.mode == "benchmark":
        return [_session_root(args.session_id) / "benchmark" / "console.log"]

    return [_session_root(args.session_id) / "console.log"]


@contextlib.contextmanager
def _capture_console(paths: list[Path]):
    """Mirror stdout/stderr to log files while preserving normal console output."""
    if not paths:
        yield
        return

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with contextlib.ExitStack() as stack:
        handles = []
        unique_paths = list(dict.fromkeys(paths))
        for path in unique_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles.append(stack.enter_context(open(path, "w", encoding="utf-8")))

        sys.stdout = _TeeStream(original_stdout, *handles)
        sys.stderr = _TeeStream(original_stderr, *handles)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


async def run_experiment(args: argparse.Namespace) -> None:
    """Entry point for live and benchmark execution modes."""
    if args.mode == "benchmark" and args.iterations <= 0:
        raise ValueError("--iterations must be > 0 in benchmark mode")

    algorithms = resolve_algorithms(args.mode, args.algorithm, getattr(args, "config_data", None))
    seeds = parse_seeds(args.seeds)
    learned_algorithms = [name for name in algorithms if name not in {"static", "threshold"}]
    if args.policy_mode == "evaluate" and learned_algorithms and not args.policy_path:
        raise ValueError("--policy-path is required to evaluate a frozen learned policy")
    if args.policy_mode == "evaluate" and len(learned_algorithms) > 1:
        raise ValueError("Frozen evaluation accepts one learned algorithm and one matching --policy-path per run")

    if args.mode == "live":
        session_id = args.session_id or generate_session_id()
        summary = await _run_single_experiment(
            args,
            algorithm=algorithms[0],
            seed=seeds[0],
            session_id=session_id,
        )
        print(f"\nLive run summary written to: {summary['output_dir']}")
        return

    benchmark_group = args.session_id or generate_session_id()
    benchmark_root = _session_root(benchmark_group) / "benchmark"
    results: list[dict[str, object]] = []

    for algorithm in algorithms:
        for seed in seeds:
            run_args = deepcopy(args)
            run_session_id = f"{benchmark_group}_{algorithm}_s{seed}"
            summary = await _run_single_experiment(
                run_args,
                algorithm=algorithm,
                seed=seed,
                session_id=run_session_id,
            )
            summary["benchmark_group_id"] = benchmark_group
            results.append(summary)

    _write_benchmark_summary(benchmark_root, results)
    print(f"\nBenchmark summary written to: {benchmark_root}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    console_logs = _console_log_paths(args)
    with _capture_console(console_logs):
        asyncio.run(run_experiment(args))
        print("\nConsole log written to:")
        for path in console_logs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
