# =============================================================================
# ARGOS - Live Traffic Trial Driver
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Run a realistic long-running traffic trial against the ARGOS REST API.

The script acts as an external client: it optionally registers nodes, starts the
orchestration loop, submits analytics jobs over time, samples cluster/RL state,
and writes a JSONL event stream plus a compact summary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from argos.common.constants import REPO_ROOT
from argos.config import DEFAULT_CONFIG_PATH, load_project_config, node_registration_list

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "evaluation" / "live_trials"
DEFAULT_PERSISTENCE_DIR = Path(os.getenv("ARGOS_PERSISTENCE_DIR", str(REPO_ROOT / "data" / "persistence")))


@dataclass(frozen=True)
class ApiResponse:
    status: int
    payload: dict[str, Any]


class OrchestratorAPIClient:
    """Small stdlib-only API client for the orchestrator."""

    def __init__(self, base_url: str, api_token: str = "", timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["X-API-Key"] = api_token

    def request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> ApiResponse:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=body, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                return ApiResponse(status=response.status, payload=data)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                data = {"detail": raw}
            return ApiResponse(status=exc.code, payload=data)

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = self.request("GET", path, params=params)
        if response.status >= 400:
            raise RuntimeError(f"GET {path} failed with HTTP {response.status}: {response.payload}")
        return response.payload

    def post(self, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = self.request("POST", path, payload=payload or {})
        if response.status >= 400:
            raise RuntimeError(f"POST {path} failed with HTTP {response.status}: {response.payload}")
        return response.payload

    def delete(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        response = self.request("DELETE", path, params=params)
        if response.status == 404:
            return {"status": "not_found"}
        if response.status >= 400:
            raise RuntimeError(f"DELETE {path} failed with HTTP {response.status}: {response.payload}")
        return response.payload


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_nodes_from_config(config_path: Path) -> list[str]:
    """Load node registrations from a ARGOS YAML config."""
    return node_registration_list(load_project_config(config_path, required=True))


def parse_node_registration(raw: str) -> tuple[str, str]:
    """Parse a node registration in the form node-id=http://host:port."""
    if "=" not in raw:
        raise ValueError(f"Invalid --node value: {raw}. Expected node-id=http://host:port")
    node_id, endpoint = raw.split("=", 1)
    node_id = node_id.strip()
    endpoint = endpoint.strip()
    if not node_id or not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Invalid --node value: {raw}. Expected node-id=http://host:port")
    return node_id, endpoint


def traffic_profiles(config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Return the configured rotating set of realistic request profiles."""
    configured = ((config or {}).get("live_trial", {}) or {}).get("profiles") or []
    if configured:
        profiles = []
        for item in configured:
            name = str(item.get("name", "")).strip()
            payload = dict(item.get("payload", {}) or {})
            if name and payload:
                profiles.append({"name": name, "payload": payload})
        if profiles:
            return profiles

    return [
        {
            "name": "lax-background",
            "payload": {
                "service_type": "geo_heatmap",
                "coverage_min": 0.25,
                "coverage_max": 0.45,
                "sample_min": 0.20,
                "sample_max": 0.45,
                "freshness_min": 90,
                "freshness_max": 180,
                "tenant_id": "city-dashboard",
                "priority": "best_effort",
                "algorithm": "auto",
                "resource_limits": {"cpu_max_percent": 70.0, "memory_max_percent": 80.0},
            },
        },
        {
            "name": "standard-operations",
            "payload": {
                "service_type": "geo_heatmap",
                "coverage_min": 0.45,
                "coverage_max": 0.75,
                "sample_min": 0.35,
                "sample_max": 0.65,
                "freshness_min": 30,
                "freshness_max": 90,
                "tenant_id": "operations",
                "priority": "standard",
                "algorithm": "auto",
                "resource_limits": {"cpu_max_percent": 80.0, "memory_max_percent": 85.0},
            },
        },
        {
            "name": "aggressive-incident",
            "payload": {
                "service_type": "geo_heatmap",
                "coverage_min": 0.75,
                "coverage_max": 1.00,
                "sample_min": 0.70,
                "sample_max": 1.00,
                "freshness_min": 5,
                "freshness_max": 30,
                "tenant_id": "incident-response",
                "priority": "critical",
                "algorithm": "auto",
                "resource_limits": {"cpu_max_percent": 85.0, "memory_max_percent": 88.0},
            },
        },
        {
            "name": "cost-sensitive",
            "payload": {
                "service_type": "geo_heatmap",
                "coverage_min": 0.33,
                "coverage_max": 0.67,
                "sample_min": 0.25,
                "sample_max": 0.55,
                "freshness_min": 60,
                "freshness_max": 150,
                "cost_budget": 0.05,
                "tenant_id": "planning",
                "priority": "standard",
                "algorithm": "auto",
                "resource_limits": {"cpu_max_percent": 65.0, "memory_max_percent": 78.0},
            },
        },
        {
            "name": "short-burst",
            "payload": {
                "service_type": "geo_heatmap",
                "coverage_min": 0.60,
                "coverage_max": 0.95,
                "sample_min": 0.60,
                "sample_max": 0.90,
                "freshness_min": 10,
                "freshness_max": 45,
                "tenant_id": "mobility",
                "priority": "standard",
                "algorithm": "auto",
                "resource_limits": {"cpu_max_percent": 82.0, "memory_max_percent": 86.0},
            },
        },
    ]


def submission_payload(profile: dict[str, Any], tenant_prefix: str) -> dict[str, Any]:
    """Build an auditable API payload with the stable profile identifier."""
    payload = dict(profile["payload"])
    payload["profile_name"] = str(profile["name"])
    payload["tenant_id"] = f"{tenant_prefix}-{payload['tenant_id']}"
    return payload


def _next_interval_minutes(args: argparse.Namespace, rng: random.Random) -> float:
    """Return minutes until the next job submission."""
    if args.traffic_model == "uniform":
        return rng.uniform(args.job_interval_min_minutes, args.job_interval_max_minutes)
    return args.job_interval_minutes


def _request_duration_minutes(args: argparse.Namespace, rng: random.Random) -> Optional[float]:
    """Return request lifetime in minutes, or None when it lasts until cleanup."""
    if args.request_duration_mode == "uniform":
        return rng.uniform(args.request_duration_min_minutes, args.request_duration_max_minutes)
    return None


def _cancel_job(
    *,
    client: OrchestratorAPIClient,
    events_path: Path,
    job: dict[str, Any],
    elapsed_s: float,
    reason: str,
) -> None:
    """Cancel a submitted job once and record the lifecycle event."""
    if job.get("cancelled_at_s") is not None:
        return
    request_id = job.get("request_id")
    if not request_id:
        return
    try:
        status_before_cancel = client.get(f"/job/{request_id}")
    except RuntimeError as exc:
        status_before_cancel = {"status": "status_error", "detail": str(exc)}
    result = client.delete(f"/job/{request_id}", params={"reason": reason})
    job["cancelled_at_s"] = round(elapsed_s, 3)
    job["cancel_reason"] = reason
    job["status_before_cancel"] = status_before_cancel
    job["cancel_response"] = result
    write_jsonl(
        events_path,
        {
            "timestamp": utc_now(),
            "type": "job_cancelled",
            "elapsed_s": round(elapsed_s, 3),
            "request_id": request_id,
            "reason": reason,
            "status_before_cancel": status_before_cancel,
            "response": result,
        },
    )


def write_jsonl(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def compact_status(cluster: dict[str, Any], jobs: dict[str, Any], rl: dict[str, Any]) -> str:
    """Format one operator-friendly status line."""
    agents = rl.get("agents", {})
    steps = sum(int(agent.get("steps", 0) or 0) for agent in agents.values())
    rewards = [float(agent.get("total_reward", 0.0) or 0.0) for agent in agents.values()]
    reward = sum(rewards) / len(rewards) if rewards else 0.0
    return (
        f"nodes={cluster.get('active_nodes', 0)}/{cluster.get('total_nodes', 0)} "
        f"jobs={jobs.get('total', 0)} loop={cluster.get('orchestrator_running')} "
        f"last_iter={cluster.get('last_iteration_ms', 0.0):.0f}ms "
        f"rl_agents={len(agents)} rl_steps={steps} avg_reward={reward:+.3f}"
    )


def collect_snapshot(client: OrchestratorAPIClient) -> dict[str, Any]:
    """Collect cluster, node, job and RL state."""
    return {
        "cluster": client.get("/cluster/status"),
        "nodes": client.get("/cluster/nodes"),
        "jobs": client.get("/jobs"),
        "rl": client.get("/rl/metrics"),
        "recent_decisions": client.get("/rl/decisions", params={"limit": 20}),
        "slo_violations": client.get("/history/slo-violations", params={"limit": 50}),
    }


def collect_trial_slo_violations(
    client: OrchestratorAPIClient,
    submitted_jobs: list[dict[str, Any]],
) -> tuple[int, dict[str, int]]:
    """Return SLO violations only for requests submitted by this trial."""
    by_request: dict[str, int] = {}
    for job in submitted_jobs:
        request_id = str(job.get("request_id") or "").strip()
        if not request_id:
            continue
        response = client.get("/history/slo-violations", params={"request_id": request_id, "limit": 2000})
        by_request[request_id] = int(response.get("total", 0) or 0)
    return sum(by_request.values()), by_request


def wait_for_active_nodes(
    client: OrchestratorAPIClient,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Wait until at least one registered node has fresh metrics."""
    deadline = time.monotonic() + timeout_seconds
    last_status: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        last_status = client.get("/cluster/status")
        if int(last_status.get("active_nodes", 0) or 0) > 0:
            return last_status
        time.sleep(max(0.5, min(poll_seconds, 5.0)))
    return last_status


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.yaml override. Also loads node registrations when provided explicitly.",
    )
    pre_args, _unknown = pre_parser.parse_known_args(argv)
    config = load_project_config(pre_args.config or DEFAULT_CONFIG_PATH)
    trial_cfg = config.get("live_trial", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}
    runtime_paths = runtime_cfg.get("paths", {}) or {}
    cluster_cfg = config.get("cluster", {}) or {}
    orchestrator_cfg = cluster_cfg.get("orchestrator", {}) or {}

    host = str(orchestrator_cfg.get("host", "127.0.0.1"))
    port = int(orchestrator_cfg.get("port", 8001))
    default_orchestrator_url = f"http://{host}:{port}"
    evaluation_output_dir = runtime_paths.get("evaluation_output_dir")
    output_root = Path(evaluation_output_dir) / "live_trials" if evaluation_output_dir else DEFAULT_OUTPUT_ROOT
    persistence_dir = Path(runtime_paths.get("runtime_persistence_dir", DEFAULT_PERSISTENCE_DIR))

    parser = argparse.ArgumentParser(
        description="Run a live ARGOS traffic trial against the orchestrator API",
        parents=[pre_parser],
    )
    parser.add_argument("--orchestrator-url", default=default_orchestrator_url)
    parser.add_argument("--api-token", default=os.getenv("ORCHESTRATOR_API_TOKEN", ""))
    parser.add_argument("--duration-minutes", type=float, default=float(trial_cfg.get("duration_minutes", 30.0)))
    parser.add_argument(
        "--traffic-model",
        choices=["fixed", "uniform"],
        default=str(trial_cfg.get("traffic_model", "fixed")),
        help="Request arrival model: fixed interval or seeded uniform interval.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=int(trial_cfg.get("random_seed", 20260428)),
        help="Seed for stochastic live-trial arrivals/durations.",
    )
    parser.add_argument("--job-interval-minutes", type=float, default=float(trial_cfg.get("job_interval_minutes", 6.0)))
    parser.add_argument(
        "--job-interval-min-minutes",
        type=float,
        default=float(trial_cfg.get("job_interval_min_minutes", trial_cfg.get("job_interval_minutes", 6.0))),
    )
    parser.add_argument(
        "--job-interval-max-minutes",
        type=float,
        default=float(trial_cfg.get("job_interval_max_minutes", trial_cfg.get("job_interval_minutes", 6.0))),
    )
    parser.add_argument(
        "--request-duration-mode",
        choices=["until_end", "uniform"],
        default=str(trial_cfg.get("request_duration_mode", "until_end")),
        help="Request lifetime model before cleanup.",
    )
    parser.add_argument(
        "--request-duration-min-minutes",
        type=float,
        default=float(trial_cfg.get("request_duration_min_minutes", 5.0)),
    )
    parser.add_argument(
        "--request-duration-max-minutes",
        type=float,
        default=float(trial_cfg.get("request_duration_max_minutes", 10.0)),
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=float(trial_cfg.get("poll_interval_seconds", 10.0)),
    )
    parser.add_argument("--print-every-seconds", type=float, default=float(trial_cfg.get("print_every_seconds", 60.0)))
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=float(trial_cfg.get("startup_wait_seconds", 30.0)),
    )
    parser.add_argument("--output-dir", type=Path, default=output_root)
    parser.add_argument("--persistence-dir", type=Path, default=persistence_dir)
    parser.add_argument("--tenant-prefix", default=str(trial_cfg.get("tenant_prefix", "live-trial")))
    parser.add_argument("--traffic-scenario", default=str(trial_cfg.get("traffic_scenario", "")))
    parser.add_argument("--runtime-mode", default=str(trial_cfg.get("runtime_mode", "")))
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        help="Optional node registration: node-id=http://host:port. Can be repeated.",
    )
    parser.add_argument("--keep-jobs", action="store_true", help="Leave submitted jobs active at the end")
    parser.add_argument(
        "--require-frozen-policy",
        action="store_true",
        help="Fail unless every admitted request used an unchanged frozen policy for at least one decision.",
    )
    args = parser.parse_args(argv)
    args.config_data = config
    args.config_provided = pre_args.config is not None
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.duration_minutes <= 0:
        raise ValueError("--duration-minutes must be > 0")
    if args.job_interval_minutes <= 0:
        raise ValueError("--job-interval-minutes must be > 0")
    if args.job_interval_min_minutes <= 0 or args.job_interval_max_minutes <= 0:
        raise ValueError("--job-interval-min-minutes and --job-interval-max-minutes must be > 0")
    if args.job_interval_min_minutes > args.job_interval_max_minutes:
        raise ValueError("--job-interval-min-minutes must be <= --job-interval-max-minutes")
    if args.request_duration_min_minutes <= 0 or args.request_duration_max_minutes <= 0:
        raise ValueError("--request-duration-min-minutes and --request-duration-max-minutes must be > 0")
    if args.request_duration_min_minutes > args.request_duration_max_minutes:
        raise ValueError("--request-duration-min-minutes must be <= --request-duration-max-minutes")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be > 0")

    client = OrchestratorAPIClient(args.orchestrator_url, api_token=args.api_token)
    rng = random.Random(args.random_seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"

    node_registrations = list(args.node)
    if args.config and args.config_provided:
        node_registrations.extend(load_nodes_from_config(args.config))

    submitted_jobs: list[dict[str, Any]] = []
    stop_requested = False

    def _request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print(f"Run: {run_id}")
    print(f"Output: {run_dir}")
    print(f"Orchestrator: {args.orchestrator_url}")
    print(f"Traffic: {args.traffic_model} arrivals, {args.request_duration_mode} durations, seed={args.random_seed}")

    try:
        for raw_node in node_registrations:
            node_id, endpoint = parse_node_registration(raw_node)
            result = client.post("/nodes/register", {"node_id": node_id, "endpoint": endpoint})
            print(f"registered {node_id}: {result.get('status')}")
            write_jsonl(
                events_path,
                {"timestamp": utc_now(), "type": "node_registered", "node_id": node_id, "endpoint": endpoint},
            )

        status = client.get("/cluster/status")
        if not status.get("orchestrator_running"):
            print("starting orchestration loop")
            client.post("/orchestrator/start", {})
        status = wait_for_active_nodes(
            client,
            timeout_seconds=args.startup_wait_seconds,
            poll_seconds=args.poll_interval_seconds,
        )
        if int(status.get("active_nodes", 0) or 0) == 0:
            raise RuntimeError("No active nodes after startup wait. Register/start nodes before running a live trial.")

        profiles = traffic_profiles(args.config_data)
        duration_s = args.duration_minutes * 60.0
        start = time.monotonic()
        next_job_at = 0.0
        next_print_at = 0.0
        sample_count = 0

        while not stop_requested:
            elapsed_s = time.monotonic() - start
            if elapsed_s >= duration_s:
                break

            if not args.keep_jobs:
                for job in submitted_jobs:
                    expires_at_s = job.get("expires_at_s")
                    if expires_at_s not in (None, "") and elapsed_s >= float(expires_at_s):
                        _cancel_job(
                            client=client,
                            events_path=events_path,
                            job=job,
                            elapsed_s=elapsed_s,
                            reason="live_trial_duration_expired",
                        )

            if elapsed_s >= next_job_at:
                profile_index = len(submitted_jobs) % len(profiles)
                profile = profiles[profile_index]
                payload = submission_payload(profile, args.tenant_prefix)
                result = client.post("/submit-job", payload)
                duration_minutes = _request_duration_minutes(args, rng)
                expires_at_s = elapsed_s + duration_minutes * 60.0 if duration_minutes is not None else None
                interval_minutes = _next_interval_minutes(args, rng)
                submitted_jobs.append(
                    {
                        "name": profile["name"],
                        "request_id": result.get("request_id"),
                        "submitted_at_s": round(elapsed_s, 3),
                        "duration_minutes": round(duration_minutes, 3) if duration_minutes is not None else "",
                        "expires_at_s": round(expires_at_s, 3) if expires_at_s is not None else "",
                        "next_interval_minutes": round(interval_minutes, 3),
                        "payload": payload,
                        "response": result,
                    }
                )
                ttl_text = f"{duration_minutes:.1f}m" if duration_minutes is not None else "until_end"
                print(
                    f"[{elapsed_s:7.1f}s] submitted {profile['name']} "
                    f"id={result.get('request_id')} status={result.get('status')} "
                    f"assigned={result.get('assigned_nodes')} ttl={ttl_text} "
                    f"next={interval_minutes:.1f}m reason={result.get('reason', '')}"
                )
                write_jsonl(
                    events_path,
                    {
                        "timestamp": utc_now(),
                        "type": "job_submitted",
                        "elapsed_s": round(elapsed_s, 3),
                        "profile": profile["name"],
                        "duration_minutes": round(duration_minutes, 3) if duration_minutes is not None else None,
                        "next_interval_minutes": round(interval_minutes, 3),
                        "response": result,
                    },
                )
                next_job_at = elapsed_s + interval_minutes * 60.0

            snapshot = collect_snapshot(client)
            sample_count += 1
            write_jsonl(
                events_path,
                {
                    "timestamp": utc_now(),
                    "type": "sample",
                    "elapsed_s": round(elapsed_s, 3),
                    "snapshot": snapshot,
                },
            )

            if elapsed_s >= next_print_at:
                print(f"[{elapsed_s:7.1f}s] {compact_status(snapshot['cluster'], snapshot['jobs'], snapshot['rl'])}")
                next_print_at += args.print_every_seconds

            sleep_s = min(args.poll_interval_seconds, max(0.0, duration_s - (time.monotonic() - start)))
            if sleep_s <= 0:
                break
            time.sleep(sleep_s)

        final_snapshot = collect_snapshot(client)
        if not args.keep_jobs:
            for job in submitted_jobs:
                _cancel_job(
                    client=client,
                    events_path=events_path,
                    job=job,
                    elapsed_s=time.monotonic() - start,
                    reason="cancelled_by_trial_cleanup",
                )

        final_rl = final_snapshot["rl"].get("agents", {})
        trial_slo_total, trial_slo_by_request = collect_trial_slo_violations(client, submitted_jobs)
        frozen_checks = []
        for job in submitted_jobs:
            status = dict(job.get("status_before_cancel") or {})
            lifecycle_status = str(status.get("status") or job.get("response", {}).get("status") or "")
            queued_only = lifecycle_status == "queued"
            frozen_checks.append(
                {
                    "request_id": job.get("request_id"),
                    "profile_name": job.get("name"),
                    "status_before_cancel": lifecycle_status,
                    "queued_only": queued_only,
                    "required": bool(status.get("frozen_policy_required", False)),
                    "loaded": bool(status.get("frozen_policy_loaded", False)),
                    "unchanged": bool(status.get("frozen_policy_unchanged", True)),
                    "steps": int(status.get("rl_steps", 0) or 0),
                }
            )
        duration_seconds = args.duration_minutes * 60.0
        tail_grace_seconds = max(3.0 * args.poll_interval_seconds, 30.0)
        frozen_failures = []
        tail_censored_requests = []
        for job, check in zip(submitted_jobs, frozen_checks):
            if check["queued_only"]:
                continue
            hard_defect = (
                not check["required"] or not check["loaded"] or not check["unchanged"]
            )
            zero_steps = check["steps"] < 1
            submitted_at = float(job.get("submitted_at_s") or 0.0)
            in_tail_window = (duration_seconds - submitted_at) <= tail_grace_seconds
            if hard_defect:
                frozen_failures.append(check)
            elif zero_steps and in_tail_window:
                # Admitted too close to shutdown to complete one frozen decision
                # epoch: a measurement boundary, not a policy failure.
                check["tail_censored"] = True
                tail_censored_requests.append(check)
            elif zero_steps:
                frozen_failures.append(check)
        summary = {
            "run_id": run_id,
            "started_at": datetime.fromtimestamp(time.time() - (time.monotonic() - start), timezone.utc).isoformat(),
            "finished_at": utc_now(),
            "traffic_scenario": args.traffic_scenario,
            "runtime_mode": args.runtime_mode,
            "duration_minutes_requested": args.duration_minutes,
            "traffic_model": args.traffic_model,
            "random_seed": args.random_seed,
            "job_interval_minutes": args.job_interval_minutes,
            "job_interval_min_minutes": args.job_interval_min_minutes,
            "job_interval_max_minutes": args.job_interval_max_minutes,
            "request_duration_mode": args.request_duration_mode,
            "request_duration_min_minutes": args.request_duration_min_minutes,
            "request_duration_max_minutes": args.request_duration_max_minutes,
            "poll_interval_seconds": args.poll_interval_seconds,
            "submitted_jobs": submitted_jobs,
            "submitted_status_counts": {
                status: sum(1 for job in submitted_jobs if job.get("response", {}).get("status") == status)
                for status in sorted({job.get("response", {}).get("status") for job in submitted_jobs})
                if status
            },
            "sample_count": sample_count,
            "final_cluster": final_snapshot["cluster"],
            "final_nodes": final_snapshot["nodes"],
            "final_jobs_total_before_cleanup": final_snapshot["jobs"].get("total", 0),
            "final_rl_agent_count": len(final_rl),
            "final_rl_steps": sum(int(agent.get("steps", 0) or 0) for agent in final_rl.values()),
            "frozen_policy_checks": frozen_checks,
            "frozen_policy_failures": frozen_failures,
            "frozen_policy_evaluated_requests": sum(
                not check["queued_only"] and not any(
                    (
                        not check["required"],
                        not check["loaded"],
                        not check["unchanged"],
                        check["steps"] < 1,
                    )
                )
                for check in frozen_checks
            ),
            "queued_without_policy_evaluation": sum(check["queued_only"] for check in frozen_checks),
            "tail_censored_requests": len(tail_censored_requests),
            "tail_censored_policy_checks": tail_censored_requests,
            "slo_violations_seen": trial_slo_total,
            "slo_violations_by_request": trial_slo_by_request,
            "historical_slo_violations_sampled": final_snapshot["slo_violations"].get("total", 0),
            "events_path": str(events_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary: {summary_path}")
        if args.require_frozen_policy and frozen_failures:
            raise RuntimeError(
                f"{len(frozen_failures)} admitted requests failed frozen-policy validation"
            )

        return 0
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
