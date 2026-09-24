#!/usr/bin/env python3
"""Run paired live trials of static and frozen learned controllers with isolated persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.run_controlled_campaign import _require_clean_git_tree, _sha256_file  # noqa: E402

SCENARIO_PARAMETERS = {
    "concurrency": {
        "interval_min": 0.5,
        "interval_max": 1.0,
        "duration_mode": "until_end",
        "request_duration_min": 20.0,
        "request_duration_max": 30.0,
    },
    "realistic": {
        "interval_min": 2.0,
        "interval_max": 4.0,
        "duration_mode": "uniform",
        "request_duration_min": 10.0,
        "request_duration_max": 15.0,
    },
}


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _int_csv(raw: str) -> list[int]:
    return [int(part) for part in _csv(raw)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _policy_map(
    index_paths: list[Path],
    *,
    training_seed: int,
    expected_commit: str,
    algorithm: str = "ppo",
    allow_commit_mismatch: bool = False,
) -> dict[str, str]:
    """Resolve one learned-policy artifact per profile from canonical training records."""
    mapping: dict[str, str] = {}
    for index_path in index_paths:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not index.get("completed_at"):
            raise RuntimeError(f"Controlled campaign is incomplete: {index_path}")
        if index.get("git_commit") != expected_commit and not allow_commit_mismatch:
            raise RuntimeError(
                f"Controlled campaign commit differs from live code: {index_path}. "
                "Artifact SHA-256 hashes are still enforced; pass "
                "--allow-controlled-commit-mismatch to proceed and record both commits."
            )
        for row in index.get("runs", []):
            if (
                row.get("role") != "train"
                or row.get("algorithm") != algorithm
                or int(row.get("seed", -1)) != training_seed
            ):
                continue
            profile = str(row.get("profile") or "")
            artifact = Path(str(row.get("policy_artifact") or ""))
            expected_hash = str(row.get("policy_sha256") or "")
            if not profile or not artifact.is_file():
                raise RuntimeError(f"Missing {algorithm} artifact for seed {training_seed}: {row}")
            if not expected_hash or _sha256_file(artifact) != expected_hash:
                raise RuntimeError(f"{algorithm} artifact hash mismatch: {artifact}")
            existing = mapping.get(profile)
            if existing and existing != str(artifact):
                raise RuntimeError(f"Conflicting {algorithm} artifacts for {profile}, seed {training_seed}")
            mapping[profile] = str(artifact.resolve())
    return mapping


def _archive_failed_attempt(campaign_root: Path, run_id: str, attempt_root: Path, attempt: int) -> None:
    """Move a failed attempt aside so evidence is retained without blocking retries."""
    failed_root = campaign_root / "runs" / "_failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    shutil.move(str(attempt_root), str(failed_root / f"{run_id}_attempt{attempt}_{stamp}"))


def _wait_for_api(url: str, process: subprocess.Popen, timeout_seconds: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() <= deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Orchestrator exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/cluster/status", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "orchestrator_running" in payload:
                return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise RuntimeError(f"Orchestrator did not become ready: {last_error}")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _latest_summary(run_root: Path) -> Path:
    summaries = sorted(run_root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one live summary below {run_root}, found {len(summaries)}")
    return summaries[0]


def _validate_summary(path: Path, *, variant: str, runtime: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("submitted_jobs"):
        raise RuntimeError(f"Live trial submitted no jobs: {path}")
    if variant != "static":
        failures = summary.get("frozen_policy_failures") or []
        if failures:
            raise RuntimeError(f"Frozen-policy failures in {path}: {failures[:1]}")
        if int(summary.get("frozen_policy_evaluated_requests", 0)) < 1:
            raise RuntimeError(f"Live trial evaluated no frozen {variant} request: {path}")
    final_nodes = (summary.get("final_nodes") or {}).get("nodes") or []
    observed_modes = {
        str(node.get("runtime_mode"))
        for node in final_nodes
        if isinstance(node, dict) and node.get("runtime_mode")
    }
    if observed_modes != {runtime}:
        raise RuntimeError(
            f"Live trial runtime mismatch in {path}: expected {runtime}, observed {sorted(observed_modes)}"
        )
    return summary


def _trial_command(
    *,
    args: argparse.Namespace,
    scenario: str,
    live_seed: int,
    variant: str,
    run_root: Path,
    persistence_dir: Path,
    orchestrator_url: str,
) -> list[str]:
    params = SCENARIO_PARAMETERS[scenario]
    command = [
        sys.executable,
        "-m",
        "argos.live_trial",
        "--config",
        str(args.config),
        "--orchestrator-url",
        orchestrator_url,
        "--duration-minutes",
        str(args.duration_minutes),
        "--traffic-model",
        "uniform",
        "--random-seed",
        str(live_seed),
        "--job-interval-min-minutes",
        str(params["interval_min"]),
        "--job-interval-max-minutes",
        str(params["interval_max"]),
        "--request-duration-mode",
        str(params["duration_mode"]),
        "--request-duration-min-minutes",
        str(params["request_duration_min"]),
        "--request-duration-max-minutes",
        str(params["request_duration_max"]),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--output-dir",
        str(run_root),
        "--persistence-dir",
        str(persistence_dir),
        "--tenant-prefix",
        f"{args.campaign_id}-{scenario}-{variant}-s{live_seed}",
        "--traffic-scenario",
        scenario,
        "--runtime-mode",
        args.runtime,
        "--no-generate-evidence",
    ]
    if variant != "static":
        command.append("--require-frozen-policy")
    if args.api_token:
        command.extend(["--api-token", args.api_token])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-index", action="append", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--runtime", choices=("thread", "process"), default="thread")
    parser.add_argument("--training-seeds", default="11,22,33,44,55")
    parser.add_argument("--live-seeds", default="1001,1002,1003,1004,1005")
    parser.add_argument("--scenarios", default="concurrency,realistic")
    parser.add_argument(
        "--variants",
        default="static,dqn,ppo",
        help="Comma-separated controller variants; 'static' plus any of ppo,dqn,qlearning.",
    )
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--api-token", default=os.getenv("ORCHESTRATOR_API_TOKEN", ""))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-controlled-commit-mismatch", action="store_true")
    args = parser.parse_args()

    training_seeds = _int_csv(args.training_seeds)
    live_seeds = _int_csv(args.live_seeds)
    scenarios = _csv(args.scenarios)
    if len(training_seeds) != 5 or len(live_seeds) != 5:
        raise ValueError("Canonical live campaigns require exactly five training and five live seeds")
    if len(set(training_seeds)) != 5 or len(set(live_seeds)) != 5:
        raise ValueError("Seed schedules contain duplicates")
    if any(scenario not in SCENARIO_PARAMETERS for scenario in scenarios):
        raise ValueError(f"Unknown live scenario in {scenarios}")
    if args.duration_minutes <= 0:
        raise ValueError("--duration-minutes must be positive")

    commit = _require_clean_git_tree()
    variant_list = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    allowed = {"static", "ppo", "dqn", "qlearning"}
    if not variant_list or any(v not in allowed for v in variant_list) or "static" not in variant_list:
        raise RuntimeError(f"--variants must include 'static' and only {sorted(allowed)}: {variant_list}")
    learned_variants = [v for v in variant_list if v != "static"]
    policy_maps = {
        algorithm: {
            seed: _policy_map(
                args.campaign_index,
                training_seed=seed,
                expected_commit=commit,
                algorithm=algorithm,
                allow_commit_mismatch=args.allow_controlled_commit_mismatch,
            )
            for seed in training_seeds
        }
        for algorithm in learned_variants
    }
    for algorithm in learned_variants:
        expected_profiles = set(policy_maps[algorithm][training_seeds[0]])
        if len(expected_profiles) != 5:
            raise RuntimeError(
                f"Expected five {algorithm} profile policies, found {sorted(expected_profiles)}"
            )
        if any(set(policy_maps[algorithm][seed]) != expected_profiles for seed in training_seeds):
            raise RuntimeError(f"{algorithm} policy maps differ across training seeds")

    campaign_root = ROOT / "data" / "live_campaigns" / args.campaign_id
    index_path = campaign_root / "campaign_index.json"
    if args.resume and index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected_protocol = {
            "git_commit": commit,
            "runtime": args.runtime,
            "training_seeds": training_seeds,
            "live_seeds": live_seeds,
            "scenarios": scenarios,
            "duration_minutes": args.duration_minutes,
        }
        mismatched = {
            field: (index.get(field), expected)
            for field, expected in expected_protocol.items()
            if index.get(field) != expected
        }
        if mismatched:
            commit_only = set(mismatched) == {"git_commit"}
            if commit_only and args.allow_controlled_commit_mismatch:
                extensions = index.setdefault("extension_commits", [])
                if commit not in extensions:
                    extensions.append(commit)
            else:
                raise RuntimeError(
                    f"Cannot resume live campaign under a different protocol: {mismatched}"
                )
    else:
        index = {
            "schema_version": "argos.live-campaign-index.v1",
            "campaign_id": args.campaign_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "runtime": args.runtime,
            "training_seeds": training_seeds,
            "live_seeds": live_seeds,
            "scenarios": scenarios,
            "duration_minutes": args.duration_minutes,
            "runs": [],
            "excluded_runs": [],
        }
        _write_json(index_path, index)

    orchestrator_url = f"http://127.0.0.1:{args.port}"
    for scenario in scenarios:
        for pair_index, (training_seed, live_seed) in enumerate(zip(training_seeds, live_seeds)):
            shift = pair_index % len(variant_list)
            variants = tuple(variant_list[shift:] + variant_list[:shift])
            for variant in variants:
                run_id = f"{scenario}_{variant}_train{training_seed}_live{live_seed}"
                existing = next((row for row in index["runs"] if row["run_id"] == run_id), None)
                if args.resume and existing and Path(existing["summary_path"]).is_file():
                    _validate_summary(
                        Path(existing["summary_path"]),
                        variant=variant,
                        runtime=args.runtime,
                    )
                    continue

                base_root = campaign_root / "runs" / run_id
                if base_root.exists():
                    _archive_failed_attempt(campaign_root, run_id, base_root, 0)
                last_error = None
                for attempt in (1, 2):
                    attempt_root = campaign_root / "runs" / run_id
                    run_root = attempt_root / "client"
                    persistence_dir = attempt_root / "persistence"
                    log_path = attempt_root / "orchestrator.log"
                    attempt_root.mkdir(parents=True, exist_ok=False)

                    env = dict(os.environ)
                    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(SRC)
                    env["ARGOS_SESSION_ID"] = run_id
                    env["ARGOS_PERSISTENCE_DIR"] = str(persistence_dir)
                    env["ARGOS_LIVE_ENABLE_RL"] = "1"
                    env["ARGOS_LIVE_RL_TRAINING"] = "0"
                    env["ARGOS_LIVE_FROZEN_ALGORITHM"] = variant
                    env["ARGOS_LIVE_INITIAL_QUANTILES"] = "0.5,0.5,0.5"
                    env["ARGOS_DEPLOYMENT_PROFILE"] = "local"
                    env["ARGOS_CONTROL_PLANE_ENABLED"] = "0"
                    env["ARGOS_CONTROL_PLANE_NATS_ENABLED"] = "0"
                    if variant != "static":
                        env["ARGOS_LIVE_FROZEN_POLICY_MAP"] = json.dumps(
                            policy_maps[variant][training_seed], sort_keys=True
                        )
                    else:
                        env.pop("ARGOS_LIVE_FROZEN_POLICY_MAP", None)
                        env.pop("ARGOS_LIVE_FROZEN_POLICY", None)

                    process = None
                    try:
                        with log_path.open("w", encoding="utf-8") as log_handle:
                            process = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-m",
                                    "uvicorn",
                                    "argos.orchestrator.api:app",
                                    "--host",
                                    "127.0.0.1",
                                    "--port",
                                    str(args.port),
                                ],
                                cwd=ROOT,
                                env=env,
                                stdout=log_handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                            _wait_for_api(orchestrator_url, process)
                            subprocess.run(
                                _trial_command(
                                    args=args,
                                    scenario=scenario,
                                    live_seed=live_seed,
                                    variant=variant,
                                    run_root=run_root,
                                    persistence_dir=persistence_dir,
                                    orchestrator_url=orchestrator_url,
                                ),
                                cwd=ROOT,
                                env={**os.environ, "PYTHONPATH": str(SRC)},
                                check=True,
                                timeout=args.duration_minutes * 60 + 600,
                            )
                    except Exception as exc:
                        last_error = exc
                        index["excluded_runs"].append(
                            {
                                "run_id": run_id,
                                "attempt": attempt,
                                "reason": f"{type(exc).__name__}: {exc}",
                                "path": str(attempt_root),
                                "excluded_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        _write_json(index_path, index)
                        _archive_failed_attempt(campaign_root, run_id, attempt_root, attempt)
                        continue
                    finally:
                        if process is not None:
                            _stop_process(process)

                    try:
                        summary_path = _latest_summary(run_root)
                        summary = _validate_summary(
                            summary_path,
                            variant=variant,
                            runtime=args.runtime,
                        )
                    except Exception as exc:
                        last_error = exc
                        index["excluded_runs"].append(
                            {
                                "run_id": run_id,
                                "attempt": attempt,
                                "reason": f"{type(exc).__name__}: {exc}",
                                "path": str(attempt_root),
                                "excluded_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        _write_json(index_path, index)
                        _archive_failed_attempt(campaign_root, run_id, attempt_root, attempt)
                        continue
                    record = {
                        "run_id": run_id,
                        "scenario": scenario,
                        "variant": variant,
                        "git_commit": commit,
                        "training_seed": training_seed,
                        "live_seed": live_seed,
                        "summary_path": str(summary_path),
                        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                        "persistence_dir": str(persistence_dir),
                        "submitted_jobs": len(summary["submitted_jobs"]),
                        "evaluated_requests": int(summary.get("frozen_policy_evaluated_requests", 0)),
                        "queued_requests": int(summary.get("queued_without_policy_evaluation", 0)),
                        "slo_violations": int(summary.get("slo_violations_seen", 0)),
                    }
                    index["runs"] = [row for row in index["runs"] if row["run_id"] != run_id]
                    index["runs"].append(record)
                    _write_json(index_path, index)
                    last_error = None
                    break
                if last_error is not None:
                    print(f"WARNING: giving up on {run_id} after 2 attempts: {last_error}", flush=True)

    index["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(index_path, index)
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
