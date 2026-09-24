#!/usr/bin/env python3
"""Run train, fixed-policy selection, and frozen evaluation as one campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = [sys.executable, "-m", "argos.experiment"]
RUN_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
LEARNERS = ("qlearning", "dqn", "ppo")
FIXED_LEVELS = (0.0, 1.0)
FIXED_CANDIDATES = {
    f"grid_c{coverage:g}_s{sample:g}_f{freshness:g}": f"{coverage:g},{sample:g},{freshness:g}"
    for coverage in FIXED_LEVELS
    for sample in FIXED_LEVELS
    for freshness in FIXED_LEVELS
}
FIXED_CANDIDATES["grid_c0.5_s0.5_f0.5"] = "0.5,0.5,0.5"
DEFAULT_INPUT_MULTIPLIERS = {
    "lax-background": 2,
    "standard-operations": 4,
    "aggressive-incident": 8,
    "cost-sensitive": 4,
    "short-burst": 8,
}


@dataclass(frozen=True)
class Profile:
    name: str
    payload: dict[str, Any]


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _seeds(raw: str) -> list[int]:
    return [int(value) for value in _csv(raw)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profiles(config: dict[str, Any], names: list[str]) -> list[Profile]:
    configured = {
        str(item.get("name")): dict(item.get("payload") or {})
        for item in ((config.get("live_trial") or {}).get("profiles") or [])
    }
    missing = [name for name in names if name not in configured]
    if missing:
        raise ValueError(f"Unknown profiles: {', '.join(missing)}")
    return [Profile(name, configured[name]) for name in names]


def _write_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _require_clean_git_tree() -> str:
    """Return HEAD and reject campaigns whose exact code is not committed."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot establish campaign Git provenance: {exc}") from exc
    if len(commit) != 40:
        raise RuntimeError(f"Invalid Git commit: {commit!r}")
    if status:
        raise RuntimeError(
            "Canonical campaigns require a clean committed tree. "
            f"Changed paths:\n{status}"
        )
    return commit


def _validate_coverage_reachability(profiles: list[Profile], node_count: int) -> None:
    """Reject contracts with no physically reachable coverage on this cluster."""
    if node_count < 1:
        raise ValueError("At least one node endpoint is required")
    reachable = tuple(count / node_count for count in range(1, node_count + 1))
    for profile in profiles:
        low = float(profile.payload["coverage_min"])
        high = float(profile.payload["coverage_max"])
        if not any(low - 1e-9 <= value <= high + 1e-9 for value in reachable):
            raise ValueError(
                f"Profile {profile.name} has coverage [{low}, {high}] but "
                f"{node_count} nodes only provide {reachable}"
            )


def _record_excluded_attempt(
    index: dict[str, Any],
    *,
    session_id: str,
    reason: str,
    attempt: int,
    quarantine: Optional[Path],
) -> None:
    """Record every discarded execution attempt in the campaign index."""
    index.setdefault("excluded_runs", []).append(
        {
            "session_id": session_id,
            "attempt": attempt,
            "exclusion_reason": reason,
            "quarantine_path": str(quarantine or ""),
            "excluded_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _summary_path(session_id: str) -> Path:
    return ROOT / "data" / "sessions" / session_id / "benchmark" / "benchmark_summary.json"


def _quarantine_run_artifacts(summary_path: Path, *, session_id: str) -> Optional[Path]:
    """Move an incomplete run aside so a retry starts with clean persistence."""
    sources: list[tuple[str, Path]] = [("benchmark", summary_path.parents[1])]
    if summary_path.is_file():
        try:
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = rows[0] if len(rows) == 1 else {}
        except (json.JSONDecodeError, OSError, TypeError):
            summary = {}
        for key in ("persistence_dir", "output_dir"):
            raw_path = str(summary.get(key, "")).strip()
            if raw_path:
                path = Path(raw_path)
                sources.append((key, path if path.is_absolute() else ROOT / path))

    existing: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    data_root = (ROOT / "data").resolve()
    for label, source in sources:
        resolved = source.resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_relative_to(data_root):
            continue
        seen.add(resolved)
        existing.append((label, resolved))
    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = data_root / "quarantine" / "incomplete_campaign_runs" / f"{session_id}_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for label, source in existing:
        shutil.move(str(source), str(destination / label))
    return destination


def _validated_summary(
    summary_path: Path,
    *,
    session_id: str,
    expected_iterations: int,
    expected_commit: Optional[str] = None,
    expected_runtime: Optional[str] = None,
) -> dict[str, Any]:
    summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(summary_rows) != 1:
        raise RuntimeError(f"Expected one summary row in {summary_path}")
    summary = summary_rows[0]
    observed_iterations = int(summary.get("iterations", -1))
    observed_steps = int(summary.get("steps", -1))
    if observed_iterations != expected_iterations or observed_steps != expected_iterations:
        raise RuntimeError(
            f"Incomplete run {session_id}: iterations={observed_iterations}, "
            f"steps={observed_steps}, expected={expected_iterations}"
        )
    manifest_path = Path(summary.get("manifest_path", ""))
    if not manifest_path.is_file():
        raise RuntimeError(f"Run did not produce a provenance manifest: {session_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    git = manifest.get("git") or {}
    if not git.get("commit") or git.get("dirty") is not False:
        raise RuntimeError(f"Run has invalid or dirty Git provenance: {session_id}")
    if expected_commit and git.get("commit") != expected_commit:
        raise RuntimeError(
            f"Run commit mismatch for {session_id}: "
            f"expected {expected_commit}, observed {git.get('commit')}"
        )
    if expected_runtime and summary.get("runtime_mode") != expected_runtime:
        raise RuntimeError(
            f"Run runtime mismatch for {session_id}: expected {expected_runtime}, "
            f"observed {summary.get('runtime_mode')}"
        )
    if summary.get("policy_mode") == "evaluate" and not summary.get("policy_unchanged"):
        raise RuntimeError(f"Frozen evaluation mutated its policy: {session_id}")
    return summary


def _validate_loaded_policy(
    summary: dict[str, Any],
    *,
    session_id: str,
    policy_path: Path,
    expected_policy_fingerprint: Optional[str],
) -> None:
    """Verify artifact bytes and the train-to-evaluation fingerprint chain."""
    expected_hash = _sha256_file(policy_path)
    if summary.get("policy_source_sha256") != expected_hash:
        raise RuntimeError(f"Loaded policy hash mismatch for {session_id}")
    observed_fingerprint = str(summary.get("policy_fingerprint_before") or "")
    if not expected_policy_fingerprint or observed_fingerprint != expected_policy_fingerprint:
        raise RuntimeError(
            f"Loaded policy fingerprint mismatch for {session_id}: "
            f"expected {expected_policy_fingerprint!r}, observed {observed_fingerprint!r}"
        )


def _import_fixed_selections(
    index_paths: list[Path],
    *,
    forbidden_seeds: set[int],
) -> dict[str, dict[str, Any]]:
    imported: dict[str, dict[str, Any]] = {}
    campaigns_root = (ROOT / "data" / "campaigns").resolve()
    for index_path in index_paths:
        resolved_index = index_path.resolve()
        if not resolved_index.is_relative_to(campaigns_root):
            raise RuntimeError(f"Fixed-policy source must live under {campaigns_root}: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not payload.get("completed_at"):
            raise RuntimeError(f"Fixed-policy source campaign is incomplete: {index_path}")
        source_tuning_seeds = {int(seed) for seed in payload.get("tuning_seeds", [])}
        if not source_tuning_seeds or source_tuning_seeds & forbidden_seeds:
            raise RuntimeError(f"Fixed-policy source has missing or overlapping tuning seeds: {index_path}")
        for profile, raw_selection in (payload.get("fixed_selection") or {}).items():
            tuning_runs = [
                row
                for row in payload.get("runs", [])
                if row.get("role") == "fixed_tune" and row.get("profile") == profile
            ]
            if not tuning_runs:
                raise RuntimeError(
                    f"Fixed-policy source has no real fixed_tune runs for {profile}: {index_path}"
                )
            selection = dict(raw_selection)
            candidate = str(selection.get("candidate", ""))
            quantiles = str(selection.get("quantiles", ""))
            if candidate not in FIXED_CANDIDATES or FIXED_CANDIDATES[candidate] != quantiles:
                raise RuntimeError(f"Invalid fixed-policy selection for {profile} in {index_path}")
            existing = imported.get(profile)
            if existing and existing["quantiles"] != quantiles:
                raise RuntimeError(f"Conflicting fixed-policy selections for {profile}")
            selection["source_campaign_id"] = payload.get("campaign_id", "")
            selection["source_campaign_index"] = str(resolved_index)
            selection["source_fixed_tune_run_count"] = len(tuning_runs)
            imported[profile] = selection
    return imported


def _amend_seed_schedule(index: dict[str, Any], seeds: list[int]) -> bool:
    previous = [int(seed) for seed in index.get("seeds", [])]
    if previous == seeds:
        return False
    if not set(seeds).issubset(previous):
        raise RuntimeError("A seed-schedule amendment can only retain existing campaign seeds")

    excluded_ids = {row["session_id"] for row in index.get("excluded_runs", [])}
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = list(index.get("excluded_runs", []))
    for row in index["runs"]:
        if int(row["seed"]) in seeds:
            retained.append(row)
        elif row["session_id"] not in excluded_ids:
            excluded.append({**row, "exclusion_reason": "seed_schedule_reduced_before_evaluation"})
    index["runs"] = retained
    index["excluded_runs"] = excluded
    index["seed_schedule_amendments"] = [
        *index.get("seed_schedule_amendments", []),
        {
            "previous_seeds": previous,
            "effective_seeds": seeds,
            "reason": "seed_schedule_reduced_before_evaluation",
            "amended_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    index["seeds"] = seeds
    return True


def _run(
    *,
    args: argparse.Namespace,
    index: dict[str, Any],
    index_path: Path,
    profile: Profile,
    seed: int,
    controller: str,
    role: str,
    iterations: int,
    policy_mode: str,
    initial_quantiles: str = "0.5,0.5,0.5",
    policy_path: Optional[Path] = None,
    expected_policy_fingerprint: Optional[str] = None,
    training_seed: Optional[int] = None,
    input_schedule: Optional[str] = None,
    label: Optional[str] = None,
) -> dict[str, Any]:
    run_label = label or controller
    session_id = f"{args.campaign_id}_{role}_{profile.name}_{run_label}_s{seed}"
    summary_path = _summary_path(session_id)
    existing = next((row for row in index["runs"] if row["session_id"] == session_id), None)
    if args.resume and existing and summary_path.exists():
        try:
            resumed_summary = _validated_summary(
                summary_path,
                session_id=session_id,
                expected_iterations=iterations,
                expected_commit=index.get("git_commit"),
                expected_runtime=args.runtime,
            )
            if policy_path:
                _validate_loaded_policy(
                    resumed_summary,
                    session_id=session_id,
                    policy_path=policy_path,
                    expected_policy_fingerprint=expected_policy_fingerprint,
                )
        except RuntimeError as exc:
            print(f"Discarding incomplete resume record: {exc}", flush=True)
            index["runs"] = [row for row in index["runs"] if row["session_id"] != session_id]
            destination = _quarantine_run_artifacts(summary_path, session_id=session_id)
            _record_excluded_attempt(
                index,
                session_id=session_id,
                reason=str(exc),
                attempt=0,
                quarantine=destination,
            )
            _write_index(index_path, index)
        else:
            return existing
    elif args.resume and summary_path.exists():
        destination = _quarantine_run_artifacts(summary_path, session_id=session_id)
        _record_excluded_attempt(
            index,
            session_id=session_id,
            reason="unindexed_run_found_before_resume",
            attempt=0,
            quarantine=destination,
        )
        _write_index(index_path, index)
        print(f"Quarantined unindexed run before retry: {destination}", flush=True)

    payload = profile.payload
    input_multiplier = int(payload.get("input_multiplier", DEFAULT_INPUT_MULTIPLIERS.get(profile.name, 4)))
    command = [
        *RUN_EXPERIMENT,
        "--config",
        str(args.config),
        "--mode",
        "benchmark",
        "--nodes",
        args.nodes,
        "--session-id",
        session_id,
        "--experiment-name",
        session_id,
        "--iterations",
        str(iterations),
        "--poll-interval",
        str(args.poll_interval),
        "--algorithm",
        controller,
        "--seeds",
        str(seed),
        "--coverage-range",
        f"{payload['coverage_min']},{payload['coverage_max']}",
        "--sample-range",
        f"{payload['sample_min']},{payload['sample_max']}",
        "--freshness-range",
        f"{payload['freshness_min']},{payload['freshness_max']}",
        "--input-multiplier",
        str(input_multiplier),
        "--input-schedule",
        input_schedule or args.input_schedule,
        "--schedule-segment-iterations",
        str(args.schedule_segment_iterations),
        "--initial-quantiles",
        initial_quantiles,
        "--profile-name",
        profile.name,
        "--tenant-id",
        f"campaign-{profile.name}",
        "--priority",
        str(payload.get("priority", "standard")),
        "--cpu-max",
        str((payload.get("resource_limits") or {}).get("cpu_max_percent", 80.0)),
        "--memory-max",
        str((payload.get("resource_limits") or {}).get("memory_max_percent", 85.0)),
        "--rl-params-source",
        "default",
        "--exploration-decay-steps",
        str(iterations if policy_mode == "train" else args.train_iterations),
        "--policy-mode",
        policy_mode,
    ]
    if policy_path:
        command.extend(["--policy-path", str(policy_path)])

    print("$", " ".join(command), flush=True)
    if not args.dry_run:
        for attempt in range(1, args.run_attempts + 1):
            try:
                subprocess.run(command, cwd=ROOT, env=RUN_ENV, check=True)
                summary = _validated_summary(
                    summary_path,
                    session_id=session_id,
                    expected_iterations=iterations,
                    expected_commit=index.get("git_commit"),
                    expected_runtime=args.runtime,
                )
            except (RuntimeError, subprocess.CalledProcessError) as exc:
                destination = _quarantine_run_artifacts(summary_path, session_id=session_id)
                _record_excluded_attempt(
                    index,
                    session_id=session_id,
                    reason=str(exc),
                    attempt=attempt,
                    quarantine=destination,
                )
                _write_index(index_path, index)
                if attempt == args.run_attempts:
                    raise
                print(
                    f"Run attempt {attempt}/{args.run_attempts} failed for {session_id}: {exc}. "
                    f"Quarantined at {destination}; retrying.",
                    flush=True,
                )
            else:
                break
        if policy_path:
            _validate_loaded_policy(
                summary,
                session_id=session_id,
                policy_path=policy_path,
                expected_policy_fingerprint=expected_policy_fingerprint,
            )
    else:
        summary = {"total_reward": 0.0, "model_dir": ""}

    record = {
        "session_id": session_id,
        "role": role,
        "controller": run_label,
        "algorithm": controller,
        "profile": profile.name,
        "seed": seed,
        "training_seed": training_seed,
        "policy_mode": policy_mode,
        "initial_quantiles": initial_quantiles,
        "policy_path": str(policy_path or ""),
        "policy_sha256": str(summary.get("policy_source_sha256", "")),
        "policy_fingerprint_before": str(summary.get("policy_fingerprint_before", "")),
        "policy_fingerprint_after": str(summary.get("policy_fingerprint_after", "")),
        "source_policy_fingerprint": str(expected_policy_fingerprint or ""),
        "summary_path": str(summary_path),
        "total_reward": float(summary.get("total_reward", 0.0)),
        "model_dir": str(summary.get("model_dir", "")),
    }
    index["runs"] = [row for row in index["runs"] if row["session_id"] != session_id]
    index["runs"].append(record)
    _write_index(index_path, index)
    return record


def _policy_artifact(record: dict[str, Any]) -> Path:
    model_dir = Path(record["model_dir"])
    candidates = sorted(model_dir.glob(f"*_{record['algorithm']}.*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one policy artifact in {model_dir}, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--runtime", choices=("thread", "process"), required=True)
    parser.add_argument("--profiles", default="aggressive-incident")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--evaluation-seeds", default="111,222,333,444,555")
    parser.add_argument("--tuning-seeds", default="101,202,303")
    parser.add_argument("--learners", default=",".join(LEARNERS))
    parser.add_argument("--train-iterations", type=int, default=2048)
    parser.add_argument("--tune-iterations", type=int, default=64)
    parser.add_argument("--eval-iterations", type=int, default=64)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--input-schedule", default="2,16,64,4")
    parser.add_argument("--evaluation-input-schedule", default="3,12,48,6")
    parser.add_argument("--schedule-segment-iterations", type=int, default=16)
    parser.add_argument("--fixed-selection-index", action="append", type=Path, default=[])
    parser.add_argument("--run-attempts", type=int, default=3)
    parser.add_argument("--amend-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.run_attempts < 1:
        parser.error("--run-attempts must be at least 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    profiles = _profiles(config, _csv(args.profiles))
    seeds = _seeds(args.seeds)
    evaluation_seeds = _seeds(args.evaluation_seeds)
    tuning_seeds = _seeds(args.tuning_seeds)
    if len(seeds) != 5 or len(evaluation_seeds) != 5:
        raise ValueError("Canonical campaigns require exactly five training and evaluation seeds")
    if len(tuning_seeds) != 3:
        raise ValueError("Canonical campaigns require exactly three fixed-policy tuning seeds")
    if len(seeds) != len(evaluation_seeds):
        raise ValueError("Training and evaluation seed schedules must have the same length")
    if any(len(set(group)) != len(group) for group in (seeds, evaluation_seeds, tuning_seeds)):
        raise ValueError("Training, evaluation, and tuning seed schedules must not contain duplicates")
    seed_groups = [set(seeds), set(evaluation_seeds), set(tuning_seeds)]
    if any(seed_groups[left] & seed_groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Training, evaluation, and fixed-policy tuning seeds must be disjoint")
    learners = _csv(args.learners)
    if any(name not in LEARNERS for name in learners):
        raise ValueError(f"Learners must be a subset of {LEARNERS}")
    imported_fixed_selections = _import_fixed_selections(
        args.fixed_selection_index,
        forbidden_seeds=set(seeds) | set(evaluation_seeds),
    )
    node_count = len(_csv(args.nodes))
    _validate_coverage_reachability(profiles, node_count)
    schedule_length = len(_csv(args.input_schedule))
    evaluation_schedule_length = len(_csv(args.evaluation_input_schedule))
    if evaluation_schedule_length != schedule_length:
        raise ValueError("Training and evaluation input schedules must have the same number of phases")
    if args.evaluation_input_schedule == args.input_schedule:
        raise ValueError("Evaluation input schedule must be held out from training")
    cycle_iterations = schedule_length * args.schedule_segment_iterations
    if cycle_iterations <= 0:
        raise ValueError("Input schedule and segment length must define a positive cycle")
    segment_seconds = args.schedule_segment_iterations * args.poll_interval
    if segment_seconds < 5.0:
        raise ValueError("Each input-schedule phase must last at least 5 seconds")
    for name, value in (
        ("train", args.train_iterations),
        ("tune", args.tune_iterations),
        ("evaluate", args.eval_iterations),
    ):
        if value % cycle_iterations != 0:
            raise ValueError(f"{name} iterations ({value}) must be a multiple of schedule cycle ({cycle_iterations})")

    campaign_commit = "dry-run" if args.dry_run else _require_clean_git_tree()

    campaign_root = ROOT / "data" / "campaigns" / args.campaign_id
    index_path = campaign_root / "campaign_index.json"
    if args.resume and index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("evaluation_seeds") != evaluation_seeds:
            raise RuntimeError("Evaluation seed schedule differs from the existing campaign index")
        if index.get("evaluation_input_schedule") != args.evaluation_input_schedule:
            raise RuntimeError("Held-out evaluation schedule differs from the existing campaign index")
        if index.get("runtime") != args.runtime:
            raise RuntimeError("Runtime differs from the existing campaign index")
        protocol_fields = {
            "profiles": [profile.name for profile in profiles],
            "seeds": seeds,
            "tuning_seeds": tuning_seeds,
            "learners": learners,
            "input_schedule": args.input_schedule,
            "train_iterations": args.train_iterations,
            "tune_iterations": args.tune_iterations,
            "eval_iterations": args.eval_iterations,
            "schedule_segment_iterations": args.schedule_segment_iterations,
            "fixed_candidate_count": len(FIXED_CANDIDATES),
        }
        mismatched = {
            key: (index.get(key), expected)
            for key, expected in protocol_fields.items()
            if index.get(key) != expected
        }
        if mismatched:
            raise RuntimeError(f"Protocol differs from the existing campaign index: {mismatched}")
        if index.get("git_commit") != campaign_commit:
            raise RuntimeError("Cannot resume a canonical campaign under a different Git commit")
        if index.get("seeds") != seeds:
            if not args.amend_seeds:
                raise RuntimeError(
                    "Campaign seed schedule differs from the existing index. "
                    "Use --amend-seeds only to reduce the schedule before evaluation."
                )
            if _amend_seed_schedule(index, seeds):
                _write_index(index_path, index)
    else:
        index = {
            "schema_version": "argos.campaign-index.v3",
            "campaign_id": args.campaign_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profiles": [profile.name for profile in profiles],
            "runtime": args.runtime,
            "seeds": seeds,
            "evaluation_seeds": evaluation_seeds,
            "tuning_seeds": tuning_seeds,
            "learners": learners,
            "input_schedule": args.input_schedule,
            "evaluation_input_schedule": args.evaluation_input_schedule,
            "schedule_segment_iterations": args.schedule_segment_iterations,
            "fixed_candidate_design": "eight contract vertices plus midpoint",
            "fixed_candidate_count": len(FIXED_CANDIDATES),
            "train_iterations": args.train_iterations,
            "tune_iterations": args.tune_iterations,
            "eval_iterations": args.eval_iterations,
            "runs": [],
            "excluded_runs": [],
            "fixed_selection": {},
            "git_commit": campaign_commit,
        }
        _write_index(index_path, index)

    trained: dict[tuple[str, int, str], tuple[Path, str]] = {}
    for profile in profiles:
        for seed in seeds:
            for learner in learners:
                record = _run(
                    args=args,
                    index=index,
                    index_path=index_path,
                    profile=profile,
                    seed=seed,
                    controller=learner,
                    role="train",
                    iterations=args.train_iterations,
                    policy_mode="train",
                )
                if not args.dry_run:
                    artifact = _policy_artifact(record)
                    record["policy_artifact"] = str(artifact)
                    record["policy_sha256"] = _sha256_file(artifact)
                    fingerprint = str(record.get("policy_fingerprint_after") or "")
                    if not fingerprint:
                        raise RuntimeError(f"Training run has no final policy fingerprint: {record['session_id']}")
                    trained[(profile.name, seed, learner)] = (artifact, fingerprint)
                    _write_index(index_path, index)

    for profile in profiles:
        if profile.name in imported_fixed_selections:
            index["fixed_selection"][profile.name] = imported_fixed_selections[profile.name]
            _write_index(index_path, index)
            continue
        candidate_rewards: dict[str, list[float]] = {name: [] for name in FIXED_CANDIDATES}
        for seed in tuning_seeds:
            candidates = list(FIXED_CANDIDATES)
            random.Random(seed * 917 + sum(map(ord, profile.name))).shuffle(candidates)
            for candidate in candidates:
                quantiles = FIXED_CANDIDATES[candidate]
                record = _run(
                    args=args,
                    index=index,
                    index_path=index_path,
                    profile=profile,
                    seed=seed,
                    controller="static",
                    role="fixed_tune",
                    iterations=args.tune_iterations,
                    policy_mode="evaluate",
                    initial_quantiles=quantiles,
                    label=f"fixed_{candidate}",
                )
                candidate_rewards[candidate].append(record["total_reward"])
        selected = max(candidate_rewards, key=lambda name: sum(candidate_rewards[name]) / len(candidate_rewards[name]))
        index["fixed_selection"][profile.name] = {
            "candidate": selected,
            "quantiles": FIXED_CANDIDATES[selected],
            "tuning_mean_reward": sum(candidate_rewards[selected]) / len(candidate_rewards[selected]),
        }
        _write_index(index_path, index)

    for profile in profiles:
        selected_quantiles = index["fixed_selection"][profile.name]["quantiles"]
        for training_seed, evaluation_seed in zip(seeds, evaluation_seeds):
            controllers = [*learners, "static", "threshold", "best_fixed"]
            random.Random(evaluation_seed * 1009 + sum(map(ord, profile.name))).shuffle(controllers)
            for controller in controllers:
                algorithm = "static" if controller == "best_fixed" else controller
                quantiles = selected_quantiles if controller == "best_fixed" else "0.5,0.5,0.5"
                policy_record = (
                    trained.get((profile.name, training_seed, controller))
                    if controller in learners
                    else None
                )
                policy_path = policy_record[0] if policy_record else None
                expected_fingerprint = policy_record[1] if policy_record else None
                _run(
                    args=args,
                    index=index,
                    index_path=index_path,
                    profile=profile,
                    seed=evaluation_seed,
                    controller=algorithm,
                    role="evaluate",
                    iterations=args.eval_iterations,
                    policy_mode="evaluate",
                    initial_quantiles=quantiles,
                    policy_path=policy_path,
                    expected_policy_fingerprint=expected_fingerprint,
                    training_seed=training_seed,
                    input_schedule=args.evaluation_input_schedule,
                    label=controller,
                )

    index["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_index(index_path, index)
    print(index_path)


if __name__ == "__main__":
    main()
