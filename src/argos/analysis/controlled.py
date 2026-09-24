"""Analyze only frozen runs listed by an ARGOS campaign index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from scipy.stats import t


# Campaign indexes record absolute paths from the machine that produced the
# campaign. Paths are re-rooted at their "data/<campaign dir>" segment under
# --evidence-root so the analysis still works after the data/ directory is moved.
EVIDENCE_DIRS = {"campaigns", "live_campaigns", "sessions", "runs", "analysis", "quarantine"}
EVIDENCE_ROOT: Path | None = None


def _evidence_path(value: str | Path) -> Path:
    path = Path(value)
    if EVIDENCE_ROOT is None:
        return path
    parts = path.parts
    for position in range(len(parts) - 1):
        if parts[position] == "data" and parts[position + 1] in EVIDENCE_DIRS:
            return EVIDENCE_ROOT.joinpath(*parts[position:])
    return path if path.is_absolute() else EVIDENCE_ROOT / path

LEARNERS = ("qlearning", "dqn", "ppo")
BASELINES = ("static", "threshold", "best_fixed")
INDEX_PROTOCOL_FIELDS = (
    "seeds",
    "evaluation_seeds",
    "tuning_seeds",
    "learners",
    "input_schedule",
    "evaluation_input_schedule",
    "schedule_segment_iterations",
    "fixed_candidate_design",
    "fixed_candidate_count",
    "train_iterations",
    "tune_iterations",
    "eval_iterations",
    "git_commit",
)


def _jsonl(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise RuntimeError(f"Required evidence stream is missing: {path}")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _bootstrap_mean_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(resamples))
    low = estimates[int(0.025 * (resamples - 1))]
    high = estimates[int(0.975 * (resamples - 1))]
    return low, high


def _t_mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Student-t interval for the paired-seed mean."""
    if not values:
        return math.nan, math.nan
    center = mean(values)
    if len(values) == 1:
        return center, center
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    return center - critical * standard_error, center + critical * standard_error


def _rank_biserial(deltas: list[float]) -> float:
    nonzero = [value for value in deltas if value != 0]
    if not nonzero:
        return 0.0
    ordered = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(nonzero)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and abs(ordered[end][1]) == abs(ordered[cursor][1]):
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position][0]] = average_rank
        cursor = end
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value"])
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * row["p_value"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def _validate_index_compatibility(indexes: list[tuple[Path, dict[str, Any]]]) -> None:
    """Reject incomplete campaign indexes or incompatible protocols."""
    incomplete = [str(path) for path, index in indexes if not index.get("completed_at")]
    if incomplete:
        raise RuntimeError(f"Campaign indexes are incomplete: {incomplete}")

    reference_path, reference = indexes[0]
    for field in INDEX_PROTOCOL_FIELDS:
        expected = reference.get(field)
        if expected in (None, "", []):
            raise RuntimeError(f"Campaign index {reference_path} has no {field}")
        mismatches = [
            (str(path), index.get(field))
            for path, index in indexes[1:]
            if index.get(field) != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"Campaign indexes use different {field}: "
                f"reference={expected!r}, mismatches={mismatches}"
            )

    evaluation_seeds = [int(seed) for seed in reference["evaluation_seeds"]]
    if len(evaluation_seeds) != 5 or len(set(evaluation_seeds)) != 5:
        raise RuntimeError("Canonical analysis requires exactly five unique evaluation seeds")


def _run_metrics(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    summary_path = _evidence_path(record["summary_path"])
    summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(summary_rows) != 1:
        raise RuntimeError(f"Expected one summary in {summary_path}")
    summary = summary_rows[0]
    manifest_path = _evidence_path(summary["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "argos.run-manifest.v1":
        raise RuntimeError(f"Invalid manifest schema in {manifest_path}")
    git = manifest.get("git") or {}
    if not git.get("commit") or git.get("dirty") is not False:
        raise RuntimeError(f"Invalid Git provenance in {manifest_path}")
    if summary.get("policy_mode") != "evaluate":
        raise RuntimeError(f"Non-frozen run in evaluation set: {record['session_id']}")
    if not summary.get("policy_unchanged"):
        raise RuntimeError(f"Policy parameters changed during evaluation: {record['session_id']}")
    if record.get("controller") in LEARNERS:
        expected_policy_hash = str(record.get("policy_sha256") or "")
        observed_policy_hash = str(summary.get("policy_source_sha256") or "")
        if not expected_policy_hash or observed_policy_hash != expected_policy_hash:
            raise RuntimeError(f"Frozen policy hash mismatch for {record['session_id']}")
        expected_fingerprint = str(record.get("source_policy_fingerprint") or "")
        observed_fingerprint = str(summary.get("policy_fingerprint_before") or "")
        if not expected_fingerprint or observed_fingerprint != expected_fingerprint:
            raise RuntimeError(f"Train-to-evaluation fingerprint mismatch for {record['session_id']}")

    persistence = _evidence_path(summary["persistence_dir"])
    decisions = _jsonl(persistence / "rl_decisions.jsonl", required=True)
    request_metrics = _jsonl(persistence / "request_metrics.jsonl", required=True)
    violations = _jsonl(persistence / "slo_violations.jsonl", required=True)
    slo_epochs = _jsonl(persistence / "slo_decision_epochs.jsonl", required=True)
    decision_reward = sum(float(row.get("reward", 0.0)) for row in decisions)
    summary_reward = float(summary.get("total_reward", 0.0))
    if not math.isclose(decision_reward, summary_reward, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"Reward mismatch for {record['session_id']}: {decision_reward} != {summary_reward}")

    fidelity = [float(row["mean_spatial_fidelity"]) for row in request_metrics if row.get("mean_spatial_fidelity") is not None]
    hotspot = [float(row["mean_hotspot_recall"]) for row in request_metrics if row.get("mean_hotspot_recall") is not None]
    service_duty = [
        float(row["mean_service_duty_percent"])
        for row in request_metrics
        if row.get("mean_service_duty_percent") is not None
    ]
    observed_sample = sum(row.get("actual_sample") is not None for row in request_metrics)
    observed_freshness = sum(row.get("observed_freshness_age_max_s") is not None for row in request_metrics)
    violation_counts = Counter(str(row.get("metric")) for row in violations)
    steps = int(summary.get("steps", 0))
    if len(decisions) != steps or len(slo_epochs) != steps:
        raise RuntimeError(
            f"Incomplete decision evidence for {record['session_id']}: "
            f"steps={steps}, decisions={len(decisions)}, slo_epochs={len(slo_epochs)}"
        )
    epoch_by_iteration = {int(row["iteration"]): row for row in slo_epochs}
    if len(epoch_by_iteration) != len(slo_epochs):
        raise RuntimeError(f"Duplicate SLO decision epochs for {record['session_id']}")
    slo_window_size = int(summary.get("slo_window_decisions", 20))
    window: deque[int] = deque(maxlen=slo_window_size)
    for decision in sorted(decisions, key=lambda row: int(row["iteration"])):
        iteration = int(decision["iteration"])
        observed_recent = int((decision.get("state") or {}).get("recent_slo_violations", -1))
        if observed_recent != sum(window):
            raise RuntimeError(
                f"Recent-SLO state mismatch in {record['session_id']} iteration {iteration}"
            )
        epoch = epoch_by_iteration.get(iteration)
        if epoch is None or int(epoch.get("window_size", 0)) != slo_window_size:
            raise RuntimeError(
                f"Missing or invalid SLO epoch in {record['session_id']} iteration {iteration}"
            )
        window.append(1 if epoch.get("violated") else 0)
        if int(epoch.get("recent_violation_count", -1)) != sum(window):
            raise RuntimeError(
                f"SLO rolling count mismatch in {record['session_id']} iteration {iteration}"
            )
    row = {
        "session_id": record["session_id"],
        "controller": record["controller"],
        "algorithm": record["algorithm"],
        "profile": record["profile"],
        "seed": int(record["seed"]),
        "training_seed": (
            int(record["training_seed"])
            if record.get("training_seed") is not None
            else None
        ),
        "runtime": str(summary.get("runtime_mode", "unknown")),
        "steps": steps,
        "reward_total": summary_reward,
        "reward_per_step": summary_reward / steps if steps else math.nan,
        "quality_observations": len(fidelity),
        "quality_observation_ratio": len(fidelity) / len(request_metrics) if request_metrics else 0.0,
        "sample_observation_ratio": observed_sample / len(request_metrics) if request_metrics else 0.0,
        "freshness_observation_ratio": observed_freshness / len(request_metrics) if request_metrics else 0.0,
        "spatial_fidelity_mean": mean(fidelity) if fidelity else math.nan,
        "hotspot_recall_mean": mean(hotspot) if hotspot else math.nan,
        "service_duty_mean_percent": mean(service_duty) if service_duty else math.nan,
        "service_duty_max_percent": max(service_duty) if service_duty else math.nan,
        "slo_violation_events": len(violations),
        "cpu_violation_events": violation_counts["cpu_percent"],
        "memory_violation_events": violation_counts["memory_percent"],
        "coverage_violation_events": violation_counts["coverage"],
        "sample_violation_events": violation_counts["sample_rate"],
        "freshness_violation_events": violation_counts["freshness_age_seconds"],
        "git_commit": git["commit"],
        "code_fingerprint": summary["code_fingerprint"],
        "dataset_fingerprint": summary["dataset_fingerprint"],
    }
    return row, {
        "commit": git["commit"],
        "code": summary["code_fingerprint"],
        "dataset": summary["dataset_fingerprint"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-index", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-baseline", choices=BASELINES, default="best_fixed")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Directory containing the data/ tree of the campaign; recorded paths are re-rooted there",
    )
    args = parser.parse_args(argv)
    global EVIDENCE_ROOT
    EVIDENCE_ROOT = args.evidence_root.resolve() if args.evidence_root else None

    indexes = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.campaign_index
    ]
    _validate_index_compatibility(indexes)
    evaluation_seed_schedules = {
        tuple(int(seed) for seed in index.get("evaluation_seeds", []))
        for _path, index in indexes
    }
    records = [
        row
        for _path, index in indexes
        for row in index.get("runs", [])
        if row.get("role") == "evaluate"
    ]
    if not records:
        raise RuntimeError("Campaign index has no frozen evaluation runs")

    run_rows: list[dict[str, Any]] = []
    fingerprints: set[tuple[str, str, str]] = set()
    session_ids = [str(record.get("session_id")) for record in records]
    duplicates = [session_id for session_id, count in Counter(session_ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"Campaign index contains duplicate evaluation runs: {duplicates}")

    for record in records:
        row, fingerprint = _run_metrics(record)
        run_rows.append(row)
        fingerprints.add(
            (
                fingerprint["commit"],
                fingerprint["code"],
                fingerprint["dataset"],
            )
        )
    if len(fingerprints) != 1:
        raise RuntimeError(
            f"Campaign mixes {len(fingerprints)} commit/code/dataset fingerprints"
        )

    lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in run_rows:
        key = (row["runtime"], row["profile"], row["seed"], row["controller"])
        if key in lookup:
            raise RuntimeError(f"Duplicate evaluation cell: {key}")
        lookup[key] = row
    profile_deltas: list[dict[str, Any]] = []
    runtimes = sorted({row["runtime"] for row in run_rows})
    profiles = list(
        dict.fromkeys(
            profile
            for _path, index in indexes
            for profile in (index.get("profiles") or [])
        )
    ) or sorted({row["profile"] for row in run_rows})
    seeds = list(next(iter(evaluation_seed_schedules))) or sorted(
        {int(row["seed"]) for row in run_rows}
    )
    learners = list(
        dict.fromkeys(
            learner
            for _path, index in indexes
            for learner in (index.get("learners") or [])
        )
    ) or list(LEARNERS)
    expected_controllers = [*learners, *BASELINES]
    missing_cells = [
        (runtime, profile, seed, controller)
        for runtime in runtimes
        for profile in profiles
        for seed in seeds
        for controller in expected_controllers
        if (runtime, profile, seed, controller) not in lookup
    ]
    if missing_cells:
        raise RuntimeError(f"Campaign is missing {len(missing_cells)} evaluation cells, first={missing_cells[0]}")
    for runtime in runtimes:
        for profile in profiles:
            for seed in seeds:
                for learner in learners:
                    learned = lookup.get((runtime, profile, seed, learner))
                    if not learned:
                        continue
                    for baseline in BASELINES:
                        control = lookup.get((runtime, profile, seed, baseline))
                        if not control:
                            raise RuntimeError(f"Missing paired baseline cell: {(runtime, profile, seed, baseline)}")
                        if learned["steps"] != control["steps"]:
                            raise RuntimeError(
                                f"Paired steps differ for {(runtime, profile, seed, learner, baseline)}"
                            )
                        delta = learned["reward_per_step"] - control["reward_per_step"]
                        profile_deltas.append(
                            {
                                "runtime": runtime,
                                "profile": profile,
                                "seed": seed,
                                "learner": learner,
                                "baseline": baseline,
                                "learned_reward_per_step": learned["reward_per_step"],
                                "baseline_reward_per_step": control["reward_per_step"],
                                "delta_reward_per_step": delta,
                                "delta_percent": (
                                    100.0 * delta / abs(control["reward_per_step"])
                                    if control["reward_per_step"] != 0
                                    else math.nan
                                ),
                            }
                        )

    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in profile_deltas:
        grouped[(row["runtime"], row["seed"], row["learner"], row["baseline"])].append(
            row["delta_reward_per_step"]
        )
    seed_deltas = [
        {
            "runtime": key[0],
            "seed": key[1],
            "learner": key[2],
            "baseline": key[3],
            "profile_count": len(values),
            "mean_delta_reward_per_step": mean(values),
        }
        for key, values in sorted(grouped.items())
    ]
    incomplete_seed_rows = [
        row for row in seed_deltas if int(row["profile_count"]) != len(profiles)
    ]
    if incomplete_seed_rows:
        raise RuntimeError(f"Seed aggregates use incomplete profile sets: {incomplete_seed_rows[0]}")

    tests: list[dict[str, Any]] = []
    for runtime in runtimes:
        for learner in learners:
            values = [
                row["mean_delta_reward_per_step"]
                for row in seed_deltas
                if row["runtime"] == runtime
                and row["learner"] == learner
                and row["baseline"] == args.primary_baseline
            ]
            if not values:
                continue
            low, high = _t_mean_ci(values)
            tests.append(
                {
                    "runtime": runtime,
                    "learner": learner,
                    "baseline": args.primary_baseline,
                    "n_seeds": len(values),
                    "mean_delta_reward_per_step": mean(values),
                    "median_delta_reward_per_step": median(values),
                    "ci95_low": low,
                    "ci95_high": high,
                    "interval_method": "two-sided Student-t interval over paired seeds",
                    "rank_biserial": _rank_biserial(values),
                    "wins": sum(value > 0 for value in values),
                    "ties": sum(value == 0 for value in values),
                    "losses": sum(value < 0 for value in values),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "run_metrics.csv", run_rows)
    _write_csv(args.output_dir / "paired_profile_deltas.csv", profile_deltas)
    _write_csv(args.output_dir / "paired_seed_deltas.csv", seed_deltas)
    _write_csv(args.output_dir / "statistical_tests.csv", tests)
    analysis_manifest = {
        "schema_version": "argos.analysis-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_indexes": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, _index in indexes
        ],
        "primary_endpoint": "mean frozen-evaluation reward per step, averaged over profiles within seed",
        "experimental_unit": "seed",
        "primary_baseline": args.primary_baseline,
        "inference": (
            "Descriptive paired-seed analysis. No confirmatory p-values are "
            "reported because each comparison contains five seed pairs."
        ),
        "interval_method": "two-sided Student-t 95% interval over paired-seed means",
        "fingerprints": [
            dict(commit=commit, code=code, dataset=dataset)
            for commit, code, dataset in sorted(fingerprints)
        ],
        "run_count": len(run_rows),
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
