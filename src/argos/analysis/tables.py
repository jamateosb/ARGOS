"""Build the ARGOS result tables from the records of completed campaigns.

First stage of the results pipeline:

    campaign records        ->  argos.analysis.tables   ->  results/*/tables/*.csv
    results/*/tables/*.csv  ->  generate_figures.py     ->  results/figures/

It is normally run through scripts/analyze_results.py.

The script checks that the campaigns are complete and consistent (full design,
frozen evaluation, rewards that match the decision logs) and refuses to write
tables otherwise. The repository already contains the tables of the reference
evaluation, so this step is only needed after running new campaigns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from argos.common.constants import REPO_ROOT as ROOT

DEFAULT_EVIDENCE_ROOT = ROOT
DEFAULT_OUTPUT = ROOT / "results"

# Campaign records store absolute paths. Every recorded path is re-rooted at its
# "data/" segment under the evidence root, so campaign data can be moved.
EVIDENCE_ROOT = DEFAULT_EVIDENCE_ROOT

CONTROLLERS = ("static", "threshold", "best_fixed", "qlearning", "dqn", "ppo")
LEARNERS = ("qlearning", "dqn", "ppo")
PROFILES = (
    "lax-background",
    "standard-operations",
    "cost-sensitive",
    "short-burst",
    "aggressive-incident",
)
ACTIONS = (
    "hold",
    "increase_coverage",
    "decrease_coverage",
    "increase_sample",
    "decrease_sample",
    "decrease_freshness",
    "increase_freshness",
)
RUNTIMES = ("thread", "process")
SCENARIOS = ("realistic", "concurrency")
VARIANTS = ("static", "dqn", "ppo")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise RuntimeError(f"Required evidence stream is missing: {path}")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EVIDENCE_DIRS = {"campaigns", "live_campaigns", "sessions", "runs", "analysis", "quarantine"}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    parts = value.parts
    for position in range(len(parts) - 1):
        if parts[position] == "data" and parts[position + 1] in EVIDENCE_DIRS:
            return EVIDENCE_ROOT.joinpath(*parts[position:])
    return value if value.is_absolute() else EVIDENCE_ROOT / value


def _tenant_profiles(config_path: Path) -> dict[str, str]:
    """Map each workload profile's tenant name to the profile name (config.yaml live_trial.profiles)."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = (config.get("live_trial") or {}).get("profiles") or []
    return {str(item["payload"]["tenant_id"]): str(item["name"]) for item in profiles}


def _live_profile(tenant_id: str, tenant_profiles: dict[str, str]) -> str:
    """Live tenants are named <campaign>-<scenario>-<variant>-s<seed>-<profile tenant>."""
    matches = [profile for tenant, profile in tenant_profiles.items() if tenant_id.endswith(f"-{tenant}")]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot map live tenant {tenant_id!r} to one workload profile")
    return matches[0]


def _violation_rows(
    violations: list[dict[str, Any]],
    metadata: dict[str, Any],
    tenant_profiles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Count violation events per metric and bound direction.

    The bound is the one recorded with the event (details.bound). Events whose
    record carries no bound (CPU and memory caps) take it from the comparison
    between the measured value and the limit, and are marked as derived.
    """
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in violations:
        profile = (
            _live_profile(str(row.get("tenant_id", "")), tenant_profiles) if tenant_profiles is not None else ""
        )
        recorded = (row.get("details") or {}).get("bound")
        if recorded in ("maximum", "minimum"):
            bound, source = str(recorded), "recorded"
        else:
            measured, limit = float(row["measured_value"]), float(row["limit_value"])
            bound, source = ("maximum" if measured > limit else "minimum"), "derived"
        counts[(profile, str(row.get("metric")), bound, source)] += 1
    return [
        {
            **metadata,
            **({"profile": profile} if tenant_profiles is not None else {}),
            "metric": metric,
            "bound": bound,
            "bound_source": source,
            "events": events,
        }
        for (profile, metric, bound, source), events in sorted(counts.items())
    ]


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    return float(np.mean(clean)) if clean else math.nan


def _mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    clean = np.asarray([float(value) for value in values if pd.notna(value)], dtype=float)
    if clean.size == 0:
        return math.nan, math.nan, math.nan
    center = float(clean.mean())
    if clean.size == 1:
        return center, center, center
    margin = float(t.ppf(0.975, clean.size - 1) * clean.std(ddof=1) / math.sqrt(clean.size))
    return center, center - margin, center + margin



def _last_request_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("A controlled run has no request contract")
    return rows[-1]


def _position(value: Any, lower: Any, upper: Any, *, reverse: bool = False) -> float:
    value_f = float(value)
    lower_f = float(lower)
    upper_f = float(upper)
    if upper_f <= lower_f:
        return 0.5
    result = (value_f - lower_f) / (upper_f - lower_f)
    return float(1.0 - result if reverse else result)


def _load_controlled(
    indexes: tuple[Path, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    index_records = [(path, _json(path)) for path in indexes]
    commits = {str(index.get("git_commit")) for _, index in index_records}
    if len(commits) != 1 or "" in commits:
        raise RuntimeError(f"Controlled indexes do not share one Git commit: {commits}")

    all_records: list[dict[str, Any]] = []
    runtime_profiles: dict[str, set[str]] = {runtime: set() for runtime in RUNTIMES}
    for path, index in index_records:
        if index.get("schema_version") != "argos.campaign-index.v3":
            raise RuntimeError(f"Unsupported controlled campaign schema: {path}")
        if not index.get("completed_at"):
            raise RuntimeError(f"Controlled campaign is incomplete: {path}")
        runtime = str(index["runtime"])
        runtime_profiles[runtime].update(str(profile) for profile in index["profiles"])
        for record in index["runs"]:
            if record.get("role") == "evaluate":
                all_records.append({**record, "runtime": runtime, "index_path": str(path)})

    expected = {
        (runtime, profile, controller, seed)
        for runtime in RUNTIMES
        for profile in PROFILES
        for controller in CONTROLLERS
        for seed in (111, 222, 333, 444, 555)
    }
    observed = {
        (record["runtime"], record["profile"], record["controller"], int(record["seed"])) for record in all_records
    }
    if observed != expected or len(all_records) != 300:
        raise RuntimeError(
            f"Incomplete controlled evaluation design: missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}, runs={len(all_records)}"
        )
    if any(runtime_profiles[runtime] != set(PROFILES) for runtime in RUNTIMES):
        raise RuntimeError(f"Controlled indexes do not cover all profiles: {runtime_profiles}")

    run_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []

    for record in sorted(
        all_records,
        key=lambda row: (
            RUNTIMES.index(row["runtime"]),
            PROFILES.index(row["profile"]),
            CONTROLLERS.index(row["controller"]),
            int(row["seed"]),
        ),
    ):
        summary_rows = _json(_resolve(record["summary_path"]))
        if len(summary_rows) != 1:
            raise RuntimeError(f"Expected one summary for {record['session_id']}")
        summary = summary_rows[0]
        if summary.get("policy_mode") != "evaluate" or summary.get("policy_unchanged") is not True:
            raise RuntimeError(f"Non-frozen controlled run: {record['session_id']}")

        persistence = _resolve(summary["persistence_dir"])
        output_dir = _resolve(summary["output_dir"])
        decisions = _jsonl(persistence / "rl_decisions.jsonl")
        request_metrics = _jsonl(persistence / "request_metrics.jsonl")
        violations = _jsonl(persistence / "slo_violations.jsonl")
        contract = _last_request_contract(_jsonl(persistence / "requests.jsonl"))
        steps = int(summary["steps"])
        if len(decisions) != steps or steps != 64:
            raise RuntimeError(f"Unexpected decision count in {record['session_id']}: {len(decisions)}")
        reward_total = sum(float(row.get("reward", 0.0)) for row in decisions)
        if not math.isclose(reward_total, float(summary["total_reward"]), rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Reward mismatch in {record['session_id']}")

        node_metrics_path = output_dir / "node_metrics.csv"
        if not node_metrics_path.is_file():
            raise RuntimeError(f"Missing node metrics: {node_metrics_path}")
        nodes = pd.read_csv(node_metrics_path)
        fidelity = [
            float(row["mean_spatial_fidelity"])
            for row in request_metrics
            if row.get("mean_spatial_fidelity") is not None
        ]
        recall = [
            float(row["mean_hotspot_recall"]) for row in request_metrics if row.get("mean_hotspot_recall") is not None
        ]
        duty = [
            float(row["mean_service_duty_percent"])
            for row in request_metrics
            if row.get("mean_service_duty_percent") is not None
        ]
        violation_counts = Counter(str(row.get("metric")) for row in violations)
        metadata = {
            "runtime": record["runtime"],
            "profile": record["profile"],
            "controller": record["controller"],
            "seed": int(record["seed"]),
            "training_seed": int(record["training_seed"]),
            "session_id": record["session_id"],
        }
        run_rows.append(
            {
                **metadata,
                "steps": steps,
                "reward_total": reward_total,
                "reward_per_step": reward_total / steps,
                "spatial_fidelity_mean": _mean(fidelity),
                "hotspot_recall_mean": _mean(recall),
                "service_duty_mean_percent": _mean(duty),
                "violation_events": len(violations),
                "cpu_events": violation_counts["cpu_percent"],
                "memory_events": violation_counts["memory_percent"],
                "coverage_events": violation_counts["coverage"],
                "sample_events": violation_counts["sample_rate"],
                "freshness_events": violation_counts["freshness_age_seconds"],
                "cpu_p95": float(nodes["cpu_utilization"].quantile(0.95)),
                "cpu_max": float(nodes["cpu_utilization"].max()),
                "memory_p95": float(nodes["memory_utilization"].quantile(0.95)),
                "memory_max": float(nodes["memory_utilization"].max()),
                "bandwidth_p95_mbps": float(nodes["bandwidth_mbps"].quantile(0.95)),
            }
        )

        violation_rows.extend(_violation_rows(violations, metadata))
        counts = Counter(str(row["action"]) for row in decisions)
        for action in ACTIONS:
            action_rows.append({**metadata, "action": action, "count": counts[action], "share": counts[action] / steps})

        cumulative = 0.0
        for decision in sorted(decisions, key=lambda row: int(row["iteration"])):
            reward = float(decision.get("reward", 0.0))
            cumulative += reward
            state = decision.get("state") or {}
            decision_rows.append(
                {
                    **metadata,
                    "iteration": int(decision["iteration"]),
                    "action": str(decision["action"]),
                    "reward": reward,
                    "cumulative_reward": cumulative,
                    "coverage_quality_position": _position(
                        state["coverage"],
                        contract["coverage_min"],
                        contract["coverage_max"],
                    ),
                    "sample_quality_position": _position(
                        state["sample"],
                        contract["sample_min"],
                        contract["sample_max"],
                    ),
                    "freshness_quality_position": _position(
                        state["freshness"],
                        contract["freshness_min"],
                        contract["freshness_max"],
                        reverse=True,
                    ),
                    "cpu_percent": float(state.get("cpu_utilization", math.nan)),
                    "memory_percent": float(state.get("memory_utilization", math.nan)),
                    "input_multiplier": float(state.get("input_multiplier", math.nan)),
                }
            )

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(decision_rows),
        pd.DataFrame(action_rows),
        pd.DataFrame(violation_rows),
        [
            {
                "path": str(path.relative_to(EVIDENCE_ROOT)),
                "sha256": _sha256(path),
                "campaign_id": index["campaign_id"],
                "runtime": index["runtime"],
            }
            for path, index in index_records
        ],
    )


def _step_values(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    positions = np.searchsorted(times, grid, side="right") - 1
    positions = np.clip(positions, 0, len(times) - 1)
    return values[positions]


def _load_live(
    index_path: Path,
    analysis: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    index = _json(index_path)
    if index.get("schema_version") != "argos.live-campaign-index.v1":
        raise RuntimeError("Unsupported live campaign schema")
    if not index.get("completed_at") or index.get("excluded_runs"):
        raise RuntimeError("Live campaign is incomplete or has unresolved exclusions")
    expected = {
        (scenario, variant, seed)
        for scenario in SCENARIOS
        for variant in VARIANTS
        for seed in (1001, 1002, 1003, 1004, 1005)
    }
    observed = {(str(run["scenario"]), str(run["variant"]), int(run["live_seed"])) for run in index["runs"]}
    if observed != expected or len(index["runs"]) != len(expected):
        raise RuntimeError(f"Incomplete live design: missing={expected - observed}, extra={observed - expected}")

    tenant_profiles = _tenant_profiles(ROOT / "config.yaml")
    # Run-level live metrics come from argos.analysis.live.
    run_metrics = pd.read_csv(analysis)
    if len(run_metrics) != len(expected):
        raise RuntimeError(f"Expected {len(expected)} live analysis rows, found {len(run_metrics)}")
    metric_lookup = {str(row.run_id): row._asdict() for row in run_metrics.itertuples(index=False)}

    action_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    live_grid = np.arange(0.0, float(index["duration_minutes"]) + 0.001, 0.5)

    for run in sorted(index["runs"], key=lambda row: (row["scenario"], row["variant"], int(row["live_seed"]))):
        run_id = str(run["run_id"])
        scenario = str(run["scenario"])
        variant = str(run["variant"])
        live_seed = int(run["live_seed"])
        persistence = _resolve(run["persistence_dir"])
        summary_path = _resolve(run["summary_path"])
        if _sha256(summary_path) != str(run["summary_sha256"]):
            raise RuntimeError(f"Live summary hash mismatch: {run_id}")

        decisions = _jsonl(persistence / "rl_decisions.jsonl")
        violation_rows.extend(
            _violation_rows(
                _jsonl(persistence / "slo_violations.jsonl"),
                {"run_id": run_id, "scenario": scenario, "variant": variant, "live_seed": live_seed},
                tenant_profiles,
            )
        )
        counts = Counter(str(row["action"]) for row in decisions)
        total = len(decisions)
        for action in ACTIONS:
            action_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "variant": variant,
                    "live_seed": live_seed,
                    "action": action,
                    "count": counts[action],
                    "share": counts[action] / total if total else 0.0,
                }
            )

        events_path = summary_path.parent / "events.jsonl"
        samples = [
            row for row in _jsonl(events_path) if row.get("type") == "sample" and isinstance(row.get("snapshot"), dict)
        ]
        sample_times = np.asarray([float(row["elapsed_s"]) / 60.0 for row in samples], dtype=float)
        active = np.asarray(
            [float((row["snapshot"].get("cluster") or {}).get("active_requests", 0)) for row in samples],
            dtype=float,
        )
        queued = np.asarray(
            [float((row["snapshot"].get("cluster") or {}).get("queued_requests", 0)) for row in samples],
            dtype=float,
        )
        active_grid = _step_values(sample_times, active, live_grid)
        queued_grid = _step_values(sample_times, queued, live_grid)
        for minute, active_value, queued_value in zip(live_grid, active_grid, queued_grid):
            timeline_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "variant": variant,
                    "live_seed": live_seed,
                    "minute": minute,
                    "active_requests": active_value,
                    "queued_requests": queued_value,
                }
            )

        event_rows = _jsonl(events_path)
        if not event_rows:
            raise RuntimeError(f"Live run has no client events: {run_id}")
        start = min(pd.Timestamp(row["timestamp"]) for row in event_rows if row.get("timestamp"))
        decision_times = np.asarray(
            [(pd.Timestamp(row["timestamp"]) - start).total_seconds() / 60.0 for row in decisions],
            dtype=float,
        )
        rewards = np.asarray([float(row.get("reward", 0.0)) for row in decisions], dtype=float)
        order = np.argsort(decision_times)
        cumulative = np.cumsum(rewards[order])
        cumulative_grid = _step_values(decision_times[order], cumulative, live_grid)
        for minute, cumulative_value in zip(live_grid, cumulative_grid):
            reward_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "variant": variant,
                    "live_seed": live_seed,
                    "minute": minute,
                    "cumulative_reward": cumulative_value,
                }
            )

        if run_id not in metric_lookup:
            raise RuntimeError(f"Live analysis is missing {run_id}")

    manifest_row = {
        "path": str(index_path.relative_to(EVIDENCE_ROOT)),
        "sha256": _sha256(index_path),
        "campaign_id": index["campaign_id"],
        "runtime": index["runtime"],
    }
    return (
        run_metrics,
        pd.DataFrame(action_rows),
        pd.DataFrame(violation_rows),
        pd.DataFrame(timeline_rows),
        pd.DataFrame(reward_rows),
        [manifest_row],
    )



def _write_summary_tables(
    controlled_out: Path,
    live_out: Path,
    controlled_runs: pd.DataFrame,
    live_runs: pd.DataFrame,
) -> None:
    seed_rewards = (
        controlled_runs.groupby(["runtime", "controller", "seed"], as_index=False)
        .reward_per_step.mean()
        .rename(columns={"reward_per_step": "mean_reward"})
    )
    reward_summary_rows: list[dict[str, Any]] = []
    for (runtime, controller), group in seed_rewards.groupby(["runtime", "controller"]):
        center, lower, upper = _mean_ci(group.mean_reward)
        reward_summary_rows.append(
            {
                "runtime": runtime,
                "controller": controller,
                "seed_unit_count": len(group),
                "mean_reward": center,
                "ci95_low": lower,
                "ci95_high": upper,
            }
        )
    pd.DataFrame(reward_summary_rows).to_csv(
        controlled_out / "controlled_reward_summary.csv",
        index=False,
    )

    profile_summary = (
        controlled_runs.groupby(["runtime", "profile", "controller"])
        .reward_per_step.agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_reward",
                "std": "seed_std",
                "count": "seed_count",
            }
        )
    )
    profile_summary.to_csv(
        controlled_out / "controlled_profile_reward_summary.csv",
        index=False,
    )

    paired_rows: list[dict[str, Any]] = []
    paired_summary_rows: list[dict[str, Any]] = []
    for runtime in RUNTIMES:
        runtime_rewards = seed_rewards[seed_rewards.runtime == runtime]
        for baseline in ("static", "threshold", "best_fixed"):
            baseline_rewards = runtime_rewards[runtime_rewards.controller == baseline][["seed", "mean_reward"]].rename(
                columns={"mean_reward": "baseline_reward"}
            )
            for controller in CONTROLLERS:
                if controller == baseline:
                    continue
                controller_rewards = runtime_rewards[runtime_rewards.controller == controller][
                    ["seed", "mean_reward"]
                ].rename(columns={"mean_reward": "controller_reward"})
                paired = controller_rewards.merge(
                    baseline_rewards,
                    on="seed",
                    validate="one_to_one",
                )
                paired["reward_delta"] = paired["controller_reward"] - paired["baseline_reward"]
                for row in paired.itertuples(index=False):
                    paired_rows.append(
                        {
                            "runtime": runtime,
                            "controller": controller,
                            "baseline": baseline,
                            "seed": row.seed,
                            "controller_reward": row.controller_reward,
                            "baseline_reward": row.baseline_reward,
                            "reward_delta": row.reward_delta,
                        }
                    )
                center, lower, upper = _mean_ci(paired.reward_delta)
                paired_summary_rows.append(
                    {
                        "runtime": runtime,
                        "controller": controller,
                        "baseline": baseline,
                        "seed_unit_count": len(paired),
                        "mean_paired_delta": center,
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "wins": int((paired.reward_delta > 0).sum()),
                        "ties": int((paired.reward_delta == 0).sum()),
                        "losses": int((paired.reward_delta < 0).sum()),
                    }
                )
    pd.DataFrame(paired_rows).to_csv(
        controlled_out / "controlled_paired_reward_deltas.csv",
        index=False,
    )
    pd.DataFrame(paired_summary_rows).to_csv(
        controlled_out / "controlled_paired_reward_summary.csv",
        index=False,
    )

    live_metrics = [
        "reward_per_step",
        "evaluated_requests",
        "violated_epoch_ratio",
        "unique_violation_ratio",
        "spatial_fidelity_mean",
        "hotspot_recall_mean",
    ]
    live_summary_rows: list[dict[str, Any]] = []
    for (scenario, variant), group in live_runs.groupby(["scenario", "variant"]):
        row: dict[str, Any] = {
            "scenario": scenario,
            "variant": variant,
            "trial_count": len(group),
        }
        for metric in live_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        live_summary_rows.append(row)
    pd.DataFrame(live_summary_rows).to_csv(
        live_out / "live_summary.csv",
        index=False,
    )

    static_live = live_runs[live_runs.variant == "static"][["scenario", "live_seed", *live_metrics]].rename(
        columns={metric: f"{metric}_static" for metric in live_metrics}
    )
    paired_frames = []
    for learned in [v for v in VARIANTS if v != "static"]:
        learned_live = live_runs[live_runs.variant == learned][["scenario", "live_seed", *live_metrics]].rename(
            columns={metric: f"{metric}_learned" for metric in live_metrics}
        )
        merged = learned_live.merge(
            static_live,
            on=["scenario", "live_seed"],
            validate="one_to_one",
        )
        merged.insert(1, "variant", learned)
        for metric in live_metrics:
            merged[f"{metric}_delta"] = merged[f"{metric}_learned"] - merged[f"{metric}_static"]
        paired_frames.append(merged)
    live_paired = pd.concat(paired_frames, ignore_index=True)
    live_paired.to_csv(live_out / "live_paired_deltas.csv", index=False)



def generate(output: Path, controlled_indexes: tuple[Path, ...], live_index: Path, live_analysis: Path) -> None:
    controlled_out = output / "controlled" / "tables"
    live_out = output / "live" / "tables"
    controlled_out.mkdir(parents=True, exist_ok=True)
    live_out.mkdir(parents=True, exist_ok=True)

    controlled_runs, controlled_decisions, controlled_actions, controlled_violations, _ = (
        _load_controlled(controlled_indexes)
    )
    live_runs, live_actions, live_violations, live_timeline, live_rewards, _ = _load_live(
        live_index, live_analysis
    )

    controlled_runs.to_csv(controlled_out / "controlled_runs.csv", index=False)
    controlled_decisions.to_csv(controlled_out / "controlled_decisions.csv", index=False)
    controlled_actions.to_csv(controlled_out / "controlled_action_counts.csv", index=False)
    controlled_violations.to_csv(controlled_out / "controlled_violation_bounds.csv", index=False)
    live_runs.to_csv(live_out / "live_runs.csv", index=False)
    live_actions.to_csv(live_out / "live_action_counts.csv", index=False)
    live_violations.to_csv(live_out / "live_violation_bounds.csv", index=False)
    live_timeline.to_csv(live_out / "live_load_timeline.csv", index=False)
    live_rewards.to_csv(live_out / "live_reward_timeline.csv", index=False)
    _write_summary_tables(controlled_out, live_out, controlled_runs, live_runs)

    table_count = len(list(controlled_out.glob("*.csv"))) + len(list(live_out.glob("*.csv")))
    print(
        f"Wrote {table_count} tables from {len(controlled_runs)} controlled evaluations "
        f"and {len(live_runs)} live trials to {output}"
    )


def main(argv: list[str] | None = None) -> None:
    global EVIDENCE_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--controlled-index",
        type=Path,
        action="append",
        required=True,
        help="campaign_index.json of a controlled campaign; repeat for every campaign (both runtimes)",
    )
    parser.add_argument("--live-index", type=Path, required=True, help="campaign_index.json of the live campaign")
    parser.add_argument(
        "--live-analysis",
        type=Path,
        default=ROOT / "data" / "analysis" / "live" / "live_run_metrics.csv",
        help="Run-level live metrics written by argos.analysis.live",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT,
        help="Directory containing the data/ tree of the campaigns (default: repository root)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    EVIDENCE_ROOT = args.evidence_root.resolve()
    generate(
        args.output_dir.resolve(),
        tuple(path.resolve() for path in args.controlled_index),
        args.live_index.resolve(),
        args.live_analysis.resolve(),
    )


if __name__ == "__main__":
    main()
