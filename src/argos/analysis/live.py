"""Analyze an indexed live campaign pairing the static midpoint with frozen learned policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_nan(values: list[float]) -> float:
    return mean(values) if values else math.nan


def _validate_campaign(index: dict[str, Any]) -> None:
    if index.get("schema_version") != "argos.live-campaign-index.v1":
        raise RuntimeError("Unsupported live campaign schema")
    if not index.get("completed_at"):
        raise RuntimeError("Live campaign is incomplete")
    training_seeds = [int(value) for value in index.get("training_seeds", [])]
    live_seeds = [int(value) for value in index.get("live_seeds", [])]
    scenarios = [str(value) for value in index.get("scenarios", [])]
    if len(training_seeds) != 5 or len(set(training_seeds)) != 5:
        raise RuntimeError("Canonical live analysis requires five training seeds")
    if len(live_seeds) != 5 or len(set(live_seeds)) != 5:
        raise RuntimeError("Canonical live analysis requires five live seeds")
    if not scenarios:
        raise RuntimeError("Live campaign has no scenarios")
    if index.get("excluded_runs"):
        raise RuntimeError("Live campaign contains unresolved exclusions")

    variants = sorted({str(run.get("variant")) for run in index.get("runs", [])})
    if "static" not in variants or len(variants) < 2:
        raise RuntimeError(f"Live campaign needs 'static' plus learned variants: {variants}")
    expected = {
        (scenario, live_seed, variant)
        for scenario in scenarios
        for live_seed in live_seeds
        for variant in variants
    }
    observed = {
        (str(run.get("scenario")), int(run.get("live_seed", -1)), str(run.get("variant")))
        for run in index.get("runs", [])
    }
    if observed != expected or len(index.get("runs", [])) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Incomplete live design: missing={missing}, extra={extra}")


def _validate_policy_checks(
    summary: dict[str, Any],
    *,
    run_id: str,
    variant: str,
) -> tuple[int, int]:
    all_checks = list(summary.get("frozen_policy_checks") or [])
    tail_checks = list(summary.get("tail_censored_policy_checks") or []) or [
        check for check in all_checks if check.get("tail_censored")
    ]
    tail_ids = {str(check.get("request_id")) for check in tail_checks}
    checks = [check for check in all_checks if str(check.get("request_id")) not in tail_ids]
    if variant == "static":
        return 0, len(tail_checks)
    failures = list(summary.get("frozen_policy_failures") or [])
    if failures:
        raise RuntimeError(f"Frozen-policy failures remain in {run_id}: {failures[:1]}")
    evaluated = 0
    for check in checks:
        steps = int(check.get("steps", 0) or 0)
        if not check.get("required"):
            raise RuntimeError(f"Learned-variant request did not require a frozen policy in {run_id}")
        if check.get("queued_only"):
            if steps != 0 or check.get("loaded"):
                raise RuntimeError(f"Invalid queued policy state in {run_id}: {check}")
            continue
        if not check.get("loaded") or not check.get("unchanged"):
            raise RuntimeError(f"Invalid frozen policy state in {run_id}: {check}")
        if steps > 0:
            evaluated += 1
    for check in tail_checks:
        if not check.get("required") or not check.get("loaded") or not check.get("unchanged"):
            raise RuntimeError(f"Invalid tail-censored policy state in {run_id}: {check}")
        if int(check.get("steps", 0) or 0) != 0:
            raise RuntimeError(f"Tail-censored request has decisions in {run_id}: {check}")
    if evaluated < 1:
        raise RuntimeError(f"No frozen learned-policy request was evaluated in {run_id}")
    return evaluated, len(tail_checks)


def _run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    run_id = str(run["run_id"])
    variant = str(run["variant"])
    summary_path = _evidence_path(run["summary_path"])
    if not summary_path.is_file():
        raise RuntimeError(f"Missing live summary: {summary_path}")
    expected_hash = str(run.get("summary_sha256") or "")
    if not expected_hash or _sha256(summary_path) != expected_hash:
        raise RuntimeError(f"Live summary hash mismatch: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    persistence = _evidence_path(run["persistence_dir"])
    decisions = _jsonl(persistence / "rl_decisions.jsonl")
    epochs = _jsonl(persistence / "slo_decision_epochs.jsonl")
    violations = _jsonl(persistence / "slo_violations.jsonl")
    request_metrics = _jsonl(persistence / "request_metrics.jsonl")
    errors = _jsonl(persistence / "errors.jsonl", required=False)

    decision_keys = {(str(row.get("request_id")), int(row.get("iteration", -1))) for row in decisions}
    epoch_keys = {(str(row.get("request_id")), int(row.get("iteration", -1))) for row in epochs}
    if len(decision_keys) != len(decisions):
        raise RuntimeError(f"Duplicate live decisions in {run_id}")
    if len(epoch_keys) != len(epochs) or epoch_keys != decision_keys:
        raise RuntimeError(f"Live SLO epochs do not match decisions in {run_id}")

    policy_evaluated, tail_censored = _validate_policy_checks(
        summary,
        run_id=run_id,
        variant=variant,
    )
    submissions = list(summary.get("submitted_jobs") or [])
    if not submissions:
        raise RuntimeError(f"Live trial submitted no requests: {run_id}")
    initial_status = Counter(str(row.get("response", {}).get("status", "unknown")) for row in submissions)

    rewards_by_request: dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        rewards_by_request[str(decision["request_id"])].append(float(decision.get("reward", 0.0)))
    violation_counts = Counter(str(row.get("metric")) for row in violations)
    violating_requests = {str(row.get("request_id")) for row in violations}
    evaluated_requests = set(rewards_by_request)
    fidelity = [
        float(row["mean_spatial_fidelity"]) for row in request_metrics if row.get("mean_spatial_fidelity") is not None
    ]
    recall = [
        float(row["mean_hotspot_recall"]) for row in request_metrics if row.get("mean_hotspot_recall") is not None
    ]
    sample_observations = sum(row.get("actual_sample") is not None for row in request_metrics)
    freshness_observations = sum(row.get("observed_freshness_age_max_s") is not None for row in request_metrics)
    rewards = [float(row.get("reward", 0.0)) for row in decisions]
    violated_epochs = sum(bool(row.get("violated")) for row in epochs)

    return {
        "run_id": run_id,
        "scenario": str(run["scenario"]),
        "variant": variant,
        "training_seed": int(run["training_seed"]),
        "live_seed": int(run["live_seed"]),
        "submitted": len(submissions),
        "initially_accepted": initial_status["accepted"],
        "initially_queued": initial_status["queued"],
        "evaluated_requests": len(evaluated_requests),
        "tail_censored_requests": tail_censored,
        "decision_steps": len(decisions),
        "reward_total": sum(rewards),
        "reward_per_step": _mean_or_nan(rewards),
        "mean_request_reward_per_step": _mean_or_nan([mean(values) for values in rewards_by_request.values()]),
        "slo_epochs": len(epochs),
        "violated_epochs": violated_epochs,
        "violated_epoch_ratio": violated_epochs / len(epochs),
        "unique_violating_requests": len(violating_requests & evaluated_requests),
        "unique_violation_ratio": (len(violating_requests & evaluated_requests) / len(evaluated_requests)),
        "violation_events": len(violations),
        "coverage_events": violation_counts["coverage"],
        "sample_events": violation_counts["sample_rate"],
        "freshness_events": violation_counts["freshness_age_seconds"],
        "cpu_events": violation_counts["cpu_percent"],
        "memory_events": violation_counts["memory_percent"],
        "spatial_fidelity_mean": _mean_or_nan(fidelity),
        "hotspot_recall_mean": _mean_or_nan(recall),
        "sample_observation_ratio": sample_observations / len(request_metrics),
        "freshness_observation_ratio": freshness_observations / len(request_metrics),
        "frozen_policy_evaluated_requests": policy_evaluated,
        "error_records": len(errors),
    }


PAIR_METRICS = (
    "reward_per_step",
    "mean_request_reward_per_step",
    "violated_epoch_ratio",
    "unique_violation_ratio",
    "spatial_fidelity_mean",
    "hotspot_recall_mean",
)


def _paired_rows(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
    live_seeds: list[int],
) -> list[dict[str, Any]]:
    lookup = {(row["scenario"], row["live_seed"], row["variant"]): row for row in rows}
    learned_variants = sorted({row["variant"] for row in rows} - {"static"})
    pairs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for live_seed in live_seeds:
            static = lookup[(scenario, live_seed, "static")]
            for variant in learned_variants:
                learned = lookup[(scenario, live_seed, variant)]
                pair: dict[str, Any] = {
                    "scenario": scenario,
                    "variant": variant,
                    "live_seed": live_seed,
                    "training_seed": learned["training_seed"],
                }
                for metric in PAIR_METRICS:
                    pair[f"static_{metric}"] = static[metric]
                    pair[f"learned_{metric}"] = learned[metric]
                    pair[f"delta_{metric}"] = learned[metric] - static[metric]
                pair["static_evaluated_requests"] = static["evaluated_requests"]
                pair["learned_evaluated_requests"] = learned["evaluated_requests"]
                pair["static_violation_events"] = static["violation_events"]
                pair["learned_violation_events"] = learned["violation_events"]
                pairs.append(pair)
    return pairs


def _summary_rows(
    pairs: list[dict[str, Any]],
    *,
    scenarios: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learned_variants = sorted({row["variant"] for row in pairs})
    for scenario in scenarios:
      for variant in learned_variants:
        group = [row for row in pairs if row["scenario"] == scenario and row["variant"] == variant]
        for metric in PAIR_METRICS:
            deltas = [float(row[f"delta_{metric}"]) for row in group]
            center = mean(deltas)
            spread = stdev(deltas)
            critical = float(t.ppf(0.975, df=len(deltas) - 1))
            margin = critical * spread / math.sqrt(len(deltas))
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "metric": metric,
                    "n": len(deltas),
                    "static_mean": mean(float(row[f"static_{metric}"]) for row in group),
                    "learned_mean": mean(float(row[f"learned_{metric}"]) for row in group),
                    "mean_delta": center,
                    "sd_delta": spread,
                    "ci95_low": center - margin,
                    "ci95_high": center + margin,
                    "wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "losses": sum(value < 0 for value in deltas),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Directory containing the data/ tree of the campaign; recorded paths are re-rooted there",
    )
    args = parser.parse_args(argv)
    global EVIDENCE_ROOT
    EVIDENCE_ROOT = args.evidence_root.resolve() if args.evidence_root else None

    index = json.loads(args.campaign_index.read_text(encoding="utf-8"))
    _validate_campaign(index)
    run_rows = [_run_metrics(run) for run in index["runs"]]
    scenarios = [str(value) for value in index["scenarios"]]
    live_seeds = [int(value) for value in index["live_seeds"]]
    paired_rows = _paired_rows(
        run_rows,
        scenarios=scenarios,
        live_seeds=live_seeds,
    )
    summary_rows = _summary_rows(paired_rows, scenarios=scenarios)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "live_run_metrics.csv", run_rows)
    _write_csv(args.output_dir / "live_paired_seed_deltas.csv", paired_rows)
    _write_csv(args.output_dir / "live_summary.csv", summary_rows)
    manifest = {
        "schema_version": "argos.live-analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": index["campaign_id"],
        "campaign_index": str(args.campaign_index.resolve()),
        "campaign_index_sha256": _sha256(args.campaign_index),
        "run_count": len(run_rows),
        "paired_seed_count_per_scenario": len(live_seeds),
        "experimental_unit": "paired live seed",
        "primary_endpoint": "mean decision reward per step",
        "inference": "Descriptive paired-seed analysis with N=5 per scenario.",
        "notes": [
            "Coverage violations include three-node deployment granularity.",
            "One request was right-censored after admission within the final poll interval.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for row in summary_rows:
        if row["metric"] == "reward_per_step":
            print(
                f"{row['scenario']}: Static={row['static_mean']:.3f}, "
                f"{row['variant']}={row['learned_mean']:.3f}, delta={row['mean_delta']:+.3f}, "
                f"95% CI=[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}], "
                f"wins={row['wins']}/{row['n']}"
            )


if __name__ == "__main__":
    main()
