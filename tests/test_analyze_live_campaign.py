from __future__ import annotations


import pytest

from argos.analysis import live as MODULE


def test_validate_campaign_requires_complete_paired_design() -> None:
    scenarios = ["concurrency", "realistic"]
    seeds = [1001, 1002, 1003, 1004, 1005]
    index = {
        "schema_version": "argos.live-campaign-index.v1",
        "completed_at": "2026-07-25T23:28:52+00:00",
        "training_seeds": [11, 22, 33, 44, 55],
        "live_seeds": seeds,
        "scenarios": scenarios,
        "excluded_runs": [],
        "runs": [
            {
                "scenario": scenario,
                "live_seed": seed,
                "variant": variant,
            }
            for scenario in scenarios
            for seed in seeds
            for variant in ("static", "ppo")
        ],
    }

    MODULE._validate_campaign(index)
    index["runs"].pop()
    with pytest.raises(RuntimeError, match="Incomplete live design"):
        MODULE._validate_campaign(index)


def test_paired_summary_uses_seed_level_deltas() -> None:
    pairs = []
    for seed, static, ppo in [
        (1001, 0.1, 0.2),
        (1002, 0.2, 0.3),
        (1003, 0.3, 0.4),
        (1004, 0.4, 0.5),
        (1005, 0.5, 0.6),
    ]:
        row = {"scenario": "realistic", "variant": "ppo", "live_seed": seed}
        for metric in MODULE.PAIR_METRICS:
            row[f"static_{metric}"] = static
            row[f"learned_{metric}"] = ppo
            row[f"delta_{metric}"] = ppo - static
        pairs.append(row)

    summary = MODULE._summary_rows(pairs, scenarios=["realistic"])
    reward = next(row for row in summary if row["metric"] == "reward_per_step")

    assert reward["n"] == 5
    assert reward["mean_delta"] == pytest.approx(0.1)
    assert reward["wins"] == 5
    assert reward["losses"] == 0


def test_policy_validation_separates_queued_requests() -> None:
    summary = {
        "frozen_policy_checks": [
            {
                "required": True,
                "loaded": False,
                "unchanged": True,
                "queued_only": True,
                "steps": 0,
            },
            {
                "required": True,
                "loaded": True,
                "unchanged": True,
                "queued_only": False,
                "steps": 4,
            },
        ],
        "frozen_policy_failures": [],
    }

    evaluated, tail = MODULE._validate_policy_checks(
        summary,
        run_id="test-run",
        variant="ppo",
    )

    assert evaluated == 1
    assert tail == 0
