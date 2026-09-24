"""Tests for controlled campaign statistics."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_controlled_campaign as campaign  # noqa: E402
from argos.analysis.controlled import (  # noqa: E402
    _bootstrap_mean_ci,
    _holm,
    _rank_biserial,
    _t_mean_ci,
    _validate_index_compatibility,
)
from scripts.run_controlled_campaign import (  # noqa: E402
    _amend_seed_schedule,
    _import_fixed_selections,
    _validate_coverage_reachability,
    _validated_summary,
)


def test_bootstrap_interval_is_reproducible_and_contains_mean():
    values = [0.1, 0.2, 0.3, 0.4]

    first = _bootstrap_mean_ci(values, resamples=1000, seed=11)
    second = _bootstrap_mean_ci(values, resamples=1000, seed=11)

    assert first == second
    assert first[0] <= 0.25 <= first[1]


def test_rank_biserial_respects_direction():
    assert _rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert _rank_biserial([-1.0, -2.0, -3.0]) == pytest.approx(-1.0)
    assert _rank_biserial([0.0, 0.0]) == 0.0


def test_t_interval_is_centered_on_paired_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    low, high = _t_mean_ci(values)

    assert (low + high) / 2 == pytest.approx(0.3)
    assert low < 0.3 < high


def test_holm_adjustment_is_monotonic_and_bounded():
    rows = [{"p_value": 0.01}, {"p_value": 0.03}, {"p_value": 0.2}]

    _holm(rows)

    adjusted = [row["p_holm"] for row in rows]
    assert adjusted == pytest.approx([0.03, 0.06, 0.2])
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_validated_summary_rejects_partial_run(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"git": {"commit": "a" * 40, "dirty": False}}),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps([{"iterations": 17, "steps": 17, "manifest_path": str(manifest)}]),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Incomplete run"):
        _validated_summary(summary, session_id="partial", expected_iterations=64)


def test_validated_summary_accepts_exact_run(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"git": {"commit": "a" * 40, "dirty": False}}),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    payload = {"iterations": 64, "steps": 64, "manifest_path": str(manifest)}
    summary.write_text(json.dumps([payload]), encoding="utf-8")

    assert _validated_summary(summary, session_id="complete", expected_iterations=64) == payload


def test_quarantine_moves_benchmark_persistence_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    summary = tmp_path / "data/sessions/run/benchmark/benchmark_summary.json"
    persistence = tmp_path / "data/sessions/run_static_s11/persistence"
    output = tmp_path / "data/runs/run_static_seed11"
    summary.parent.mkdir(parents=True)
    persistence.mkdir(parents=True)
    output.mkdir(parents=True)
    (persistence / "rl_decisions.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    summary.write_text(
        json.dumps([{"persistence_dir": str(persistence), "output_dir": str(output)}]),
        encoding="utf-8",
    )

    destination = campaign._quarantine_run_artifacts(summary, session_id="run")

    assert destination is not None
    assert (destination / "benchmark/benchmark/benchmark_summary.json").is_file()
    assert (destination / "persistence_dir/rl_decisions.jsonl").is_file()
    assert (destination / "output_dir/run_manifest.json").is_file()
    assert not persistence.exists()
    assert not output.exists()


def test_import_fixed_selections_requires_completed_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    index = tmp_path / "data/campaigns/source/campaign_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({"fixed_selection": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source campaign is incomplete"):
        _import_fixed_selections([index], forbidden_seeds={11})


def test_import_fixed_selections_preserves_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    index = tmp_path / "data/campaigns/source/campaign_index.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "campaign_id": "source-campaign",
                "completed_at": "2026-07-11T00:00:00+00:00",
                "tuning_seeds": [101, 202, 303],
                "runs": [
                    {
                        "role": "fixed_tune",
                        "profile": "aggressive-incident",
                        "seed": 101,
                    }
                ],
                "fixed_selection": {
                    "aggressive-incident": {
                        "candidate": "grid_c0.5_s0.5_f0.5",
                        "quantiles": "0.5,0.5,0.5",
                        "tuning_mean_reward": 12.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    imported = _import_fixed_selections([index], forbidden_seeds={11, 22})

    selection = imported["aggressive-incident"]
    assert selection["quantiles"] == "0.5,0.5,0.5"
    assert selection["source_campaign_id"] == "source-campaign"
    assert selection["source_campaign_index"] == str(index.resolve())
    assert selection["source_fixed_tune_run_count"] == 1


def test_three_node_profile_reachability_rejects_impossible_contract():
    valid = [
        campaign.Profile(
            "cost-sensitive",
            {"coverage_min": 0.33, "coverage_max": 0.67},
        )
    ]
    _validate_coverage_reachability(valid, 3)

    invalid = [
        campaign.Profile(
            "impossible",
            {"coverage_min": 0.35, "coverage_max": 0.65},
        )
    ]
    with pytest.raises(ValueError, match="only provide"):
        _validate_coverage_reachability(invalid, 3)


def test_fixed_reference_uses_vertices_and_midpoint_only():
    candidates = set(campaign.FIXED_CANDIDATES.values())

    assert len(candidates) == 9
    assert "0.5,0.5,0.5" in candidates
    assert {
        f"{coverage:g},{sample:g},{freshness:g}"
        for coverage in (0.0, 1.0)
        for sample in (0.0, 1.0)
        for freshness in (0.0, 1.0)
    }.issubset(candidates)


def test_validated_summary_rejects_wrong_runtime(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"git": {"commit": "a" * 40, "dirty": False}}),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            [
                {
                    "iterations": 64,
                    "steps": 64,
                    "runtime_mode": "process",
                    "manifest_path": str(manifest),
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime mismatch"):
        _validated_summary(
            summary,
            session_id="wrong-runtime",
            expected_iterations=64,
            expected_runtime="thread",
        )


def test_amend_seed_schedule_excludes_surplus_runs_before_evaluation():
    index = {
        "seeds": [11, 22, 33],
        "runs": [
            {"session_id": "train-11", "seed": 11, "role": "train"},
            {"session_id": "train-33", "seed": 33, "role": "train"},
        ],
    }

    assert _amend_seed_schedule(index, [11, 22]) is True
    assert index["seeds"] == [11, 22]
    assert [row["session_id"] for row in index["runs"]] == ["train-11"]
    assert [row["session_id"] for row in index["excluded_runs"]] == ["train-33"]
    assert index["seed_schedule_amendments"][0]["reason"] == "seed_schedule_reduced_before_evaluation"


def test_amend_seed_schedule_rejects_new_seed():
    index = {"seeds": [11], "runs": []}

    with pytest.raises(RuntimeError, match="only retain existing"):
        _amend_seed_schedule(index, [11, 22])


def _compatible_index(**overrides):
    payload = {
        "completed_at": "2026-07-20T00:00:00+00:00",
        "seeds": [11, 22, 33, 44, 55],
        "evaluation_seeds": [111, 222, 333, 444, 555],
        "tuning_seeds": [101, 202, 303],
        "learners": ["qlearning", "dqn", "ppo"],
        "input_schedule": "2,16,64,4",
        "evaluation_input_schedule": "3,12,48,6",
        "schedule_segment_iterations": 16,
        "fixed_candidate_design": "eight contract vertices plus midpoint",
        "fixed_candidate_count": 9,
        "train_iterations": 512,
        "tune_iterations": 64,
        "eval_iterations": 64,
        "git_commit": "a" * 40,
    }
    payload.update(overrides)
    return payload


def test_index_compatibility_accepts_matching_runtime_partitions(tmp_path):
    indexes = [
        (tmp_path / "thread.json", _compatible_index()),
        (tmp_path / "process.json", _compatible_index()),
    ]

    _validate_index_compatibility(indexes)


def test_index_compatibility_rejects_mixed_commits(tmp_path):
    indexes = [
        (tmp_path / "thread.json", _compatible_index()),
        (tmp_path / "process.json", _compatible_index(git_commit="b" * 40)),
    ]

    with pytest.raises(RuntimeError, match="different git_commit"):
        _validate_index_compatibility(indexes)
