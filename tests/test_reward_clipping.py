"""Reward-clipping audit in the result tables."""

import pandas as pd
import pytest

from argos.analysis import tables

PENALTIES = dict.fromkeys(tables.REWARD_PENALTIES, 0.0)


def _decision(quality: float, reward: float, **penalties: float) -> dict:
    return {"reward": reward, "reward_components": {"requirement_quality": quality, **PENALTIES, **penalties}}


def test_raw_rewards_rebuild_the_unclipped_score():
    decisions = [
        _decision(0.6, 0.4, cost_penalty=0.2),
        _decision(0.1, -1.0, resource_penalty=1.2, resource_overload_penalty=0.5),
        _decision(1.0, 1.0),
    ]
    assert tables._raw_rewards(decisions, "run").tolist() == pytest.approx([0.4, -1.6, 1.0])


def test_raw_rewards_reject_components_that_do_not_reproduce_the_reward():
    with pytest.raises(RuntimeError, match="do not reproduce"):
        tables._raw_rewards([_decision(0.6, 0.5, cost_penalty=0.2)], "run")


def test_raw_rewards_require_every_component():
    decision = _decision(0.6, 0.6)
    del decision["reward_components"]["fairness_penalty"]
    with pytest.raises(KeyError):
        tables._raw_rewards([decision], "run")


def test_clipping_row_counts_each_bound():
    decisions = [
        _decision(0.2, -1.0, resource_penalty=1.2, capacity_penalty=0.3),
        _decision(0.5, 0.5),
        _decision(0.3, 0.1, latency_penalty=0.2),
    ]
    row = tables._clipping_row({"run_id": "r"}, decisions, "r")
    assert row["decisions"] == 3
    assert (row["lower_clipped"], row["upper_clipped"]) == (1, 0)
    assert row["raw_min"] == pytest.approx(-1.3)
    assert row["reward_per_step"] == pytest.approx((-1.0 + 0.5 + 0.1) / 3)
    assert row["raw_reward_per_step"] == pytest.approx((-1.3 + 0.5 + 0.1) / 3)


def _pairs(raw_offsets: list[float]) -> pd.DataFrame:
    rewards = [0.10, 0.12, 0.08, 0.11, 0.09]
    return pd.DataFrame(
        {
            "runtime": "thread",
            "controller": "dqn",
            "baseline": "static",
            "reward_delta": rewards,
            "raw_reward_delta": [value + offset for value, offset in zip(rewards, raw_offsets)],
        }
    )


def test_sensitivity_is_unchanged_when_the_clip_never_binds():
    (row,) = tables._paired_sensitivity(_pairs([0.0] * 5), ["runtime", "controller", "baseline"])
    assert row["raw_reward_mean_delta"] == pytest.approx(row["reward_mean_delta"])
    assert row["reward_wins"] == row["raw_reward_wins"] == 5
    assert row["conclusion_changed"] is False


def test_sensitivity_flags_a_conclusion_that_depends_on_clipping():
    (row,) = tables._paired_sensitivity(_pairs([-0.3, -0.3, -0.3, 0.0, 0.0]), ["runtime", "controller", "baseline"])
    assert row["raw_reward_wins"] == 2
    assert row["conclusion_changed"] is True
