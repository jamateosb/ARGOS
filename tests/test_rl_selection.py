"""Tests for benchmark-driven RL algorithm selection."""

from argos.orchestrator.rl.selection import resolve_algorithm, select_best_policy_from_benchmark


def test_auto_uses_best_concrete_runtime_row(tmp_path, monkeypatch):
    ranking = tmp_path / "model_ranking_ci.csv"
    ranking.write_text(
        "\n".join(
            [
                "runtime,mode,algorithm,n,reward_mean",
                "all,tuned,dqn,10,99.0",
                "process,tuned,ppo,5,25.9",
                "thread,tuned,qlearning,5,20.4",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(ranking))

    selection = select_best_policy_from_benchmark()

    assert selection.algorithm == "ppo"
    assert selection.runtime == "process"
    assert selection.mode == "tuned"
    assert selection.reward_mean == 25.9


def test_resolve_algorithm_honors_explicit_backend(tmp_path, monkeypatch):
    ranking = tmp_path / "model_ranking_ci.csv"
    ranking.write_text("runtime,mode,algorithm,n,reward_mean\nprocess,tuned,ppo,5,25.9\n", encoding="utf-8")
    monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(ranking))

    assert resolve_algorithm("qlearning") == "qlearning"
    assert resolve_algorithm("dqn") == "dqn"


def test_resolve_algorithm_accepts_auto_alias(tmp_path, monkeypatch):
    ranking = tmp_path / "model_ranking_ci.csv"
    ranking.write_text("runtime,mode,algorithm,n,reward_mean\nthread,tuned,dqn,5,30.0\n", encoding="utf-8")
    monkeypatch.setenv("ARGOS_BENCHMARK_RANKING_CSV", str(ranking))

    assert resolve_algorithm("benchmark") == "dqn"
