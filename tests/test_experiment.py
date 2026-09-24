import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argos import experiment  # noqa: E402


def _args(source: str) -> argparse.Namespace:
    return argparse.Namespace(
        rl_params_source=source,
        epsilon=None,
        learning_rate=None,
        discount_factor=None,
        epsilon_decay=None,
        epsilon_min=None,
        config_data={
            "rl": {
                "exploration_rate": 0.15,
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "exploration_decay": 0.995,
                "exploration_min": 0.01,
            }
        },
    )


def test_rl_params_source_default_ignores_best_params(monkeypatch, tmp_path):
    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()
    (tuning_dir / "best_params_qlearning.yaml").write_text("learning_rate: 0.9\n", encoding="utf-8")
    monkeypatch.setattr(experiment, "TUNING_DIR", tuning_dir)
    args = _args("default")

    experiment._apply_rl_defaults(args, True, "qlearning")

    assert args.learning_rate == 0.1


def test_rl_params_source_best_loads_best_params(monkeypatch, tmp_path):
    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()
    (tuning_dir / "best_params_qlearning.yaml").write_text("learning_rate: 0.9\n", encoding="utf-8")
    monkeypatch.setattr(experiment, "TUNING_DIR", tuning_dir)
    args = _args("best")

    experiment._apply_rl_defaults(args, True, "qlearning")

    assert args.learning_rate == 0.9


def test_neural_agents_use_algorithm_specific_adam_defaults():
    dqn_args = _args("default")
    ppo_args = _args("default")

    experiment._apply_rl_defaults(dqn_args, True, "dqn")
    experiment._apply_rl_defaults(ppo_args, True, "ppo")

    assert dqn_args.learning_rate == 0.001
    assert dqn_args.discount_factor == 0.99
    assert dqn_args.epsilon == 1.0
    assert ppo_args.learning_rate == 0.0003
    assert ppo_args.discount_factor == 0.99
    assert ppo_args.epsilon == 0.0


def test_benchmark_console_log_is_kept_in_session(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "SESSIONS_DIR", tmp_path / "data/sessions")
    args = argparse.Namespace(mode="benchmark", session_id="s1")

    paths = experiment._console_log_paths(args)

    assert paths == [tmp_path / "data/sessions/s1/benchmark/console.log"]


def test_parse_args_defaults_to_fresh_training_policy():
    args = experiment.parse_args(["--mode", "benchmark", "--iterations", "1"])

    assert args.policy_mode == "train"
    assert args.warm_start is False


def test_parse_quantiles_requires_three_bounded_values():
    assert experiment.parse_quantiles("1,0.5,0") == (1.0, 0.5, 0.0)

    for invalid in ("0.5,0.5", "0,0,1.2"):
        try:
            experiment.parse_quantiles(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid quantiles to fail: {invalid}")


def test_parse_input_schedule_is_explicit_and_bounded():
    assert experiment.parse_input_schedule("") == []
    assert experiment.parse_input_schedule("2,8,16,4") == [2, 8, 16, 4]

    for invalid in ("0,4", "101"):
        try:
            experiment.parse_input_schedule(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid schedule to fail: {invalid}")
