"""Tests for the live traffic trial driver configuration contract."""


import yaml


from argos import live_trial  # noqa: E402


def test_live_trial_loads_stochastic_defaults_from_config(tmp_path):
    config = {
        "cluster": {"orchestrator": {"host": "127.0.0.1", "port": 8001}},
        "live_trial": {
            "duration_minutes": 12,
            "traffic_model": "uniform",
            "random_seed": 123,
            "job_interval_min_minutes": 2,
            "job_interval_max_minutes": 6,
            "request_duration_mode": "uniform",
            "request_duration_min_minutes": 5,
            "request_duration_max_minutes": 10,
            "traffic_scenario": "realistic",
            "runtime_mode": "thread",
            "profiles": [{"name": "p1", "payload": {"service_type": "geo_heatmap"}}],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    args = live_trial.parse_args(["--config", str(config_path)])

    assert args.traffic_model == "uniform"
    assert args.random_seed == 123
    assert args.job_interval_min_minutes == 2
    assert args.job_interval_max_minutes == 6
    assert args.request_duration_mode == "uniform"
    assert args.request_duration_min_minutes == 5
    assert args.request_duration_max_minutes == 10
    assert args.traffic_scenario == "realistic"
    assert args.runtime_mode == "thread"
    assert len(live_trial.traffic_profiles(args.config_data)) == 1


def test_live_submission_payload_carries_profile_identity():
    profile = {
        "name": "cost-sensitive",
        "payload": {
            "service_type": "geo_heatmap",
            "tenant_id": "planning",
        },
    }

    payload = live_trial.submission_payload(profile, "trial-11")

    assert payload["profile_name"] == "cost-sensitive"
    assert payload["tenant_id"] == "trial-11-planning"
