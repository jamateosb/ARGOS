from argos.orchestrator.persistence import PersistenceConfig, PersistenceManager, RequestMetrics


def test_get_request_metrics_returns_most_recent_records_in_order(tmp_path):
    manager = PersistenceManager(PersistenceConfig(output_dir=tmp_path))
    manager.initialize()

    for iteration in range(1, 4):
        manager.save_request_metrics(
            RequestMetrics(
                request_id="req-1",
                timestamp=f"2026-04-21T00:00:0{iteration}+00:00",
                iteration=iteration,
                target_coverage=0.5,
                target_sample=0.5,
                target_freshness=30.0,
                actual_coverage=0.5,
                actual_sample=0.5,
                actual_freshness=30.0,
                assigned_node_count=1,
                active_node_count=1,
                responding_node_count=1,
                avg_latency_ms=float(iteration),
                max_latency_ms=float(iteration),
            )
        )

    latest = manager.get_request_metrics(request_id="req-1", limit=1)
    assert [item.iteration for item in latest] == [3]

    tail = manager.get_request_metrics(request_id="req-1", limit=2)
    assert [item.iteration for item in tail] == [2, 3]


def test_explicit_persistence_directory_is_honored_with_session_id(tmp_path):
    target = tmp_path / "explicit"
    manager = PersistenceManager(
        PersistenceConfig(output_dir=target, session_id="session-a")
    )
    manager.initialize()

    assert manager.config.output_dir == target
    assert (target / "rl_decisions.jsonl").is_file()
    assert (target / "slo_violations.jsonl").is_file()


def test_default_manager_honors_persistence_directory_environment(tmp_path, monkeypatch):
    import argos.orchestrator.persistence as persistence

    target = tmp_path / "live-trial"
    monkeypatch.setenv("ARGOS_PERSISTENCE_DIR", str(target))
    monkeypatch.setenv("ARGOS_SESSION_ID", "live-s11")
    monkeypatch.setattr(persistence, "_default_persistence", None)

    manager = persistence.get_persistence_manager()

    assert manager.config.output_dir == target
    assert manager.config.session_id == "live-s11"
    assert (target / "slo_decision_epochs.jsonl").is_file()
    monkeypatch.setattr(persistence, "_default_persistence", None)
