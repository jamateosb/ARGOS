# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for argos.orchestrator.logger module - ExperimentLogger."""

import csv
import shutil
import tempfile
from datetime import datetime, timezone

import pytest

from argos.orchestrator.logger import (
    DecisionLog,
    ExperimentLogger,
    ExperimentLoggerConfig,
    IterationLog,
    NodeMetricLog,
    RequestMetricLog,
    RLLearningLog,
)


class TestIterationLog:
    """Tests for IterationLog dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = IterationLog(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            iteration_id=1,
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=3,
            iteration_time_ms=150.5,
            avg_cpu_utilization=45.0,
            avg_memory_utilization=60.0,
            avg_response_time_ms=25.0,
            coverage_achieved=0.8,
            active_requests=2,
            rl_decisions_made=1,
            errors_count=0,
        )
        d = log.to_dict()

        assert d["iteration_id"] == 1
        assert d["total_nodes"] == 10
        assert d["avg_cpu_utilization"] == 45.0
        assert d["active_requests"] == 2
        assert d["rl_decisions_made"] == 1


class TestNodeMetricLog:
    """Tests for NodeMetricLog dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = NodeMetricLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=5,
            node_id="node-1",
            is_active=True,
            cpu_utilization=50.0,
            memory_utilization=70.0,
            bandwidth_mbps=100.0,
            service_type="heatmap",
            sample_rate=0.5,
            request_id="req-123",
            latency_ms=25.0,
            health_failures=0,
            heatmap_cells=100,
            hotspots_detected=5,
            processing_time_ms=50.0,
        )
        d = log.to_dict()

        assert d["node_id"] == "node-1"
        assert d["is_active"] is True
        assert d["request_id"] == "req-123"
        assert d["latency_ms"] == 25.0
        assert d["heatmap_cells"] == 100


class TestDecisionLog:
    """Tests for DecisionLog dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = DecisionLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=10,
            request_id="req-456",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            action_type="increase_coverage",
            action_index=1,
            prev_coverage=0.5,
            prev_sample=0.5,
            prev_freshness=60.0,
            new_coverage=0.55,
            new_sample=0.5,
            new_freshness=60.0,
            reward=0.85,
            reward_quality=0.9,
            reward_resource_penalty=0.05,
            reward_cost_penalty=0.0,
            reward_range_penalty_coverage=0.0,
            reward_range_penalty_sample=0.0,
            reward_range_penalty_freshness=0.0,
            epsilon=0.1,
            was_exploration=False,
            state_hash="abc123",
            reason="optimizing performance",
        )
        d = log.to_dict()

        assert d["algorithm"] == "qlearning"
        assert d["action_type"] == "increase_coverage"
        assert d["reward"] == 0.85
        assert d["prev_coverage"] == 0.5
        assert d["new_coverage"] == 0.55
        assert d["epsilon"] == 0.1


class TestRLLearningLog:
    """Tests for RLLearningLog dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = RLLearningLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=10,
            request_id="req-123",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            episodes=5,
            steps=100,
            total_reward=50.0,
            avg_reward=0.5,
            epsilon=0.1,
            states_visited=20,
            avg_q_value=0.8,
            max_q_value=1.2,
            min_q_value=0.3,
        )
        d = log.to_dict()

        assert d["algorithm"] == "qlearning"
        assert d["steps"] == 100
        assert d["avg_q_value"] == 0.8


class TestRequestMetricLog:
    """Tests for RequestMetricLog dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        log = RequestMetricLog(
            timestamp=datetime.now(timezone.utc),
            iteration_id=10,
            request_id="req-123",
            service_type="geo_heatmap",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            target_coverage=0.6,
            target_sample=0.5,
            target_freshness=60.0,
            actual_coverage=0.55,
            assigned_nodes=3,
            active_nodes=3,
            responding_nodes=2,
            avg_latency_ms=25.0,
            max_latency_ms=50.0,
            total_data_volume_bytes=1024000,
            total_processing_time_ms=150.0,
            total_hotspots=10,
        )
        d = log.to_dict()

        assert d["service_type"] == "geo_heatmap"
        assert d["algorithm"] == "qlearning"
        assert d["target_coverage"] == 0.6
        assert d["actual_coverage"] == 0.55


class TestExperimentLogger:
    """Tests for ExperimentLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        dirpath = tempfile.mkdtemp()
        yield dirpath
        shutil.rmtree(dirpath)

    def test_initialization(self, temp_dir):
        """Test logger initialization."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test_exp",
        )
        logger = ExperimentLogger(config)

        assert logger.iteration_count == 0
        assert logger.output_path is None  # Not initialized until first log

    def test_log_iteration_creates_output(self, temp_dir):
        """Test that logging creates output directory."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            flush_interval=1,
        )
        logger = ExperimentLogger(config)

        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=2,
            iteration_time_ms=100.0,
        )

        assert logger.output_path is not None
        assert logger.output_path.exists()
        assert (logger.output_path / "iterations.csv").exists()

    def test_log_iteration_increments_counter(self, temp_dir):
        """Test that iteration counter increments."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
        )
        logger = ExperimentLogger(config)

        for _i in range(5):
            logger.log_iteration(
                timestamp=datetime.now(timezone.utc),
                total_nodes=10,
                active_nodes=8,
                responding_nodes=7,
                configs_pushed=0,
                iteration_time_ms=100.0,
            )

        assert logger.iteration_count == 5

    def test_log_iteration_with_node_metrics(self, temp_dir):
        """Test logging with node metrics."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            log_node_metrics=True,
            flush_interval=1,
        )
        logger = ExperimentLogger(config)

        node_metrics = {
            "node-1": {
                "is_active": True,
                "service_type": "heatmap",
                "target_sample": 0.5,
                "request_id": "req-1",
                "state": {
                    "compute.cpu_utilization": 50.0,
                    "compute.memory_utilization": 60.0,
                    "network.bandwidth_mbps": 100.0,
                },
                "heatmap_service": {
                    "last_heatmap_cells": 100,
                    "last_hotspots": 5,
                    "last_processing_time_ms": 50.0,
                },
            }
        }

        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=1,
            active_nodes=1,
            responding_nodes=1,
            configs_pushed=0,
            iteration_time_ms=50.0,
            node_metrics=node_metrics,
        )

        assert (logger.output_path / "node_metrics.csv").exists()

    def test_log_decision(self, temp_dir):
        """Test decision logging."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            log_decisions=True,
            flush_interval=1,
        )
        logger = ExperimentLogger(config)

        # First log an iteration to initialize
        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=1,
            iteration_time_ms=100.0,
        )

        logger.log_decision(
            request_id="req-123",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            action_type="increase_sample",
            action_index=3,
            prev_coverage=0.5,
            prev_sample=0.5,
            prev_freshness=60.0,
            new_coverage=0.5,
            new_sample=0.55,
            new_freshness=60.0,
            reward=0.9,
            reward_components={
                "requirement_quality": 0.95,
                "resource_penalty": 0.05,
            },
            epsilon=0.1,
            was_exploration=False,
            state_hash="abc123",
            reason="optimizing performance",
        )
        logger.flush()

        assert (logger.output_path / "decisions.csv").exists()

    def test_log_rl_learning(self, temp_dir):
        """Test RL learning logging."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            log_rl_learning=True,
            flush_interval=1,
        )
        logger = ExperimentLogger(config)

        # First log an iteration to initialize
        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=1,
            iteration_time_ms=100.0,
        )

        logger.log_rl_learning(
            request_id="req-123",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            episodes=5,
            steps=100,
            total_reward=50.0,
            epsilon=0.1,
            states_visited=20,
            q_values={"hold": 0.5, "increase_coverage": 0.8, "decrease_coverage": 0.3},
        )
        logger.flush()

        assert (logger.output_path / "rl_learning.csv").exists()

    def test_log_request_metrics(self, temp_dir):
        """Test request metrics logging."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            log_request_metrics=True,
            flush_interval=1,
        )
        logger = ExperimentLogger(config)

        # First log an iteration to initialize
        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=1,
            iteration_time_ms=100.0,
        )

        logger.log_request_metrics(
            request_id="req-123",
            service_type="geo_heatmap",
            algorithm="qlearning",
            policy_version="qlearning-v1",
            target_coverage=0.6,
            target_sample=0.5,
            target_freshness=60.0,
            actual_coverage=0.55,
            assigned_nodes=3,
            active_nodes=3,
            responding_nodes=2,
            avg_latency_ms=25.0,
            max_latency_ms=50.0,
            total_data_volume_bytes=1024000,
            total_processing_time_ms=150.0,
            total_hotspots=10,
        )
        logger.flush()

        assert (logger.output_path / "request_metrics.csv").exists()

    def test_flush(self, temp_dir):
        """Test explicit flush writes data."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            flush_interval=100,  # High to prevent auto-flush
        )
        logger = ExperimentLogger(config)

        logger.log_iteration(
            timestamp=datetime.now(timezone.utc),
            total_nodes=10,
            active_nodes=8,
            responding_nodes=7,
            configs_pushed=0,
            iteration_time_ms=100.0,
        )

        # Before flush, file may be empty (just headers)
        logger.flush()

        # After flush, data should be written
        with open(logger.output_path / "iterations.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1

    def test_close(self, temp_dir):
        """Test close flushes remaining data."""
        config = ExperimentLoggerConfig(
            output_dir=temp_dir,
            experiment_name="test",
            flush_interval=100,
        )
        logger = ExperimentLogger(config)

        for _i in range(3):
            logger.log_iteration(
                timestamp=datetime.now(timezone.utc),
                total_nodes=10,
                active_nodes=8,
                responding_nodes=7,
                configs_pushed=0,
                iteration_time_ms=100.0,
            )

        logger.close()

        with open(logger.output_path / "iterations.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
