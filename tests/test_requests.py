# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""Tests for argos.domain.requests module - AnalyticsRequest and EffectiveConfiguration."""

import pytest

from argos.domain.requests import (
    SUPPORTED_RL_ALGORITHMS,
    AnalyticsRequest,
    EffectiveConfiguration,
    NegotiationResult,
)


class TestAnalyticsRequest:
    """Tests for AnalyticsRequest dataclass."""

    def test_default_values(self):
        """Test default request values."""
        request = AnalyticsRequest()
        assert request.service_type == "heatmap"
        assert request.coverage_range == (0.5, 0.8)
        assert request.sample_range == (0.3, 0.7)
        assert request.freshness_range == (30.0, 120.0)
        assert request.algorithm == "qlearning"
        assert request.request_id is not None
        assert len(request.request_id) == 8

    def test_custom_values(self):
        """Test custom request values."""
        request = AnalyticsRequest(
            service_type="anomaly_detection",
            coverage_range=(0.2, 0.4),
            sample_range=(0.1, 0.5),
            freshness_range=(10.0, 60.0),
        )
        assert request.service_type == "anomaly_detection"
        assert request.coverage_range == (0.2, 0.4)

    def test_validate_valid_request(self):
        """Test validation of valid request."""
        request = AnalyticsRequest()
        assert request.validate() is True

    def test_validate_invalid_algorithm(self):
        """Validation should reject unknown RL algorithm identifiers."""
        request = AnalyticsRequest(algorithm="bogus")
        assert request.validate() is False

    def test_validate_invalid_coverage_range(self):
        """Test validation fails for invalid coverage range."""
        request = AnalyticsRequest(coverage_range=(0.8, 0.5))  # min > max
        assert request.validate() is False

    def test_validate_invalid_sample_range(self):
        """Test validation fails for invalid sample range."""
        request = AnalyticsRequest(sample_range=(1.5, 2.0))  # out of bounds
        assert request.validate() is False

    def test_validate_invalid_freshness_range(self):
        """Test validation fails for invalid freshness range."""
        request = AnalyticsRequest(freshness_range=(-1.0, 30.0))  # negative
        assert request.validate() is False

    def test_midpoint_config(self):
        """Test midpoint configuration generation."""
        request = AnalyticsRequest(
            coverage_range=(0.4, 0.8),
            sample_range=(0.2, 0.6),
            freshness_range=(30.0, 90.0),
            algorithm="ppo",
        )
        config = request.midpoint_config()

        assert config.request_id == request.request_id
        assert config.service_type == request.service_type
        assert config.algorithm == "ppo"
        assert config.target_coverage == pytest.approx(0.6)  # (0.4 + 0.8) / 2
        assert config.target_sample == pytest.approx(0.4)  # (0.2 + 0.6) / 2
        assert config.target_freshness == pytest.approx(60.0)  # (30 + 90) / 2

    def test_supported_algorithms_constant(self):
        """Supported algorithm identifiers should stay stable and explicit."""
        assert SUPPORTED_RL_ALGORITHMS == ("qlearning", "dqn", "ppo", "static", "threshold")


class TestEffectiveConfiguration:
    """Tests for EffectiveConfiguration dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = EffectiveConfiguration(
            request_id="test-123",
            service_type="heatmap",
            target_coverage=0.6,
            target_sample=0.5,
            target_freshness=60.0,
            assigned_node_count=3,
        )
        d = config.to_dict()

        assert d["request_id"] == "test-123"
        assert d["service_type"] == "heatmap"
        assert d["target_coverage"] == 0.6
        assert d["assigned_node_count"] == 3

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "request_id": "test-456",
            "service_type": "anomaly",
            "target_coverage": 0.7,
            "target_sample": 0.4,
            "target_freshness": 45.0,
            "assigned_node_count": 5,
        }
        config = EffectiveConfiguration.from_dict(data)

        assert config.request_id == "test-456"
        assert config.target_coverage == 0.7
        assert config.assigned_node_count == 5

    def test_from_dict_missing_assigned_count(self):
        """Test deserialization with missing optional field."""
        data = {
            "request_id": "test-789",
            "service_type": "heatmap",
            "target_coverage": 0.5,
            "target_sample": 0.5,
            "target_freshness": 60.0,
        }
        config = EffectiveConfiguration.from_dict(data)
        assert config.assigned_node_count == 1  # default


class TestNegotiationResult:
    """Tests for NegotiationResult dataclass."""

    def test_coverage_fulfilled_within_range(self):
        """Test coverage fulfilled when within range."""
        request = AnalyticsRequest(coverage_range=(0.4, 0.6))
        config = EffectiveConfiguration(
            request_id=request.request_id,
            service_type="heatmap",
            target_coverage=0.5,
            target_sample=0.5,
            target_freshness=60.0,
        )
        result = NegotiationResult(
            original_request=request,
            effective_config=config,
        )
        assert result.coverage_fulfilled() == pytest.approx(1.0)

    def test_coverage_fulfilled_below_range(self):
        """Test coverage fulfilled when below range."""
        request = AnalyticsRequest(coverage_range=(0.4, 0.6))
        config = EffectiveConfiguration(
            request_id=request.request_id,
            service_type="heatmap",
            target_coverage=0.2,  # Below minimum
            target_sample=0.5,
            target_freshness=60.0,
        )
        result = NegotiationResult(
            original_request=request,
            effective_config=config,
        )
        assert result.coverage_fulfilled() == pytest.approx(0.5)  # 0.2 / 0.4

    def test_accepted_default(self):
        """Test default accepted status."""
        request = AnalyticsRequest()
        config = request.midpoint_config()
        result = NegotiationResult(
            original_request=request,
            effective_config=config,
        )
        assert result.accepted is True
        assert result.rejection_reason is None
