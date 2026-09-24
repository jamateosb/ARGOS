"""Tests for observed heatmap quality metrics."""

import pytest

from argos.services.geo_heatmap import GeoHeatmapService, HeatmapCell


def test_identical_heatmaps_have_perfect_fidelity():
    cells = [
        HeatmapCell(latitude=1.0, longitude=2.0, frequency=10),
        HeatmapCell(latitude=1.1, longitude=2.1, frequency=5),
    ]

    fidelity, hotspot_recall = GeoHeatmapService._heatmap_fidelity(cells, cells, sample_rate=1.0)

    assert fidelity == pytest.approx(1.0)
    assert hotspot_recall == pytest.approx(1.0)


def test_missing_hotspot_reduces_fidelity_and_recall():
    reference = [
        HeatmapCell(latitude=1.0, longitude=2.0, frequency=10),
        HeatmapCell(latitude=1.1, longitude=2.1, frequency=5),
    ]
    sampled = [HeatmapCell(latitude=1.0, longitude=2.0, frequency=5)]

    fidelity, hotspot_recall = GeoHeatmapService._heatmap_fidelity(
        sampled,
        reference,
        sample_rate=0.5,
    )

    assert 0.0 < fidelity < 1.0
    assert hotspot_recall == pytest.approx(0.5)


def test_empty_reference_has_well_defined_quality():
    assert GeoHeatmapService._heatmap_fidelity([], [], sample_rate=0.5) == (1.0, 1.0)
