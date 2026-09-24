# =============================================================================
# ARGOS
# Copyright (c) Javier Mateos-Bravo
# =============================================================================
"""
Geographic heatmap service using simulated Seville mobility trajectories.

This service processes location trajectories generated with the ONE simulator
to produce urban-mobility heatmaps. The data lives in ``data/seville_bus``.

Key features:
- Geographic coordinate processing (latitude/longitude)
- Configurable spatial resolution (blur/precision)
- Circular geographic filtering by radius
- Sample rate affects how much data is processed
- Freshness controls update frequency
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Optional

from argos.common.constants import SEVILLE_DATA_DIR
from argos.domain.numeric_contract import canonical_contract_value

# =============================================================================
# Configuration
# =============================================================================

# Dataset path
DEFAULT_DATASET_PATH = SEVILLE_DATA_DIR

# Default query parameters for Seville center
# Note: begin_date/end_date are ignored, we use all available data
DEFAULT_QUERY = {
    "latitude": 37.378833,
    "longitude": -5.976987,
    "radius": 3500,  # meters
}

# Total simulated users in the Seville mobility dataset
TOTAL_USERS = 60


def get_node_user_assignments(node_id: int, total_nodes: int, total_users: int = TOTAL_USERS) -> list[int]:
    """
    Dynamically calculate which users are assigned to a given node.

    Users are evenly distributed across nodes. If users don't divide evenly,
    earlier nodes get one extra user each.

    Args:
        node_id: Node ID (1-indexed, e.g., 1, 2, 3, ...).
        total_nodes: Total number of nodes in the cluster.
        total_users: Total number of users in the dataset.

    Returns:
        List of user IDs assigned to this node.

    Examples:
        >>> get_node_user_assignments(1, 3)  # 3 nodes: [0-19]
        [0, 1, 2, ..., 19]
        >>> get_node_user_assignments(1, 2)  # 2 nodes: [0-29]
        [0, 1, 2, ..., 29]
        >>> get_node_user_assignments(2, 4)  # 4 nodes, node 2: [15-29]
        [15, 16, ..., 29]
    """
    if node_id < 1 or node_id > total_nodes:
        return []

    # Calculate base users per node and remainder
    users_per_node = total_users // total_nodes
    remainder = total_users % total_nodes

    # Earlier nodes get one extra user if there's a remainder
    # Node 1 gets extra if remainder >= 1, Node 2 if remainder >= 2, etc.
    start_user = 0
    for n in range(1, node_id):
        extra = 1 if n <= remainder else 0
        start_user += users_per_node + extra

    # This node's count
    extra = 1 if node_id <= remainder else 0
    end_user = start_user + users_per_node + extra

    return list(range(start_user, end_user))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GeoPoint:
    """A geographic point with latitude and longitude."""

    latitude: float
    longitude: float
    timestamp: Optional[str] = None
    frequency: int = 1


@dataclass
class HeatmapCell:
    """A cell in the heatmap grid."""

    latitude: float
    longitude: float
    frequency: int


@dataclass
class GeoHeatmapResult:
    """Result of a geographic heatmap computation."""

    request_id: str
    timestamp: float
    node_id: int
    query_params: dict[str, Any]
    total_points_processed: int
    filtered_points: int
    heatmap_cells: int
    hotspots_detected: int  # Cells with frequency > threshold
    processing_time_ms: float
    data_volume_bytes: int
    sample_rate_applied: float
    checksum: str
    spatial_fidelity: Optional[float] = None
    hotspot_recall: Optional[float] = None
    heatmap_data: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for API transmission."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "query_params": self.query_params,
            "total_points_processed": self.total_points_processed,
            "filtered_points": self.filtered_points,
            "heatmap_cells": self.heatmap_cells,
            "hotspots_detected": self.hotspots_detected,
            "processing_time_ms": self.processing_time_ms,
            "data_volume_bytes": self.data_volume_bytes,
            "sample_rate_applied": self.sample_rate_applied,
            "checksum": self.checksum,
            "spatial_fidelity": self.spatial_fidelity,
            "hotspot_recall": self.hotspot_recall,
            # Optionally include full heatmap data (can be large)
            "heatmap_data": self.heatmap_data if len(self.heatmap_data) <= 1000 else [],
        }


# =============================================================================
# Geographic Utilities
# =============================================================================


class GeoUtils:
    """Geographic calculation utilities."""

    EARTH_RADIUS_M = 6371000  # Earth radius in meters

    @staticmethod
    def blur_coordinate(value: float, resolution: int = 4) -> float:
        """
        Reduce precision of a coordinate for spatial aggregation.

        Args:
            value: Latitude or longitude value.
            resolution: Number of decimal places to keep.

        Returns:
            Blurred coordinate value.
        """
        round_func = math.floor if value >= 0 else math.ceil
        return round_func(value * 10**resolution) / 10**resolution

    @staticmethod
    def calculate_derived_position(
        origin: tuple[float, float], distance_m: float, bearing_deg: float
    ) -> tuple[float, float]:
        """
        Calculate a point at a given distance and bearing from origin.

        Args:
            origin: (latitude, longitude) of starting point.
            distance_m: Distance in meters.
            bearing_deg: Bearing in degrees (0=N, 90=E, 180=S, 270=W).

        Returns:
            (latitude, longitude) of destination point.
        """
        lat_a = math.radians(origin[0])
        lon_a = math.radians(origin[1])
        angular_distance = distance_m / GeoUtils.EARTH_RADIUS_M
        bearing = math.radians(bearing_deg)

        lat = math.asin(
            math.sin(lat_a) * math.cos(angular_distance)
            + math.cos(lat_a) * math.sin(angular_distance) * math.cos(bearing)
        )

        dlon = math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat_a),
            math.cos(angular_distance) - math.sin(lat_a) * math.sin(lat),
        )

        lon = ((lon_a + dlon + math.pi) % (math.pi * 2)) - math.pi

        return (math.degrees(lat), math.degrees(lon))

    @staticmethod
    def get_bounding_box(center: tuple[float, float], radius_m: float) -> dict[str, float]:
        """
        Calculate bounding box for circular region.

        Args:
            center: (latitude, longitude) of center point.
            radius_m: Radius in meters.

        Returns:
            Dictionary with north, south, east, west bounds.
        """
        north = GeoUtils.calculate_derived_position(center, radius_m, 0)
        east = GeoUtils.calculate_derived_position(center, radius_m, 90)
        south = GeoUtils.calculate_derived_position(center, radius_m, 180)
        west = GeoUtils.calculate_derived_position(center, radius_m, 270)

        return {
            "north": north[0],
            "south": south[0],
            "east": east[1],
            "west": west[1],
        }


# =============================================================================
# Data Loader
# =============================================================================


class DatasetLoader:
    """Loads and caches trajectory data from JSON files."""

    def __init__(self, dataset_path: Path = DEFAULT_DATASET_PATH):
        self.dataset_path = Path(dataset_path)
        self._cache: dict[int, list[dict]] = {}
        self._cache_lock = Lock()

    def load_user_data(self, user_id: int) -> list[dict]:
        """
        Load trajectory data for a specific user.

        Args:
            user_id: User ID (0-59).

        Returns:
            List of location records.
        """
        with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]

        filename = f"SB_User{user_id}.json"
        filepath = self.dataset_path / filename

        if not filepath.exists():
            return []

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            with self._cache_lock:
                self._cache[user_id] = data

            return data
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error loading {filename}: {e}")
            return []

    def load_node_data(self, node_id: int, total_nodes: int = 3) -> list[dict]:
        """
        Load all trajectory data assigned to a node.

        Users are dynamically distributed across nodes based on total_nodes.

        Args:
            node_id: Node ID (1-indexed).
            total_nodes: Total number of nodes in the cluster.

        Returns:
            Combined list of all user trajectories for this node.
        """
        user_ids = get_node_user_assignments(node_id, total_nodes)
        combined_data = []

        for user_id in user_ids:
            user_data = self.load_user_data(user_id)
            combined_data.extend(user_data)

        return combined_data

# =============================================================================
# Heatmap Builder
# =============================================================================


class HeatmapBuilder:
    """Builds geographic heatmaps from trajectory data."""

    def __init__(self, resolution: int = 4):
        """
        Initialize heatmap builder.

        Args:
            resolution: Decimal places for coordinate precision (default 4 = ~11m).
        """
        self.resolution = resolution

    def convert_locations(self, raw_data: list[dict]) -> list[GeoPoint]:
        """
        Convert raw JSON data to GeoPoint objects.

        Args:
            raw_data: List of records with 'lat', 'lng', and optional 'timestamp'.

        Returns:
            List of GeoPoint objects.
        """
        points = []
        for record in raw_data:
            if "lat" in record and "lng" in record:
                points.append(
                    GeoPoint(
                        latitude=record["lat"],
                        longitude=record["lng"],
                        timestamp=record.get("timestamp"),
                    )
                )
        return points

    def filter_by_region(
        self, points: list[GeoPoint], center_lat: float, center_lon: float, radius_m: float
    ) -> list[GeoPoint]:
        """
        Filter points to those within a circular region.

        Args:
            points: List of GeoPoint objects.
            center_lat: Center latitude.
            center_lon: Center longitude.
            radius_m: Radius in meters.

        Returns:
            Filtered list of points within the region.
        """
        bounds = GeoUtils.get_bounding_box((center_lat, center_lon), radius_m)

        filtered = []
        for point in points:
            if bounds["south"] < point.latitude < bounds["north"] and bounds["west"] < point.longitude < bounds["east"]:
                filtered.append(point)

        return filtered

    def apply_sample_rate(self, points: list[GeoPoint], sample_rate: float) -> list[GeoPoint]:
        """
        Apply sample rate to reduce data volume.

        Args:
            points: List of GeoPoint objects.
            sample_rate: Fraction of data to keep (0.0 to 1.0).

        Returns:
            Sampled list of points.
        """
        if sample_rate >= 1.0:
            return points

        if sample_rate <= 0.0:
            return []

        # Deterministic sampling based on point index
        sample_count = max(1, int(len(points) * sample_rate))
        step = len(points) / sample_count

        sampled = []
        for i in range(sample_count):
            idx = int(i * step)
            if idx < len(points):
                sampled.append(points[idx])

        return sampled

    def build_heatmap(self, points: list[GeoPoint], hotspot_threshold: int = 5) -> tuple[list[HeatmapCell], int]:
        """
        Build heatmap from points by aggregating into cells.

        Args:
            points: List of GeoPoint objects.
            hotspot_threshold: Minimum frequency to count as hotspot.

        Returns:
            Tuple of (list of HeatmapCell, number of hotspots).
        """
        # Aggregate by blurred coordinates
        heatmap_matrix: dict[float, dict[float, int]] = defaultdict(lambda: defaultdict(int))

        for point in points:
            lat = GeoUtils.blur_coordinate(point.latitude, self.resolution)
            lon = GeoUtils.blur_coordinate(point.longitude, self.resolution)
            heatmap_matrix[lat][lon] += point.frequency

        # Convert to cell list
        cells = []
        hotspots = 0

        for lat, lon_map in heatmap_matrix.items():
            for lon, freq in lon_map.items():
                cells.append(
                    HeatmapCell(
                        latitude=lat,
                        longitude=lon,
                        frequency=freq,
                    )
                )
                if freq >= hotspot_threshold:
                    hotspots += 1

        return cells, hotspots


# =============================================================================
# Geographic Heatmap Service
# =============================================================================


@dataclass
class GeoHeatmapService:
    """
    Geographic heatmap service using simulated Seville mobility trajectories.

    This service processes trajectory data to generate mobility density
    heatmaps, with configurable sample rate and query parameters.

    Attributes:
        node_id: ID of the node running this service (1-indexed).
        total_nodes: Total number of nodes in the cluster.
        sample_rate: Fraction of data to process (0.0-1.0).
        resolution: Decimal places for coordinate precision.
        query_params: Geographic query parameters (lat, lon, radius).
        dataset_path: Path to the dataset directory.
    """

    node_id: int = 1
    total_nodes: int = 3
    sample_rate: float = 0.5
    input_multiplier: int = 1
    resolution: int = 4
    query_params: dict[str, Any] = field(default_factory=lambda: DEFAULT_QUERY.copy())
    dataset_path: Path = DEFAULT_DATASET_PATH
    is_running: bool = False
    _last_result: Optional[GeoHeatmapResult] = None
    _stop_event: Event = field(default_factory=Event)
    _worker_thread: Optional[Thread] = None
    _loader: Optional[DatasetLoader] = None
    _builder: Optional[HeatmapBuilder] = None
    _reference_cache: dict[tuple, list[HeatmapCell]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize components."""
        if not isinstance(self._stop_event, Event):
            self._stop_event = Event()
        self._loader = DatasetLoader(self.dataset_path)
        self._builder = HeatmapBuilder(self.resolution)

    def configure(
        self,
        sample_rate: Optional[float] = None,
        resolution: Optional[int] = None,
        query_params: Optional[dict[str, Any]] = None,
        node_id: Optional[int] = None,
        total_nodes: Optional[int] = None,
        input_multiplier: Optional[int] = None,
    ) -> None:
        """
        Update service configuration.

        Args:
            sample_rate: New sample rate (0.0 to 1.0).
            resolution: New coordinate resolution.
            query_params: New query parameters.
            node_id: New node ID for data assignment.
            total_nodes: Total number of nodes in the cluster.
            input_multiplier: Controlled multiplier for trajectory input volume.
        """
        if sample_rate is not None:
            self.sample_rate = canonical_contract_value(sample_rate, (0.0, 1.0))

        if resolution is not None:
            self.resolution = resolution
            self._builder = HeatmapBuilder(self.resolution)

        if query_params is not None:
            self.query_params.update(query_params)

        if node_id is not None:
            self.node_id = node_id

        if total_nodes is not None:
            self.total_nodes = max(1, total_nodes)

        if input_multiplier is not None:
            self.input_multiplier = max(1, int(input_multiplier))

    def compute_heatmap(self, request_id: str) -> GeoHeatmapResult:
        """
        Compute a heatmap from the node's assigned data.

        This is the main computation method that:
        1. Loads trajectory data for this node
        2. Converts to geographic points
        3. Filters by geographic region
        4. Applies sample rate
        5. Builds the heatmap

        Args:
            request_id: Unique identifier for this computation.

        Returns:
            GeoHeatmapResult with computation details and data.
        """
        start_time = time.time()

        # 1. Load raw data for this node (users dynamically assigned based on total_nodes)
        raw_data = self._loader.load_node_data(self.node_id, self.total_nodes)
        if self.input_multiplier > 1:
            raw_data = raw_data * self.input_multiplier
        total_points = len(raw_data)

        # 2. Convert to GeoPoints
        points = self._builder.convert_locations(raw_data)

        # 3. Filter by geographic region
        filtered_points = self._builder.filter_by_region(
            points,
            self.query_params.get("latitude", DEFAULT_QUERY["latitude"]),
            self.query_params.get("longitude", DEFAULT_QUERY["longitude"]),
            self.query_params.get("radius", DEFAULT_QUERY["radius"]),
        )

        reference_cells = self._reference_cells(filtered_points)

        # 4. Apply sample rate
        sampled_points = self._builder.apply_sample_rate(filtered_points, self.sample_rate)

        # 5. Build heatmap
        cells, hotspots = self._builder.build_heatmap(sampled_points)
        spatial_fidelity, hotspot_recall = self._heatmap_fidelity(
            cells,
            reference_cells,
            self.sample_rate,
        )

        # Estimate data volume without expensive serialization
        # Each cell dict is ~40 bytes (lat: float, lon: float, freq: int)
        estimated_bytes_per_cell = 40
        data_volume = len(cells) * estimated_bytes_per_cell

        # Generate checksum
        checksum_input = f"{request_id}_{len(cells)}_{hotspots}_{self.sample_rate}"
        checksum = hashlib.md5(checksum_input.encode()).hexdigest()[:8]

        processing_time = (time.time() - start_time) * 1000

        # Create result
        result = GeoHeatmapResult(
            request_id=request_id,
            timestamp=time.time(),
            node_id=self.node_id,
            query_params=self.query_params.copy(),
            total_points_processed=total_points,
            filtered_points=len(filtered_points),
            heatmap_cells=len(cells),
            hotspots_detected=hotspots,
            processing_time_ms=processing_time,
            data_volume_bytes=data_volume,
            sample_rate_applied=self.sample_rate,
            checksum=checksum,
            spatial_fidelity=spatial_fidelity,
            hotspot_recall=hotspot_recall,
            heatmap_data=[{"lat": c.latitude, "lon": c.longitude, "freq": c.frequency} for c in cells],
        )

        self._last_result = result
        return result

    def _reference_cells(self, filtered_points: list[GeoPoint]) -> list[HeatmapCell]:
        """Return the cached full-sample heatmap for the current data partition."""
        query_key = tuple(sorted((str(key), str(value)) for key, value in self.query_params.items()))
        cache_key = (self.node_id, self.total_nodes, self.resolution, self.input_multiplier, query_key)
        reference = self._reference_cache.get(cache_key)
        if reference is None:
            reference, _ = self._builder.build_heatmap(filtered_points)
            self._reference_cache[cache_key] = reference
        return reference

    @staticmethod
    def _heatmap_fidelity(
        sampled_cells: list[HeatmapCell],
        reference_cells: list[HeatmapCell],
        sample_rate: float,
        hotspot_threshold: int = 5,
    ) -> tuple[float, float]:
        """Compare a sampled heatmap with its full-sample spatial reference."""
        sampled = {(cell.latitude, cell.longitude): float(cell.frequency) for cell in sampled_cells}
        reference = {(cell.latitude, cell.longitude): float(cell.frequency) for cell in reference_cells}
        sampled_total = sum(sampled.values())
        reference_total = sum(reference.values())

        if reference_total <= 0:
            return (1.0 if sampled_total <= 0 else 0.0), 1.0
        if sampled_total <= 0:
            return 0.0, 0.0

        keys = set(reference) | set(sampled)
        total_variation = 0.5 * sum(
            abs(sampled.get(key, 0.0) / sampled_total - reference.get(key, 0.0) / reference_total)
            for key in keys
        )
        spatial_fidelity = max(0.0, min(1.0, 1.0 - total_variation))

        reference_hotspots = {key for key, frequency in reference.items() if frequency >= hotspot_threshold}
        if not reference_hotspots:
            hotspot_recall = 1.0
        else:
            scaling = max(float(sample_rate), 1e-9)
            detected = {
                key for key, frequency in sampled.items() if frequency / scaling >= hotspot_threshold
            }
            hotspot_recall = len(reference_hotspots & detected) / len(reference_hotspots)

        return spatial_fidelity, hotspot_recall

    @property
    def last_result(self) -> Optional[GeoHeatmapResult]:
        """Get the most recent computation result."""
        return self._last_result

    def get_metrics(self) -> dict:
        """Get service metrics for monitoring."""
        user_ids = get_node_user_assignments(self.node_id, self.total_nodes)
        return {
            "service_type": "geo_heatmap",
            "node_id": self.node_id,
            "total_nodes": self.total_nodes,
            "assigned_users": len(user_ids),
            "user_range": f"{min(user_ids)}-{max(user_ids)}" if user_ids else "none",
            "is_running": self.is_running,
            "sample_rate": self.sample_rate,
            "input_multiplier": self.input_multiplier,
            "resolution": self.resolution,
            "query_params": self.query_params,
            "last_processing_ms": self._last_result.processing_time_ms if self._last_result else 0.0,
            "last_data_bytes": self._last_result.data_volume_bytes if self._last_result else 0,
            "last_cells": self._last_result.heatmap_cells if self._last_result else 0,
            "last_hotspots": self._last_result.hotspots_detected if self._last_result else 0,
            "last_spatial_fidelity": self._last_result.spatial_fidelity if self._last_result else None,
            "last_hotspot_recall": self._last_result.hotspot_recall if self._last_result else None,
        }
