"""Geometry helpers that need no road network (used by the API and tests).

For real road travel times use lastmile.sim.network (OSMnx) instead.
"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_matrix_km(points: list[tuple[float, float]], detour_factor: float = 1.3) -> np.ndarray:
    """Straight-line distance x detour factor (roads are never straight)."""
    n = len(points)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i, j] = haversine_km(*points[i], *points[j]) * detour_factor
    return m


def time_matrix_s(dist_km: np.ndarray, speed_kmh: float = 25.0) -> np.ndarray:
    """Convert km to seconds at an average urban speed."""
    return np.rint(dist_km / speed_kmh * 3600).astype(int)
