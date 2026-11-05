"""Put Module 3 together: orders -> matrix -> VRPTW -> KPIs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lastmile.geo import distance_matrix_km, time_matrix_s
from lastmile.kpis import fleet_kpis
from lastmile.routing.vrptw import VrpResult, solve_vrptw
from lastmile.sim.orders import Order, random_orders

SHIFT_LENGTH_S = 14 * 3600  # 07:00 to 21:00


@dataclass
class DayResult:
    orders: list[Order]
    points: list[tuple[float, float]]  # index 0 = depot
    dist_km: np.ndarray
    vrp: VrpResult
    kpis: dict[str, float]


def solve_day(
    orders: list[Order],
    depot: tuple[float, float],
    n_vans: int = 3,
    van_capacity_parcels: int = 120,
    service_s: int = 120,
    time_s: np.ndarray | None = None,
    dist_km: np.ndarray | None = None,
    time_limit_s: int = 10,
) -> DayResult:
    points = [depot] + [(o.lat, o.lon) for o in orders]
    if dist_km is None:
        dist_km = distance_matrix_km(points)
    if time_s is None:
        time_s = time_matrix_s(dist_km)
    windows = [(0, SHIFT_LENGTH_S)] + [o.window_s for o in orders]
    demands = [0] + [o.parcels for o in orders]
    vrp = solve_vrptw(
        time_s,
        windows,
        demands,
        van_capacity_parcels,
        n_vans,
        service_s=[0] + [service_s] * len(orders),
        horizon_s=SHIFT_LENGTH_S + 3600,
        time_limit_s=time_limit_s,
    )
    return DayResult(
        orders, points, dist_km, vrp, fleet_kpis(vrp.routes, vrp.arrivals_s, dist_km, vrp.dropped)
    )


def demo_day(
    n_orders: int = 60,
    n_vans: int = 3,
    seed: int = 0,
    depot: tuple[float, float] = (51.8750, -0.4150),
) -> DayResult:
    """Quick scenario without OSMnx (straight-line x 1.3 at 25 km/h)."""
    return solve_day(random_orders(n_orders, depot, seed=seed), depot, n_vans=n_vans)
