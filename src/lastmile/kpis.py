"""Fleet KPIs shown on the dashboard.

Emission factors below are ASSUMPTIONS for a portfolio project. Replace them with the
current UK Government GHG conversion factors (DESNZ) and cite the year in your README.
"""

from __future__ import annotations

import numpy as np

DIESEL_VAN_KG_CO2_PER_KM = 0.25  # assumption: average diesel van
EV_VAN_KWH_PER_KM = 0.25  # assumption: e.g. a mid-size electric van


def route_km(route: list[int], dist_km: np.ndarray, return_to_depot: bool = True) -> float:
    if len(route) < 2:
        return 0.0
    legs = list(zip(route[:-1], route[1:], strict=False))
    if return_to_depot:
        legs.append((route[-1], route[0]))
    return float(sum(dist_km[a, b] for a, b in legs))


def fleet_kpis(
    routes: list[list[int]],
    arrivals_s: list[list[int]],
    dist_km: np.ndarray,
    dropped: list[int],
    grid_g_per_kwh: float = 150.0,
) -> dict[str, float]:
    stops = sum(max(len(r) - 1, 0) for r in routes)
    km = sum(route_km(r, dist_km) for r in routes)
    hours = sum((a[-1] - a[0]) / 3600 for a in arrivals_s if len(a) > 1) or 1e-9
    diesel_kg = km * DIESEL_VAN_KG_CO2_PER_KM
    ev_kg = km * EV_VAN_KWH_PER_KM * grid_g_per_kwh / 1000
    return {
        "vans_used": sum(len(r) > 1 for r in routes),
        "stops": stops,
        "stops_per_hour": stops / hours,
        "km_driven": km,
        "missed_windows": len(dropped),
        "kg_co2_ev": ev_kg,
        "kg_co2_saved_vs_diesel": diesel_kg - ev_kg,
    }
