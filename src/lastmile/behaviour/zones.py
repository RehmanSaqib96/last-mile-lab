"""Level 1 of driver behaviour: drivers finish a zone before moving on.

Zone ids look like "P-12.3C". We add a penalty to the travel time whenever a move
leaves the current zone, then tune the penalty on training routes so the optimiser's
sequences look more like the drivers'.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable

import numpy as np

from lastmile.amazon.loader import RouteArrays
from lastmile.routing.metrics import kendall_tau
from lastmile.routing.tsp import solve_tsp

_ZONE = re.compile(r"^([A-Z])-(\d+)\.(\d+)([A-Z])$")


def parse_zone(z: str | None) -> tuple[str, int, int, str] | None:
    if not z:
        return None
    m = _ZONE.match(str(z).strip())
    if not m:
        return None
    a, b, c, d = m.groups()
    return a, int(b), int(c), d


def super_zone(z: str | None) -> str | None:
    """'P-12.3C' -> 'P-12' (a bigger area made of several zones)."""
    p = parse_zone(z)
    return f"{p[0]}-{p[1]}" if p else None


def zone_cost(
    travel_s: np.ndarray, zones: list[str | None], zone_penalty: float, super_penalty: float
) -> np.ndarray:
    """travel time + penalty for changing zone + extra penalty for changing super-zone."""
    z = np.array([x or "" for x in zones], dtype=object)
    sz = np.array([super_zone(x) or "" for x in zones], dtype=object)
    diff_z = (z[:, None] != z[None, :]).astype(float)
    diff_sz = (sz[:, None] != sz[None, :]).astype(float)
    # the station (index 0) has no zone: don't penalise leaving it
    diff_z[0, :] = diff_sz[0, :] = 0
    return travel_s + zone_penalty * diff_z + super_penalty * diff_sz


def tune_penalties(
    routes: Iterable[RouteArrays],
    zone_grid: tuple[float, ...] = (0, 60, 120, 300, 600),
    super_grid: tuple[float, ...] = (0, 300, 900),
    time_limit_s: int = 2,
) -> tuple[tuple[float, float], dict[tuple[float, float], float]]:
    """Grid search: which penalties make sequences agree most with drivers (Kendall tau)?

    Uses the fast solver mode (no guided search) so 15 combinations x 30 routes stays quick.
    """
    routes = list(routes)
    scores: dict[tuple[float, float], float] = {}
    for zp, sp in itertools.product(zone_grid, super_grid):
        taus = []
        for r in routes:
            order = solve_tsp(
                zone_cost(r.travel_s, r.zones, zp, sp), time_limit_s=time_limit_s, guided=False
            )
            taus.append(kendall_tau(r.actual_order, order))
        scores[(zp, sp)] = float(np.mean(taus))
    best = max(scores, key=scores.get)
    return best, scores
