"""Cheapest-insertion slot pricing.

For a new customer, try inserting them into every position of every planned route,
once per candidate slot. A slot is offered only if some insertion keeps every
existing promise (time windows) and the van's capacity. Cheaper slots can be shown
first or discounted, which is how "green slots" work in online grocery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlotQuote:
    slot_start_h: int
    feasible: bool
    extra_time_s: float  # added driving + service time; inf if infeasible
    van: int | None
    position: int | None


def simulate(
    route: list[int],
    time_s: np.ndarray,
    windows: dict[int, tuple[int, int]],
    service_s: int,
    start_s: int = 0,
) -> tuple[bool, float]:
    """Drive the route; wait if early, fail if late. Returns (feasible, finish_time)."""
    t = start_s
    for a, b in zip(route[:-1], route[1:], strict=False):
        t += (service_s if a != route[0] else 0) + time_s[a, b]
        lo, hi = windows.get(b, (0, 10**9))
        if t > hi:
            return False, float("inf")
        t = max(t, lo)
    return True, float(t)


def quote_slots(
    new_node: int,
    routes: list[list[int]],
    time_s: np.ndarray,
    windows: dict[int, tuple[int, int]],
    loads: list[int],
    capacity: int,
    new_parcels: int,
    candidate_slots_h: list[int],
    shift_start_h: int = 7,
    service_s: int = 120,
) -> list[SlotQuote]:
    quotes = []
    for slot in candidate_slots_h:
        lo = (slot - shift_start_h) * 3600
        w = dict(windows)
        w[new_node] = (lo, lo + 3600)
        best = SlotQuote(slot, False, float("inf"), None, None)
        for v, route in enumerate(routes):
            if loads[v] + new_parcels > capacity:
                continue
            ok, base_end = simulate(route, time_s, w, service_s)
            if not ok:
                continue
            for pos in range(1, len(route) + 1):
                trial = route[:pos] + [new_node] + route[pos:]
                ok, end = simulate(trial, time_s, w, service_s)
                if ok and end - base_end < best.extra_time_s:
                    best = SlotQuote(slot, True, end - base_end, v, pos)
        quotes.append(best)
    return sorted(quotes, key=lambda q: q.extra_time_s)
