"""Generate a synthetic day of home-delivery orders with 1-hour slots.

Everything here is made up on purpose: no real customers, no work data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SLOT_HOURS = list(range(8, 21))  # slots 08:00-09:00 ... 20:00-21:00
SHIFT_START_H = 7  # vans leave the depot at 07:00


@dataclass
class Order:
    order_id: str
    lat: float
    lon: float
    slot_start_h: int  # e.g. 14 means 14:00-15:00
    parcels: int

    @property
    def window_s(self) -> tuple[int, int]:
        """Window in seconds after shift start."""
        start = (self.slot_start_h - SHIFT_START_H) * 3600
        return start, start + 3600


def random_orders(
    n: int,
    center: tuple[float, float],
    radius_km: float = 6.0,
    seed: int = 0,
    slot_weights: list[float] | None = None,
) -> list[Order]:
    """Uniform points in a circle (no road network needed). Evening slots are busier by default."""
    rng = np.random.default_rng(seed)
    weights = np.array(slot_weights or [1, 1, 1, 1, 1.2, 1.2, 1, 1, 1.3, 1.6, 1.8, 1.6, 1.2])
    weights = weights / weights.sum()
    lat0, lon0 = center
    orders = []
    for k in range(n):
        r = radius_km * math.sqrt(rng.random())
        theta = rng.random() * 2 * math.pi
        dlat = (r * math.cos(theta)) / 111.0
        dlon = (r * math.sin(theta)) / (111.0 * math.cos(math.radians(lat0)))
        orders.append(
            Order(
                order_id=f"O{k:04d}",
                lat=lat0 + dlat,
                lon=lon0 + dlon,
                slot_start_h=int(rng.choice(SLOT_HOURS, p=weights)),
                parcels=int(rng.integers(1, 5)),
            )
        )
    return orders


def orders_on_graph(G, n: int, seed: int = 0) -> tuple[list[Order], list[int]]:
    """Place orders on random road-network nodes (Module 3 with OSMnx). Returns orders + node ids."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)
    picked = [nodes[i] for i in rng.choice(len(nodes), size=n, replace=False)]
    base = random_orders(n, (0.0, 0.0), seed=seed)  # reuse slot/parcel sampling
    orders = [
        Order(o.order_id, G.nodes[v]["y"], G.nodes[v]["x"], o.slot_start_h, o.parcels)
        for o, v in zip(base, picked, strict=True)
    ]
    return orders, picked
