"""Level 2 of driver behaviour: learn P(next stop = j | current stop = i).

Train a classifier on real driver moves, turn its probabilities into a cost matrix
(cost = -log p), and let the TSP solver find the most "driver-like" sequence.
Needs scikit-learn:  pip install -e ".[ml]"
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lastmile.amazon.loader import RouteArrays
from lastmile.behaviour.zones import parse_zone, super_zone

FEATURES = [
    "travel_s",
    "travel_rank",
    "same_zone",
    "same_super_zone",
    "zone_num_gap",
    "tw_end_h",
    "n_packages",
    "to_station",
]


def _zone_parts(zones: list[str | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([x or "" for x in zones], dtype=object)
    sz = np.array([super_zone(x) or "" for x in zones], dtype=object)
    num = np.array([(p[2] if (p := parse_zone(x)) else -1) for x in zones], dtype=float)
    return z, sz, num


def features_from(r: RouteArrays, i: int, candidates: np.ndarray) -> np.ndarray:
    """One row per candidate next stop j, given the van is at stop i."""
    z, sz, num = _zone_parts(r.zones)
    t = r.travel_s[i, candidates]
    rank = t.argsort().argsort() / max(len(candidates) - 1, 1)
    tw_end_h = np.clip(np.nan_to_num(r.tw_end_s[candidates], posinf=36 * 3600) / 3600, 0, 36)
    same_zone = (z[candidates] == z[i]) & (z[i] != "")
    same_super = (sz[candidates] == sz[i]) & (sz[i] != "")
    gap = np.where((num[candidates] >= 0) & (num[i] >= 0), np.abs(num[candidates] - num[i]), 10)
    return np.column_stack(
        [
            t,
            rank,
            same_zone,
            same_super,
            gap,
            tw_end_h,
            r.n_packages[candidates],
            candidates == 0,
        ]
    ).astype(float)


def training_rows(routes: Iterable[RouteArrays], n_neg: int = 15, seed: int = 0):
    """For each real move i->j*, keep j* (label 1) and the n_neg closest other stops (label 0)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for r in routes:
        order = r.actual_order
        for a, b in zip(order[:-1], order[1:], strict=False):
            remaining = np.array([s for s in range(1, r.n) if s != a])
            near = remaining[np.argsort(r.travel_s[a, remaining])[:n_neg]]
            cands = np.unique(np.append(near, b))
            rng.shuffle(cands)
            X.append(features_from(r, a, cands))
            y.append((cands == b).astype(int))
    return np.vstack(X), np.concatenate(y)


def train(routes: Iterable[RouteArrays]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = training_rows(routes)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(X, y)
    return model


def learned_cost(model, r: RouteArrays, eps: float = 1e-4) -> np.ndarray:
    """Cost matrix where likely driver moves are cheap: cost[i, j] = -log P(j | i)."""
    n = r.n
    cost = np.zeros((n, n))
    all_stops = np.arange(n)
    for i in range(n):
        p = model.predict_proba(features_from(r, i, all_stops))[:, 1]
        cost[i] = -np.log(p + eps) * 100  # scale so integer rounding keeps detail
    np.fill_diagonal(cost, 0)
    return cost
