"""How close is a planned sequence to what the driver actually did?

These are simple, explainable metrics. The challenge's official score (sequence
deviation x edit distance with real penalty) is in the challenge's rc-cli repo;
port it later if you want numbers comparable to the published papers.
"""

from __future__ import annotations

import numpy as np


def path_time(order: list[int], cost: np.ndarray, include_return: bool = False) -> float:
    total = sum(cost[a, b] for a, b in zip(order[:-1], order[1:], strict=False))
    if include_return and order:
        total += cost[order[-1], order[0]]
    return float(total)


def kendall_tau(actual: list[int], predicted: list[int]) -> float:
    """Rank agreement between two orders of the same stops: 1 = identical, -1 = reversed."""
    pos = {s: i for i, s in enumerate(predicted)}
    seq = [pos[s] for s in actual]
    n = len(seq)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def same_next_rate(actual: list[int], predicted: list[int]) -> float:
    """Share of stops whose next stop is the same in both sequences."""
    nxt = {a: b for a, b in zip(predicted[:-1], predicted[1:], strict=False)}
    pairs = list(zip(actual[:-1], actual[1:], strict=False))
    if not pairs:
        return 1.0
    return sum(nxt.get(a) == b for a, b in pairs) / len(pairs)


def zone_order(order: list[int], zones: list[str | None]) -> list[str]:
    """Collapse a stop order into the order zones are first visited."""
    seen, out = set(), []
    for s in order:
        z = zones[s]
        if z and z not in seen:
            seen.add(z)
            out.append(z)
    return out


def compare(
    actual: list[int],
    predicted: list[int],
    travel: np.ndarray,
    zones: list[str | None] | None = None,
) -> dict[str, float]:
    t_act, t_pred = path_time(actual, travel), path_time(predicted, travel)
    out = {
        "actual_time_s": t_act,
        "planned_time_s": t_pred,
        "time_gap_pct": 100 * (t_act - t_pred) / t_pred if t_pred else 0.0,
        "kendall_tau": kendall_tau(actual, predicted),
        "same_next_rate": same_next_rate(actual, predicted),
    }
    if zones is not None:
        za, zp = zone_order(actual, zones), zone_order(predicted, zones)
        common = [z for z in za if z in set(zp)]
        zp_common = [z for z in zp if z in set(common)]
        out["zone_kendall_tau"] = kendall_tau(common, zp_common)
    return out
