import numpy as np

from lastmile.routing.metrics import compare, kendall_tau, path_time, same_next_rate
from lastmile.routing.tsp import solve_tsp
from lastmile.routing.vrptw import solve_vrptw


def line_matrix(n: int) -> np.ndarray:
    """Stops on a line: the best open path from 0 is 0,1,2,...,n-1."""
    x = np.arange(n)
    return np.abs(x[:, None] - x[None, :]) * 10.0


def test_tsp_finds_line_order():
    assert solve_tsp(line_matrix(8), time_limit_s=1) == list(range(8))


def test_tsp_handles_asymmetric():
    m = line_matrix(5)
    m[0, 4] = 1  # a shortcut only in one direction
    order = solve_tsp(m, time_limit_s=1)
    assert order[0] == 0 and sorted(order) == list(range(5))


def test_metrics():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0
    assert same_next_rate([0, 1, 2], [0, 1, 2]) == 1.0
    assert path_time([0, 1, 2], line_matrix(3)) == 20.0
    out = compare([0, 2, 1], [0, 1, 2], line_matrix(3), ["", "A", "B"])
    assert out["time_gap_pct"] > 0


def test_vrptw_respects_windows_and_capacity():
    n = 7
    t = (line_matrix(n) * 6).astype(int)  # 60 s per unit
    windows = [(0, 10_000)] + [(0, 10_000)] * (n - 1)
    windows[6] = (0, 100)  # impossible: node 6 is 360 s away -> must be dropped
    demands = [0, 1, 1, 1, 1, 1, 1]
    res = solve_vrptw(t, windows, demands, vehicle_capacity=3, num_vehicles=2, time_limit_s=1)
    assert 6 in res.dropped
    served = [s for r in res.routes for s in r[1:]]
    assert sorted(served) == [1, 2, 3, 4, 5]
    for r in res.routes:
        assert sum(demands[s] for s in r) <= 3
