"""Multi-van routing with time windows and capacity (CVRPTW) using OR-Tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class VrpResult:
    routes: list[list[int]]  # per van: node indices, starting at depot (depot not repeated at end)
    arrivals_s: list[list[int]]  # arrival time at each node in `routes`
    dropped: list[int]  # orders that could not be served in their window
    total_time_s: int


def solve_vrptw(
    time_s: np.ndarray,
    time_windows: list[tuple[int, int]],
    demands: list[int],
    vehicle_capacity: int,
    num_vehicles: int,
    service_s: list[int] | None = None,
    depot: int = 0,
    horizon_s: int = 24 * 3600,
    drop_penalty: int = 100_000,
    time_limit_s: int = 10,
) -> VrpResult:
    """time_windows[i] = (earliest, latest) arrival in seconds from shift start.

    Orders may be dropped (at a big penalty) so that the model always returns something;
    dropped orders become "missed time windows" in the KPIs.
    """
    time_s = np.asarray(time_s, dtype=int)
    n = time_s.shape[0]
    service = service_s or [0] * n

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index: int, to_index: int) -> int:
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(time_s[i, j] + service[i])

    t_cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(t_cb)
    routing.AddDimension(t_cb, 3600, horizon_s, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node, (start, end) in enumerate(time_windows):
        if node == depot:
            continue
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(start), int(end))
    for v in range(num_vehicles):
        lo, hi = time_windows[depot]
        time_dim.CumulVar(routing.Start(v)).SetRange(int(lo), int(hi))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    def demand(from_index: int) -> int:
        return int(demands[manager.IndexToNode(from_index)])

    d_cb = routing.RegisterUnaryTransitCallback(demand)
    routing.AddDimensionWithVehicleCapacity(
        d_cb, 0, [vehicle_capacity] * num_vehicles, True, "Capacity"
    )

    for node in range(n):
        if node != depot:
            routing.AddDisjunction([manager.NodeToIndex(node)], drop_penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_s

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")

    routes, arrivals, visited = [], [], {depot}
    for v in range(num_vehicles):
        idx, r, a = routing.Start(v), [], []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            r.append(node)
            a.append(sol.Value(time_dim.CumulVar(idx)))
            visited.add(node)
            idx = sol.Value(routing.NextVar(idx))
        routes.append(r)
        arrivals.append(a)

    dropped = [i for i in range(n) if i not in visited]
    total = sum(
        sol.Value(time_dim.CumulVar(routing.End(v)))
        - sol.Value(time_dim.CumulVar(routing.Start(v)))
        for v in range(num_vehicles)
    )
    return VrpResult(routes=routes, arrivals_s=arrivals, dropped=dropped, total_time_s=total)
