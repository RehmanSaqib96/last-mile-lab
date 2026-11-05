"""Single-vehicle sequencing with OR-Tools (works for asymmetric matrices)."""

from __future__ import annotations

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_tsp(
    cost: np.ndarray,
    depot: int = 0,
    open_path: bool = True,
    time_limit_s: int = 5,
    guided: bool = True,
) -> list[int]:
    """Return a visiting order that starts at `depot`.

    open_path=True means the van does not need to come back (return arcs cost 0),
    which matches how the Amazon sequences are scored.
    `cost` can be travel time or any learned cost matrix (Module 2).
    guided=True runs guided local search for the full time limit (best quality);
    guided=False stops at the first local optimum (much faster, good for grid searches).
    """
    cost = np.asarray(cost)
    n = cost.shape[0]
    if n <= 2:
        return list(range(n))
    scaled = np.rint(cost * 10).astype(np.int64)  # OR-Tools needs integers; keep 0.1 precision

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        if open_path and j == depot:
            return 0
        return int(scaled[i, j])

    transit = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if guided:
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
    params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution")

    order, index = [], routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return order
