"""Real UK road network with OSMnx (pip install -e ".[geo]").

First run downloads Luton's drivable roads from OpenStreetMap and caches them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lastmile.config import settings


def load_graph(place: str = "Luton, England, United Kingdom", cache: Path | None = None):
    import osmnx as ox

    cache = cache or settings.processed_dir / (place.split(",")[0].lower() + "_drive.graphml")
    if cache.exists():
        return ox.load_graphml(cache)
    G = ox.graph_from_place(place, network_type="drive")
    routing = getattr(ox, "routing", ox)  # OSMnx 2.x moved these into ox.routing
    G = routing.add_edge_speeds(G)
    G = routing.add_edge_travel_times(G)
    G = ox.truncate.largest_component(G, strongly=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache)
    return G


def nearest_node(G, lat: float, lon: float) -> int:
    import osmnx as ox

    return ox.distance.nearest_nodes(G, X=lon, Y=lat)


def travel_time_matrix(G, nodes: list[int], weight: str = "travel_time") -> np.ndarray:
    """All-pairs shortest travel time (seconds) between the given nodes. Asymmetric (one-way streets)."""
    import networkx as nx

    n = len(nodes)
    m = np.zeros((n, n))
    for i, s in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, s, weight=weight)
        for j, t in enumerate(nodes):
            m[i, j] = lengths.get(t, 1e6)  # unreachable -> huge
    return np.rint(m).astype(int)


def path_coords(G, a: int, b: int, weight: str = "travel_time") -> list[tuple[float, float]]:
    """(lon, lat) points along the road path, for drawing on the dashboard map."""
    import networkx as nx

    path = nx.shortest_path(G, a, b, weight=weight)
    return [(G.nodes[v]["x"], G.nodes[v]["y"]) for v in path]
