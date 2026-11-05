"""Module 3: one synthetic delivery day around Luton.

python scripts/run_luton_day.py               # fast, straight-line travel times
python scripts/run_luton_day.py --osm         # real roads via OSMnx (pip install -e ".[geo]")
"""

import argparse
import json
from pathlib import Path

from lastmile.config import settings
from lastmile.geo import distance_matrix_km
from lastmile.sim.orders import random_orders
from lastmile.sim.scenario import solve_day

parser = argparse.ArgumentParser()
parser.add_argument("--orders", type=int, default=80)
parser.add_argument("--vans", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--osm", action="store_true")
args = parser.parse_args()

depot = (settings.depot_lat, settings.depot_lon)
time_s = dist_km = None
if args.osm:
    from lastmile.sim.network import load_graph, nearest_node, travel_time_matrix
    from lastmile.sim.orders import orders_on_graph

    G = load_graph()
    orders, nodes = orders_on_graph(G, args.orders, seed=args.seed)
    all_nodes = [nearest_node(G, *depot)] + nodes
    time_s = travel_time_matrix(G, all_nodes)
    dist_km = travel_time_matrix(G, all_nodes, weight="length") / 1000
else:
    orders = random_orders(args.orders, depot, seed=args.seed)
    dist_km = distance_matrix_km([depot] + [(o.lat, o.lon) for o in orders])

day = solve_day(orders, depot, n_vans=args.vans, time_s=time_s, dist_km=dist_km)
for v, r in enumerate(day.vrp.routes):
    print(f"Van {v + 1}: {len(r) - 1} stops")
print(json.dumps({k: round(v, 2) for k, v in day.kpis.items()}, indent=2))

Path("results").mkdir(exist_ok=True)
Path("results/luton_day.json").write_text(
    json.dumps(
        {
            "kpis": day.kpis,
            "routes": day.vrp.routes,
            "next_day_km": [
                sum(day.dist_km[a, b] for a, b in zip(r, r[1:] + r[:1], strict=False))
                for r in day.vrp.routes
            ],
        },
        default=float,
        indent=2,
    )
)
