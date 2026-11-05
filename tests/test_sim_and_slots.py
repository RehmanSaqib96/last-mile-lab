import networkx as nx
import numpy as np

from lastmile.sim.network import travel_time_matrix
from lastmile.sim.orders import random_orders
from lastmile.sim.scenario import demo_day
from lastmile.slots.recommend import quote_slots


def test_random_orders_are_reproducible():
    a = random_orders(20, (51.875, -0.415), seed=3)
    b = random_orders(20, (51.875, -0.415), seed=3)
    assert [o.lat for o in a] == [o.lat for o in b]
    assert all(8 <= o.slot_start_h <= 20 for o in a)


def test_demo_day_kpis():
    day = demo_day(n_orders=25, n_vans=2, seed=1)
    k = day.kpis
    assert k["stops"] + k["missed_windows"] == 25
    assert k["km_driven"] > 0


def test_travel_time_matrix_on_toy_graph():
    G = nx.DiGraph()
    G.add_edge("a", "b", travel_time=10)
    G.add_edge("b", "c", travel_time=10)
    G.add_edge("c", "a", travel_time=50)  # one-way street back
    m = travel_time_matrix(G, ["a", "b", "c"])
    assert m[0, 2] == 20 and m[2, 0] == 50


def test_slot_quotes_prefer_cheap_insertion():
    x = np.array([0, 1, 2, 3, 2.5])  # node 4 sits between 2 and 3
    t = (np.abs(x[:, None] - x[None, :]) * 600).astype(int)
    routes = [[0, 1, 2, 3]]
    windows = {1: (0, 3600), 2: (0, 7200), 3: (0, 10800)}
    quotes = quote_slots(
        4,
        routes,
        t,
        windows,
        loads=[3],
        capacity=10,
        new_parcels=1,
        candidate_slots_h=[8, 9, 18],
        service_s=0,
    )
    feasible = [q for q in quotes if q.feasible]
    assert feasible and feasible[0].extra_time_s <= feasible[-1].extra_time_s
