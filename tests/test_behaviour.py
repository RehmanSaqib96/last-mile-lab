from lastmile.amazon.loader import iter_arrays, load_sample, make_sample
from lastmile.behaviour.next_stop import learned_cost, train
from lastmile.behaviour.zones import parse_zone, super_zone, zone_cost
from lastmile.routing.metrics import kendall_tau
from lastmile.routing.tsp import solve_tsp


def test_zone_parsing():
    assert parse_zone("P-12.3C") == ("P", 12, 3, "C")
    assert super_zone("P-12.3C") == "P-12"
    assert parse_zone(None) is None and parse_zone("junk") is None


def _routes(fake_amazon_dir, tmp_path):
    out = tmp_path / "s.json"
    make_sample(fake_amazon_dir, out, n=6)
    return list(iter_arrays(load_sample(out)))


def test_zone_penalty_moves_towards_driver(fake_amazon_dir, tmp_path):
    routes = _routes(fake_amazon_dir, tmp_path)
    plain = [kendall_tau(r.actual_order, solve_tsp(r.travel_s, time_limit_s=1)) for r in routes]
    zoned = [
        kendall_tau(
            r.actual_order, solve_tsp(zone_cost(r.travel_s, r.zones, 600, 1200), time_limit_s=1)
        )
        for r in routes
    ]
    assert sum(zoned) >= sum(plain)


def test_learned_cost_shape(fake_amazon_dir, tmp_path):
    routes = _routes(fake_amazon_dir, tmp_path)
    model = train(routes[:4])
    c = learned_cost(model, routes[4])
    assert c.shape == (routes[4].n, routes[4].n)
    assert (c >= 0).all()
