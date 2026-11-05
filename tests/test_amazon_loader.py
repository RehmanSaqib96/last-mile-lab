from lastmile.amazon.loader import iter_arrays, load_sample, make_sample


def test_make_sample_and_arrays(fake_amazon_dir, tmp_path):
    out = tmp_path / "sample.json"
    ids = make_sample(fake_amazon_dir, out, n=3)
    assert len(ids) == 3

    routes = list(iter_arrays(load_sample(out)))
    r = routes[0]
    assert r.stop_ids[0] == "ST"  # station first
    assert r.actual_order[0] == 0
    assert sorted(r.actual_order) == list(range(r.n))
    assert r.travel_s.shape == (r.n, r.n)
    assert r.n_packages[1:].sum() == r.n - 1
    assert r.tw_end_s[1] == float("inf")  # NaN windows become "no window"
