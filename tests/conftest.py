"""Shared fixtures: a tiny fake dataset with the same shape as the Amazon files."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest


def _fake_route(rng, n_stops: int):
    stops = {"ST": {"lat": 34.0, "lng": -118.0, "type": "Station", "zone_id": None}}
    for k in range(n_stops):
        zone = f"A-{1 + k // 4}.{1 + (k // 2) % 2}A"
        stops[f"S{k:02d}"] = {
            "lat": 34.0 + rng.random() / 50,
            "lng": -118.0 + rng.random() / 50,
            "type": "Dropoff",
            "zone_id": zone,
        }
    ids = list(stops)
    travel = {}
    for a in ids:
        travel[a] = {}
        for b in ids:
            d = math.dist((stops[a]["lat"], stops[a]["lng"]), (stops[b]["lat"], stops[b]["lng"]))
            travel[a][b] = 0.0 if a == b else round(d * 100000 / 8 + rng.random() * 5, 1)
    # the "driver" visits stops in zone order (which is what we want models to learn)
    order = ["ST"] + sorted(ids[1:], key=lambda s: (stops[s]["zone_id"], s))
    actual = {"actual": {s: i for i, s in enumerate(order)}}
    packages = {
        s: {
            f"P{s}": {
                "scan_status": "DELIVERED",
                "planned_service_time_seconds": 60.0,
                "time_window": {
                    "start_time_utc": "NaN_PLACEHOLDER",
                    "end_time_utc": "NaN_PLACEHOLDER",
                },
                "dimensions": {"depth_cm": 20, "height_cm": 10, "width_cm": 15},
            }
        }
        for s in ids[1:]
    }
    route = {
        "station_code": "DLA3",
        "date_YYYY_MM_DD": "2018-07-27",
        "departure_time_utc": "16:00:00",
        "executor_capacity_cm3": 3313071.0,
        "route_score": "High",
        "stops": stops,
    }
    return route, actual, travel, packages


@pytest.fixture
def fake_amazon_dir(tmp_path):
    rng = np.random.default_rng(1)
    base = tmp_path / "train" / "model_build_inputs"
    base.mkdir(parents=True)
    files = {
        "route_data.json": {},
        "actual_sequences.json": {},
        "travel_times.json": {},
        "package_data.json": {},
    }
    for k in range(6):
        rid = f"RouteID_{k}"
        route, actual, travel, packages = _fake_route(rng, n_stops=12)
        files["route_data.json"][rid] = route
        files["actual_sequences.json"][rid] = actual
        files["travel_times.json"][rid] = travel
        files["package_data.json"][rid] = packages
    for name, content in files.items():
        text = json.dumps(content).replace('"NaN_PLACEHOLDER"', "NaN")  # real files contain NaN
        (base / name).write_text(text)
    return tmp_path / "train"
