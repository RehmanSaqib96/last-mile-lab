"""Stream the (very large) Amazon JSON files and turn routes into arrays.

The training folder `model_build_inputs/` contains:
  route_data.json        route -> station, date, departure time, capacity, route_score, stops{lat,lng,type,zone_id}
  actual_sequences.json  route -> {"actual": {stop_id: position}}   (station is position 0)
  travel_times.json      route -> {from_stop: {to_stop: seconds}}    (several GB: never json.load it whole)
  package_data.json      route -> stop -> package -> {time_window, planned_service_time_seconds, dimensions}

Check the field names against the official data_structures.md if anything fails to parse.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import ijson
import numpy as np

BUILD = "model_build_inputs"
FILES = {
    "route": "route_data.json",
    "actual": "actual_sequences.json",
    "travel": "travel_times.json",
    "packages": "package_data.json",
}


class _NaNToNull:
    """File wrapper that rewrites the non-standard JSON token NaN to null while streaming."""

    def __init__(self, raw):
        self.raw = raw
        self.carry = b""

    def read(self, size: int = 65536) -> bytes:
        chunk = self.raw.read(size)
        data = self.carry + chunk
        if chunk:
            # keep the last 2 bytes in case "NaN" is split across chunks
            data, self.carry = data[:-2], data[-2:]
        else:
            self.carry = b""
        return data.replace(b"NaN", b"null")


def iter_top_level(path: Path, keep: set[str] | None = None) -> Iterator[tuple[str, Any]]:
    """Yield (route_id, value) pairs from a top-level JSON object without loading it all."""
    with open(path, "rb") as raw:
        for key, value in ijson.kvitems(_NaNToNull(raw), "", use_float=True):
            if keep is None or key in keep:
                yield key, value


def make_sample(
    train_dir: Path,
    out_path: Path,
    n: int = 200,
    route_score: str | None = "High",
    seed: int = 42,
) -> list[str]:
    """Pick n routes and copy everything about them into one small JSON file.

    Run once (it takes a few minutes); every experiment afterwards loads the small file.
    """
    base = Path(train_dir) / BUILD
    routes = {
        rid: r
        for rid, r in iter_top_level(base / FILES["route"])
        if route_score is None or r.get("route_score") == route_score
    }
    rng = random.Random(seed)
    chosen = sorted(rng.sample(sorted(routes), k=min(n, len(routes))))
    keep = set(chosen)

    sample: dict[str, dict[str, Any]] = {rid: {"route": routes[rid]} for rid in chosen}
    for part in ("actual", "travel", "packages"):
        for rid, value in iter_top_level(base / FILES[part], keep):
            sample[rid][part] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample), encoding="utf-8")
    return chosen


def load_sample(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class RouteArrays:
    """One route as numbers: index 0 is always the station."""

    route_id: str
    stop_ids: list[str]
    lat: np.ndarray
    lng: np.ndarray
    zones: list[str | None]
    travel_s: np.ndarray  # (n, n) seconds, asymmetric
    actual_order: list[int]  # indices into stop_ids, starts with 0
    service_s: np.ndarray  # planned service seconds per stop
    tw_start_s: np.ndarray  # time window start, seconds after departure (0 if none)
    tw_end_s: np.ndarray  # time window end, seconds after departure (inf if none)
    n_packages: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.stop_ids)


def _parse_ts(value: Any) -> datetime | None:
    if not value or (isinstance(value, float) and math.isnan(value)):
        return None
    return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")


def to_arrays(route_id: str, entry: dict[str, Any]) -> RouteArrays:
    route = entry["route"]
    stops: dict[str, dict] = route["stops"]
    station = [s for s, v in stops.items() if v.get("type") == "Station"]
    others = sorted(s for s in stops if s not in station)
    stop_ids = station + others
    idx = {s: i for i, s in enumerate(stop_ids)}
    n = len(stop_ids)

    travel = entry["travel"]
    travel_s = np.array([[float(travel[a][b]) for b in stop_ids] for a in stop_ids])

    positions = entry["actual"]["actual"]
    actual_order = [idx[s] for s, _ in sorted(positions.items(), key=lambda kv: kv[1])]

    depart = datetime.strptime(
        f"{route['date_YYYY_MM_DD']} {route['departure_time_utc']}", "%Y-%m-%d %H:%M:%S"
    )
    service = np.zeros(n)
    tw_start = np.zeros(n)
    tw_end = np.full(n, np.inf)
    n_pk = np.zeros(n, dtype=int)
    for stop, pkgs in (entry.get("packages") or {}).items():
        i = idx.get(stop)
        if i is None:
            continue
        for p in pkgs.values():
            n_pk[i] += 1
            service[i] += float(p.get("planned_service_time_seconds") or 0.0)
            tw = p.get("time_window") or {}
            s, e = _parse_ts(tw.get("start_time_utc")), _parse_ts(tw.get("end_time_utc"))
            if s and e:
                tw_start[i] = max(tw_start[i], (s - depart).total_seconds())
                tw_end[i] = min(tw_end[i], (e - depart).total_seconds())

    return RouteArrays(
        route_id=route_id,
        stop_ids=stop_ids,
        lat=np.array([stops[s]["lat"] for s in stop_ids]),
        lng=np.array([stops[s]["lng"] for s in stop_ids]),
        zones=[stops[s].get("zone_id") for s in stop_ids],
        travel_s=travel_s,
        actual_order=actual_order,
        service_s=service,
        tw_start_s=tw_start,
        tw_end_s=tw_end,
        n_packages=n_pk,
        meta={
            k: route.get(k)
            for k in ("station_code", "date_YYYY_MM_DD", "route_score", "executor_capacity_cm3")
        },
    )


def iter_arrays(sample: dict[str, dict[str, Any]]) -> Iterable[RouteArrays]:
    for rid, entry in sample.items():
        yield to_arrays(rid, entry)
