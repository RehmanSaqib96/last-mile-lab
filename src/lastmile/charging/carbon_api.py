"""Client for the free GB Carbon Intensity API (no key needed).

Docs: https://carbonintensity.org.uk  ·  Base URL: https://api.carbonintensity.org.uk
Data is half-hourly gCO2/kWh. We cache every response so experiments are reproducible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from lastmile.config import settings

BASE = "https://api.carbonintensity.org.uk"


def _url_48h(postcode: str, start: datetime) -> str:
    return f"{BASE}/regional/intensity/{start:%Y-%m-%dT%H:%MZ}/fw48h/postcode/{postcode}"


def parse_regional(payload: dict) -> pd.DataFrame:
    """Turn the API JSON into a tidy table. Handles `data` as an object or a one-item list."""
    data = payload["data"]
    if isinstance(data, list):
        data = data[0]
    rows = [
        {
            "from": pd.Timestamp(e["from"]),
            "to": pd.Timestamp(e["to"]),
            "gco2_kwh": float(e["intensity"]["forecast"]),
            "index": e["intensity"].get("index"),
        }
        for e in data["data"]
    ]
    df = pd.DataFrame(rows)
    df.attrs["region"] = data.get("shortname")
    return df


def fetch_forecast_48h(
    postcode: str | None = None,
    start: datetime | None = None,
    cache_dir: Path | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    postcode = (postcode or settings.carbon_postcode).upper()
    start = start or datetime.now(UTC)
    cache_dir = cache_dir or settings.data_dir / "cache"
    cache = cache_dir / f"carbon_{postcode}_{start:%Y%m%dT%H%M}.json"
    if cache.exists():
        return parse_regional(json.loads(cache.read_text()))
    resp = requests.get(
        _url_48h(postcode, start), headers={"Accept": "application/json"}, timeout=timeout
    )
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return parse_regional(resp.json())
