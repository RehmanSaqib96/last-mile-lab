import numpy as np
import pytest

from lastmile.charging.carbon_api import parse_regional
from lastmile.charging.scheduler import Van, schedule_charging

PAYLOAD = {
    "data": {
        "regionid": 10,
        "shortname": "East England",
        "postcode": "LU1",
        "data": [
            {
                "from": "2026-10-26T18:00Z",
                "to": "2026-10-26T18:30Z",
                "intensity": {"forecast": 210, "index": "moderate"},
                "generationmix": [],
            },
            {
                "from": "2026-10-26T18:30Z",
                "to": "2026-10-26T19:00Z",
                "intensity": {"forecast": 190, "index": "moderate"},
                "generationmix": [],
            },
        ],
    }
}


def test_parse_regional_object_and_list():
    df = parse_regional(PAYLOAD)
    assert list(df["gco2_kwh"]) == [210.0, 190.0]
    assert df.attrs["region"] == "East England"
    df2 = parse_regional({"data": [PAYLOAD["data"]]})
    assert len(df2) == 2


def test_scheduler_charges_in_clean_slots_and_meets_demand():
    gco2 = np.array([300, 300, 50, 50, 300, 300], dtype=float)
    price = np.full(6, 20.0)
    # needs 36*0.25 + 10 reserve - 10 already in battery = 9 kWh; fits in the two clean slots
    van = Van("v1", battery_kwh=75, soc_kwh=10, next_day_km=36)
    plan = schedule_charging([van], gco2, price, alpha=1.0)
    p = plan.power_kw["v1"].to_numpy()
    assert p[[2, 3]].sum() > 0 and p[[0, 1, 4, 5]].sum() == pytest.approx(0, abs=1e-6)
    delivered = p.sum() * 0.5 * 0.9
    assert delivered == pytest.approx(9, rel=1e-4)
    assert plan.kg_co2 < plan.baseline_kg_co2


def test_scheduler_infeasible_raises():
    van = Van("v1", battery_kwh=75, soc_kwh=0, next_day_km=200, depart_slot=1)
    with pytest.raises(RuntimeError):
        schedule_charging([van], np.array([100.0, 100.0]), np.array([10.0, 10.0]))
