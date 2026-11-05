from fastapi.testclient import TestClient

from lastmile.api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_routes_endpoint():
    orders = [
        {
            "order_id": f"o{i}",
            "lat": 51.875 + i * 0.002,
            "lon": -0.415 + i * 0.003,
            "slot_start_h": 9 + i % 4,
            "parcels": 1,
        }
        for i in range(10)
    ]
    r = client.post("/routes", json={"n_vans": 2, "time_limit_s": 1, "orders": orders})
    assert r.status_code == 200
    body = r.json()
    served = [s["order_id"] for v in body["routes"] for s in v["stops"]]
    assert len(served) + len(body["missed_order_ids"]) == 10


def test_routes_rejects_bad_input():
    r = client.post(
        "/routes", json={"orders": [{"order_id": "x", "lat": 10, "lon": 0, "slot_start_h": 9}]}
    )
    assert r.status_code == 422


def test_charging_endpoint():
    body = {
        "vans": [{"van_id": "v1", "soc_kwh": 10, "next_day_km": 40}],
        "gco2_kwh": [300, 50, 50, 300],
        "price_p_kwh": [20, 20, 20, 20],
        "alpha": 1,
    }
    r = client.post("/charging/schedule", json=body)
    assert r.status_code == 200
    assert r.json()["kg_co2_saved_vs_plug_and_charge"] >= 0
