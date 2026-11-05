"""FastAPI service: POST orders -> get routes.

Run locally:  uvicorn lastmile.api.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lastmile import __version__
from lastmile.charging.scheduler import Van, schedule_charging
from lastmile.sim.orders import Order
from lastmile.sim.scenario import solve_day

app = FastAPI(title="Last Mile Lab", version=__version__)


class OrderIn(BaseModel):
    order_id: str
    lat: float = Field(ge=49, le=61)  # roughly Great Britain
    lon: float = Field(ge=-8, le=2)
    slot_start_h: int = Field(ge=8, le=20, description="14 means the 14:00-15:00 slot")
    parcels: int = Field(default=1, ge=1, le=50)


class RouteRequest(BaseModel):
    depot_lat: float = 51.8750
    depot_lon: float = -0.4150
    n_vans: int = Field(default=3, ge=1, le=20)
    van_capacity_parcels: int = Field(default=120, ge=1)
    time_limit_s: int = Field(default=5, ge=1, le=60)
    orders: list[OrderIn] = Field(min_length=1, max_length=400)


class Stop(BaseModel):
    order_id: str
    eta_minutes_after_shift_start: float


class VanRoute(BaseModel):
    van: int
    stops: list[Stop]


class RouteResponse(BaseModel):
    routes: list[VanRoute]
    missed_order_ids: list[str]
    kpis: dict[str, float]


class VanIn(BaseModel):
    van_id: str
    battery_kwh: float = 75
    soc_kwh: float
    next_day_km: float
    plug_in_slot: int = 0
    depart_slot: int | None = None


class ChargeRequest(BaseModel):
    vans: list[VanIn]
    gco2_kwh: list[float] = Field(min_length=1, description="one value per half-hour slot")
    price_p_kwh: list[float]
    charger_kw: float = 11
    alpha: float = Field(default=0.5, ge=0, le=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/routes", response_model=RouteResponse)
def plan_routes(req: RouteRequest) -> RouteResponse:
    orders = [Order(**o.model_dump()) for o in req.orders]
    day = solve_day(
        orders,
        (req.depot_lat, req.depot_lon),
        n_vans=req.n_vans,
        van_capacity_parcels=req.van_capacity_parcels,
        time_limit_s=req.time_limit_s,
    )
    routes = []
    for v, (r, arr) in enumerate(zip(day.vrp.routes, day.vrp.arrivals_s, strict=True)):
        stops = [
            Stop(order_id=orders[n - 1].order_id, eta_minutes_after_shift_start=a / 60)
            for n, a in zip(r[1:], arr[1:], strict=True)
        ]
        routes.append(VanRoute(van=v, stops=stops))
    return RouteResponse(
        routes=routes,
        missed_order_ids=[orders[n - 1].order_id for n in day.vrp.dropped],
        kpis={k: float(v) for k, v in day.kpis.items()},
    )


@app.post("/charging/schedule")
def plan_charging(req: ChargeRequest) -> dict:
    if len(req.price_p_kwh) != len(req.gco2_kwh):
        raise HTTPException(422, "gco2_kwh and price_p_kwh must have the same length")
    try:
        plan = schedule_charging(
            [Van(**v.model_dump()) for v in req.vans],
            np.array(req.gco2_kwh),
            np.array(req.price_p_kwh),
            charger_kw=req.charger_kw,
            alpha=req.alpha,
        )
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e
    return {
        "power_kw": plan.power_kw.round(2).to_dict(orient="list"),
        "kg_co2": plan.kg_co2,
        "cost_gbp": plan.cost_gbp,
        "kg_co2_saved_vs_plug_and_charge": plan.kg_co2_saved,
    }
