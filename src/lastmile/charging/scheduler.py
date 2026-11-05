"""Overnight depot charging as a linear programme (OR-Tools GLOP).

Decision: how many kW each van draws in each half-hour slot.
Constraints: van only charges while plugged in, charger power limit, depot grid limit,
battery capacity, and enough energy for tomorrow's route (+ reserve).
Objective: alpha * kg CO2 + (1 - alpha) * GBP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


@dataclass
class Van:
    van_id: str
    battery_kwh: float
    soc_kwh: float  # energy left when it gets back to the depot
    next_day_km: float  # from tomorrow's planned route (Module 3)
    plug_in_slot: int = 0  # first slot it can charge
    depart_slot: int | None = None  # first slot it can NOT charge (None = end of horizon)


@dataclass
class ChargingPlan:
    power_kw: pd.DataFrame  # rows = slots, cols = vans
    kg_co2: float
    cost_gbp: float
    baseline_kg_co2: float
    baseline_cost_gbp: float

    @property
    def kg_co2_saved(self) -> float:
        return self.baseline_kg_co2 - self.kg_co2


def synthetic_tariff_p(slot_starts: pd.Series) -> np.ndarray:
    """ASSUMPTION: a simple time-of-use tariff, cheap 00:30-04:30. Swap for a real tariff later."""
    mins = slot_starts.dt.hour * 60 + slot_starts.dt.minute
    return np.where((mins >= 30) & (mins < 270), 7.5, 28.0)


def energy_needed(v: Van, kwh_per_km: float, reserve_kwh: float) -> float:
    need = v.next_day_km * kwh_per_km + reserve_kwh - v.soc_kwh
    return float(np.clip(need, 0, v.battery_kwh - v.soc_kwh))


def _baseline(vans, need, ci, price, charger_kw, slot_h, eff):
    """'Dumb' charging: full power from plug-in until done."""
    kg = gbp = 0.0
    for v, e in zip(vans, need, strict=True):
        remaining, t = e, v.plug_in_slot
        end = v.depart_slot if v.depart_slot is not None else len(ci)
        while remaining > 1e-9 and t < end:
            grid_kwh = min(charger_kw * slot_h, remaining / eff)
            remaining -= grid_kwh * eff
            kg += grid_kwh * ci[t] / 1000
            gbp += grid_kwh * price[t] / 100
            t += 1
    return kg, gbp


def schedule_charging(
    vans: list[Van],
    gco2_kwh: np.ndarray,
    price_p_kwh: np.ndarray,
    slot_h: float = 0.5,
    charger_kw: float = 11.0,
    site_limit_kw: float | None = None,
    kwh_per_km: float = 0.25,
    reserve_kwh: float = 10.0,
    efficiency: float = 0.9,
    alpha: float = 0.5,
) -> ChargingPlan:
    ci = np.asarray(gco2_kwh, dtype=float)
    price = np.asarray(price_p_kwh, dtype=float)
    T = len(ci)
    need = [energy_needed(v, kwh_per_km, reserve_kwh) for v in vans]

    solver = pywraplp.Solver.CreateSolver("GLOP")
    p = {}
    for k, v in enumerate(vans):
        end = v.depart_slot if v.depart_slot is not None else T
        for t in range(T):
            ub = charger_kw if v.plug_in_slot <= t < end else 0.0
            p[k, t] = solver.NumVar(0.0, ub, f"p_{k}_{t}")
        # deliver at least the energy needed (into the battery, after losses)
        solver.Add(sum(p[k, t] for t in range(T)) * slot_h * efficiency >= need[k])
    if site_limit_kw is not None:
        for t in range(T):
            solver.Add(sum(p[k, t] for k in range(len(vans))) <= site_limit_kw)

    # normalise both terms so alpha is a fair trade-off between kg and GBP
    kg_w = ci / 1000 / max(ci.max() / 1000, 1e-9)
    gbp_w = price / 100 / max(price.max() / 100, 1e-9)
    weight = alpha * kg_w + (1 - alpha) * gbp_w
    solver.Minimize(sum(p[k, t] * slot_h * weight[t] for k in range(len(vans)) for t in range(T)))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No feasible charging plan: not enough plug-in time or charger power")

    power = np.array([[p[k, t].solution_value() for k in range(len(vans))] for t in range(T)])
    grid_kwh = power.sum(axis=1) * slot_h
    base_kg, base_gbp = _baseline(vans, need, ci, price, charger_kw, slot_h, efficiency)
    return ChargingPlan(
        power_kw=pd.DataFrame(power, columns=[v.van_id for v in vans]),
        kg_co2=float((grid_kwh * ci / 1000).sum()),
        cost_gbp=float((grid_kwh * price / 100).sum()),
        baseline_kg_co2=base_kg,
        baseline_cost_gbp=base_gbp,
    )
