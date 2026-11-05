"""Module 4: plan tonight's charging for the vans from results/luton_day.json.

python scripts/run_charging.py              # live forecast for LU1 from the Carbon Intensity API
python scripts/run_charging.py --offline    # made-up curve, no internet needed
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from lastmile.charging.carbon_api import fetch_forecast_48h
from lastmile.charging.scheduler import Van, schedule_charging, synthetic_tariff_p

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true")
parser.add_argument("--alpha", type=float, default=0.5, help="1 = only CO2, 0 = only cost")
args = parser.parse_args()

# charging window: 18:00 tonight to 07:00 tomorrow = 26 half-hour slots
now = datetime.now(UTC)
start = now.replace(hour=18, minute=0, second=0, microsecond=0)
if now.hour >= 18:
    start += timedelta(days=1)
slots = pd.Series(pd.date_range(start, periods=26, freq="30min"))

if args.offline:
    hours = slots.dt.hour + slots.dt.minute / 60
    gco2 = 180 + 60 * np.cos((hours - 19) / 24 * 2 * np.pi)  # evening peak, low overnight
else:
    fc = fetch_forecast_48h(start=start)
    gco2 = fc.set_index("from")["gco2_kwh"].reindex(slots.values, method="nearest").to_numpy()

day_file = Path("results/luton_day.json")
next_km = json.loads(day_file.read_text())["next_day_km"] if day_file.exists() else [80, 95, 60]
rng = np.random.default_rng(0)
vans = [
    Van(f"van{i + 1}", battery_kwh=75, soc_kwh=float(rng.uniform(10, 35)), next_day_km=km)
    for i, km in enumerate(next_km)
    if km > 0
]

plan = schedule_charging(vans, gco2, synthetic_tariff_p(slots), alpha=args.alpha)
plan.power_kw.index = slots.dt.strftime("%H:%M")
print(plan.power_kw.round(1).to_string())
print(f"\nSmart:    {plan.kg_co2:.1f} kg CO2, GBP {plan.cost_gbp:.2f}")
print(f"Baseline: {plan.baseline_kg_co2:.1f} kg CO2, GBP {plan.baseline_cost_gbp:.2f}")
print(f"Saved:    {plan.kg_co2_saved:.1f} kg CO2")
