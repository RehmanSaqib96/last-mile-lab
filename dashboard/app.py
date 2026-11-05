"""Module 7: Streamlit dashboard.

Run:  streamlit run dashboard/app.py
Needs: pip install -e ".[dashboard]"
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from lastmile.config import settings
from lastmile.sim.orders import random_orders
from lastmile.sim.scenario import solve_day

st.set_page_config(page_title="Last Mile Lab", layout="wide")
st.title("Last Mile Lab: Luton delivery day")
st.caption("Synthetic orders only. Straight-line x 1.3 travel times unless you switch on OSMnx.")

with st.sidebar:
    n_orders = st.slider("Orders", 10, 200, 60, step=10)
    n_vans = st.slider("Vans", 1, 10, 3)
    capacity = st.slider("Van capacity (parcels)", 20, 300, 120, step=10)
    seed = st.number_input("Random seed", 0, 9999, 0)
    run = st.button("Plan routes", type="primary")

if run or "day" not in st.session_state:
    depot = (settings.depot_lat, settings.depot_lon)
    with st.spinner("Solving VRP with time windows..."):
        st.session_state.day = solve_day(
            random_orders(n_orders, depot, seed=int(seed)),
            depot,
            n_vans=n_vans,
            van_capacity_parcels=capacity,
        )

day = st.session_state.day
k = day.kpis
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stops per hour", f"{k['stops_per_hour']:.1f}")
c2.metric("Km driven", f"{k['km_driven']:.0f}")
c3.metric("Missed windows", int(k["missed_windows"]))
c4.metric("Vans used", int(k["vans_used"]))
c5.metric("kg CO2 saved vs diesel", f"{k['kg_co2_saved_vs_diesel']:.1f}")

palette = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
]
paths, pts = [], []
for v, route in enumerate(day.vrp.routes):
    if len(route) < 2:
        continue
    coords = [[day.points[n][1], day.points[n][0]] for n in route + [route[0]]]
    paths.append({"van": f"Van {v + 1}", "path": coords, "color": palette[v % len(palette)]})
    for n in route[1:]:
        o = day.orders[n - 1]
        pts.append(
            {
                "lon": o.lon,
                "lat": o.lat,
                "label": f"{o.order_id} slot {o.slot_start_h}:00",
                "color": palette[v % len(palette)],
            }
        )

layers = [
    pdk.Layer(
        "PathLayer", paths, get_path="path", get_color="color", width_min_pixels=3, pickable=True
    ),
    pdk.Layer(
        "ScatterplotLayer",
        pts,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=60,
        pickable=True,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        [{"lon": day.points[0][1], "lat": day.points[0][0]}],
        get_position="[lon, lat]",
        get_fill_color=[0, 0, 0],
        get_radius=150,
    ),
]
view = pdk.ViewState(latitude=day.points[0][0], longitude=day.points[0][1], zoom=11.5)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{van}{label}"}))

rows = []
for v, (r, a) in enumerate(zip(day.vrp.routes, day.vrp.arrivals_s, strict=True)):
    for n, t in zip(r[1:], a[1:], strict=True):
        o = day.orders[n - 1]
        rows.append(
            {
                "van": v + 1,
                "order": o.order_id,
                "slot": f"{o.slot_start_h}:00",
                "eta": f"{7 + t // 3600:02d}:{(t % 3600) // 60:02d}",
            }
        )
st.subheader("Stop list")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
