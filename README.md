# Last Mile Lab

Routing and fleet platform for retail home delivery: plans routes with customer time slots,
learns how real drivers actually sequence their stops, and schedules EV van charging for the
lowest carbon and cost.

![CI](https://github.com/RehmanSaqib96/last-mile-lab/actions/workflows/ci.yml/badge.svg)

## What's inside

| Module | What it does | Code |
| --- | --- | --- |
| 1. Routing baseline | OR-Tools TSP and VRP with time windows, compared with real driver sequences | `src/lastmile/routing/` |
| 2. Driver behaviour | Zone-aware costs and a learned next-stop model that make plans more driver-like | `src/lastmile/behaviour/` |
| 3. UK simulation | Synthetic orders around Luton on the real road network (OSMnx), 1-hour slots, multiple vans | `src/lastmile/sim/` |
| 4. Carbon-aware charging | Overnight depot charging scheduled against the regional 48-hour carbon forecast | `src/lastmile/charging/` |
| 5. Slot recommendation | Which slots to offer a new customer, by cheapest insertion | `src/lastmile/slots/` |
| 6. Engineering | FastAPI service, Docker, tests, GitHub Actions CI | `src/lastmile/api/`, `.github/` |
| 7. Dashboard | Route map and KPIs in Streamlit | `dashboard/app.py` |

## Results

_Fill this in as you go: one table or chart per module, with the numbers from `results/`._

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
pytest
uvicorn lastmile.api.main:app --reload      # http://127.0.0.1:8000/docs
streamlit run dashboard/app.py
```

## Data and licences

- **Amazon Last Mile Routing Research Challenge dataset** (2021): 9,184 historical routes driven
  in 2018 in five US metro areas, released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Accessed from the
  [Registry of Open Data on AWS](https://registry.opendata.aws/amazon-last-mile-challenges).
  Described in Merchán et al. (2022), *Transportation Science*. Not redistributed here.
- **Carbon Intensity API** (NESO), [carbonintensity.org.uk](https://carbonintensity.org.uk).
- **Road network** © OpenStreetMap contributors, via OSMnx.
- All UK orders are synthetic. This project uses only public and synthetic data; it contains no
  data or internal knowledge from any employer.

Code: MIT licence.
