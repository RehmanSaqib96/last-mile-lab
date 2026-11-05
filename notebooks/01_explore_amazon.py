# %% [markdown]
# # Explore the Amazon Last Mile sample
# In VS Code, click "Run Cell" above each `# %%` line (needs the Jupyter extension).
# Run `python scripts/make_sample.py` first.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lastmile.amazon.loader import iter_arrays, load_sample

sample = load_sample(Path("data/processed/amazon_sample.json"))
routes = list(iter_arrays(sample))
print(len(routes), "routes")

# %% Route sizes and cities
df = pd.DataFrame([{"route_id": r.route_id, "stops": r.n - 1, **r.meta} for r in routes])
df["city"] = df["station_code"].str[1:3]  # e.g. DLA3 -> LA
print(df.groupby("city")["stops"].describe())

# %% One route: actual driver path coloured by zone
r = routes[0]
xs, ys = r.lng[r.actual_order], r.lat[r.actual_order]
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(xs, ys, "-", lw=0.8, color="grey")
zones = pd.Series([r.zones[i] for i in r.actual_order]).fillna("station")
for z in zones.unique():
    m = (zones == z).to_numpy()
    ax.scatter(xs[m], ys[m], s=12, label=z)
ax.set_title(f"{r.route_id}: actual sequence ({r.n - 1} stops)")
ax.legend(fontsize=6, ncol=2)
plt.show()

# %% How often does the driver stay in the same zone?
stays = [
    sum(
        r.zones[a] == r.zones[b]
        for a, b in zip(r.actual_order[1:-1], r.actual_order[2:], strict=False)
    )
    / max(len(r.actual_order) - 2, 1)
    for r in routes
]
print(f"Share of moves that stay in the same zone: {pd.Series(stays).mean():.0%}")
