"""Module 1: how different is the optimiser from real drivers?

    python scripts/run_baseline.py --limit 50
Writes results/baseline.csv and prints a summary.
"""

import argparse
from pathlib import Path

import pandas as pd

from lastmile.amazon.loader import iter_arrays, load_sample
from lastmile.config import settings
from lastmile.routing.metrics import compare
from lastmile.routing.tsp import solve_tsp

parser = argparse.ArgumentParser()
parser.add_argument("--sample", default=str(settings.processed_dir / "amazon_sample.json"))
parser.add_argument("--limit", type=int, default=50)
parser.add_argument("--seconds", type=int, default=5, help="solver time per route")
args = parser.parse_args()

sample = load_sample(Path(args.sample))
rows = []
for k, r in enumerate(iter_arrays(sample)):
    if k >= args.limit:
        break
    order = solve_tsp(r.travel_s, time_limit_s=args.seconds)
    m = compare(r.actual_order, order, r.travel_s, r.zones)
    rows.append({"route_id": r.route_id, "stops": r.n - 1, **r.meta, **m})
    print(
        f"{k + 1:>3} {r.route_id[:20]:<20} stops={r.n - 1:>3} "
        f"gap={m['time_gap_pct']:5.1f}% tau={m['kendall_tau']:.2f}"
    )

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/baseline.csv", index=False)
print("\nMean over routes:")
print(df[["time_gap_pct", "kendall_tau", "same_next_rate", "zone_kendall_tau"]].mean().round(3))
