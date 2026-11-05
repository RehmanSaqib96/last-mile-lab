"""Module 2: does learning from drivers make sequences more driver-like?

    python scripts/run_behaviour.py --train 150 --test 50
Compares three planners on held-out routes:
  shortest   - TSP on travel time (Module 1)
  zones      - TSP on travel time + tuned zone-change penalties
  learned    - TSP on -log P(next stop) from a classifier trained on real moves
"""

import argparse
from pathlib import Path

import pandas as pd

from lastmile.amazon.loader import iter_arrays, load_sample
from lastmile.behaviour.next_stop import learned_cost, train
from lastmile.behaviour.zones import tune_penalties, zone_cost
from lastmile.config import settings
from lastmile.routing.metrics import compare
from lastmile.routing.tsp import solve_tsp

parser = argparse.ArgumentParser()
parser.add_argument("--sample", default=str(settings.processed_dir / "amazon_sample.json"))
parser.add_argument("--train", type=int, default=150)
parser.add_argument("--test", type=int, default=50)
parser.add_argument("--seconds", type=int, default=3)
args = parser.parse_args()

routes = list(iter_arrays(load_sample(Path(args.sample))))
train_r, test_r = routes[: args.train], routes[args.train : args.train + args.test]
print(f"train={len(train_r)} test={len(test_r)}")

(zp, sp), _ = tune_penalties(train_r[:30])  # grid search is slow: tune on a subset
print(f"best zone penalty={zp}s, super-zone penalty={sp}s")
model = train(train_r)

rows = []
for r in test_r:
    plans = {
        "shortest": solve_tsp(r.travel_s, time_limit_s=args.seconds),
        "zones": solve_tsp(zone_cost(r.travel_s, r.zones, zp, sp), time_limit_s=args.seconds),
        "learned": solve_tsp(learned_cost(model, r), time_limit_s=args.seconds),
    }
    for name, order in plans.items():
        rows.append(
            {
                "route_id": r.route_id,
                "planner": name,
                **compare(r.actual_order, order, r.travel_s, r.zones),
            }
        )

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/behaviour.csv", index=False)
print(
    df.groupby("planner")[["kendall_tau", "same_next_rate", "zone_kendall_tau", "time_gap_pct"]]
    .mean()
    .round(3)
)
