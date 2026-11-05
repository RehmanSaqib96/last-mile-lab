"""Extract a small, fast-to-load sample of Amazon routes.

python scripts/make_sample.py --n 200 --score High
"""

import argparse
from pathlib import Path

from lastmile.amazon.loader import make_sample
from lastmile.config import settings

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--score", default="High", help="High / Medium / Low, or 'any'")
parser.add_argument("--out", default=str(settings.processed_dir / "amazon_sample.json"))
args = parser.parse_args()

ids = make_sample(
    settings.amazon_train_dir,
    Path(args.out),
    n=args.n,
    route_score=None if args.score == "any" else args.score,
)
print(f"Saved {len(ids)} routes to {args.out}")
