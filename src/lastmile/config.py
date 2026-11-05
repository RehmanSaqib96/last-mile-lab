"""Central settings. Values come from environment variables (or .env) with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env reader so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("LASTMILE_DATA_DIR", "data"))
    amazon_train_dir: Path = Path(
        os.getenv("AMAZON_TRAIN_DIR", "data/raw/almrrc2021/almrrc2021-data-training")
    )
    # Made-up depot near M1 junction 10, Luton. Change it to any point you like.
    depot_lat: float = float(os.getenv("DEPOT_LAT", "51.8750"))
    depot_lon: float = float(os.getenv("DEPOT_LON", "-0.4150"))
    carbon_postcode: str = os.getenv("CARBON_POSTCODE", "LU1")

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


settings = Settings()
