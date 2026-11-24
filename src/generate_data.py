from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "distance_km",
    "prep_time_p50_min",
    "courier_load",
    "zone_order_rate",
    "hour_of_day",
    "day_of_week",
    "is_rain",
    "traffic_index",
    "time_since_accept_min",
]

TARGET_COLUMN = "actual_delivery_min"


def generate_deliveries(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance_km = rng.uniform(0.8, 12.0, n_samples)
    prep_time_p50_min = rng.uniform(8.0, 35.0, n_samples)
    courier_load = rng.integers(0, 4, n_samples).astype(float)
    zone_order_rate = rng.uniform(5.0, 80.0, n_samples)
    hour_of_day = rng.integers(8, 23, n_samples).astype(float)
    day_of_week = rng.integers(0, 7, n_samples).astype(float)
    is_rain = rng.binomial(1, 0.22, n_samples).astype(float)
    traffic_index = rng.uniform(0.4, 1.8, n_samples)
    time_since_accept_min = rng.uniform(0.0, 12.0, n_samples)

    peak = ((hour_of_day >= 12) & (hour_of_day <= 14)) | (
        (hour_of_day >= 19) & (hour_of_day <= 21)
    )
    travel = distance_km * (3.8 + 1.6 * traffic_index) * (1.0 + 0.18 * is_rain)
    dispatch_wait = 2.5 + 1.8 * courier_load + 0.04 * zone_order_rate + 1.5 * peak
    prep = prep_time_p50_min * (1.0 + 0.12 * is_rain) + 0.15 * time_since_accept_min
    weekend = (day_of_week >= 5).astype(float) * 1.2
    noise = rng.normal(0.0, 2.4, n_samples)

    actual = prep + dispatch_wait + travel + weekend + noise
    actual = np.clip(actual, 12.0, 95.0)

    frame = pd.DataFrame(
        {
            "distance_km": distance_km,
            "prep_time_p50_min": prep_time_p50_min,
            "courier_load": courier_load,
            "zone_order_rate": zone_order_rate,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_rain": is_rain,
            "traffic_index": traffic_index,
            "time_since_accept_min": time_since_accept_min,
            TARGET_COLUMN: actual,
        }
    )
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_deliveries()
    path = out_dir / "deliveries.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    print(df[[TARGET_COLUMN]].describe().to_string())


if __name__ == "__main__":
    main()
