"""Generate a synthetic PV SCADA CSV for smoke-testing the FDHF pipeline.

The synthetic signal mimics the key structure of the real dataset: a smooth
daily solar curve (low frequency) plus volatile weather-driven fluctuation
(high frequency), with meteorological covariates that are partially correlated
with power. It is NOT physically accurate — only for verifying that the code
runs end-to-end.

Usage
-----
    python make_dummy_data.py --out data/pv_scada.csv --n 30000
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # 15-min steps -> 4 per hour
    t = np.arange(n) * 0.25  # hours
    day = t / 24.0
    # Solar elevation proxy: positive during daytime, zero at night
    sun = np.clip(np.sin(2 * np.pi * (day % 1) - np.pi / 2), 0, 1)
    # Low-frequency power (smooth daily trend)
    low = 18.0 * sun ** 1.2
    # High-frequency weather-driven fluctuation
    high = 1.2 * np.sin(2 * np.pi * t / 3.0) + rng.normal(0, 0.8, n)
    high *= sun  # fluctuation only during daytime
    power = np.clip(low + high, 0, 20.0)

    ghi = 800 * sun + rng.normal(0, 30, n)
    dni = 700 * sun + rng.normal(0, 40, n)
    dhi = ghi - dni
    temp = 15 + 12 * sun + rng.normal(0, 1.5, n)
    wind_speed = np.abs(rng.normal(3, 1.2, n))
    wind_dir = rng.uniform(0, 360, n)
    pressure = 1013 + rng.normal(0, 3, n)
    df = pd.DataFrame({
        "datetime": pd.date_range("2018-06-30", periods=n, freq="15min"),
        "power": power,
        "GHI": ghi, "DNI": dni, "DHI": dhi,
        "temperature": temp, "wind_speed": wind_speed,
        "wind_direction": wind_dir, "pressure": pressure,
        "local_temp": temp + rng.normal(0, 0.5, n),
        "local_pressure": pressure + rng.normal(0, 1, n),
        "local_wind_speed": wind_speed + rng.normal(0, 0.3, n),
        "local_wind_direction": wind_dir + rng.normal(0, 5, n),
    })
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pv_scada.csv")
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()
    df = generate(args.n)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
