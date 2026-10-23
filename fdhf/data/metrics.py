"""Evaluation metrics for PV power forecasting.

All metrics are computed on the physical power scale (MW) after inverse
min-max normalization, matching the manuscript (Section 4.1).

Night-time zero-power periods (solar elevation < 0) are kept in the data but
excluded from the metrics, as required by manuscript Section 4.1 (line 81).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Solar elevation (NOAA approximation) -> used for the night mask
# --------------------------------------------------------------------------- #
def solar_elevation_deg(dt, lat: float, lon: float, tz_offset: float = 8.0) -> np.ndarray:
    """Solar elevation angle (degrees) for an array of (tz-naive local) datetimes.

    Clean NOAA low-precision solar-position approximation. Negative values mean
    the sun is below the horizon (night). Station coordinates from the
    manuscript: 36.644°N, 113.642°E (China, local standard time UTC+8).

    The supplied datetimes are interpreted as LOCAL CLOCK time in a fixed
    timezone `tz_offset` hours east of UTC (default 8.0 for China). We first
    convert to UTC, then to local apparent solar time via the longitude term.
    This is the step the previous version got wrong: it added `lon/15` directly
    to the local clock hour, which double-counts the timezone and produces an
    inverted day/night signal.
    """
    import pandas as pd
    dt = pd.to_datetime(np.asarray(dt).reshape(-1))  # DatetimeIndex (local clock)
    lat = float(lat)
    lon = float(lon)
    tz_offset = float(tz_offset)

    # Day-of-year and the LOCAL clock hour fraction.
    doy = dt.dayofyear.to_numpy().astype(float)
    ns = dt.astype("int64").to_numpy() / 1e9                      # seconds (local)
    frac_day = (ns % 86400) / 86400.0                             # local 0..1
    hour_local = frac_day * 24.0

    # Low-precision solar declination. NOTE: the NOAA polynomial already
    # returns the angle in RADIANS -- do NOT wrap it in np.radians() again
    # (that was the original ~21 deg error: it shrank 23.2 deg to ~0.4 deg).
    gamma = 2 * np.pi / 365.0 * (doy - 1)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    # Equation of time (minutes)
    eqt = (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
           - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)) * 229.18

    # Local apparent solar time (hours): convert local clock -> UTC, then add
    # the longitude term (lon east-positive -> solar time ahead of UTC).
    local_solar_time = (hour_local - tz_offset) + eqt / 60.0 + lon / 15.0
    hour_angle = np.radians(15.0 * (local_solar_time - 12.0))

    lat_r = np.radians(lat)
    sin_elev = (np.sin(lat_r) * np.sin(decl)
                + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_elev))


def night_mask(dt, lat: float, lon: float) -> np.ndarray:
    """Boolean mask: True where solar elevation < 0 (night)."""
    return solar_elevation_deg(dt, lat, lon) < 0.0


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate(y_true, y_pred, night_mask_arr=None, lat=None, lon=None,
             dt=None, clip=None) -> dict:
    """Compute MAE / RMSE / R2.

    Night exclusion (manuscript Section 4.1): if `dt` and station `lat`/`lon`
    are given, or `night_mask_arr` is supplied directly, night points are
    removed before scoring. Optional `clip=(lo, hi)` clamps predictions to a
    physical range (e.g. [0, rated_capacity]) before scoring.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if clip is not None:
        y_pred = np.clip(y_pred, clip[0], clip[1])
    keep = np.ones(len(y_true), dtype=bool)
    if night_mask_arr is not None:
        nm = np.asarray(night_mask_arr).reshape(-1).astype(bool)
        if len(nm) == len(keep):
            keep &= ~nm
    elif (dt is not None) and (lat is not None) and (lon is not None):
        nm = night_mask(dt, lat, lon)
        if len(nm) == len(keep):
            keep &= ~nm
    return {
        "MAE": mae(y_true[keep], y_pred[keep]),
        "RMSE": rmse(y_true[keep], y_pred[keep]),
        "R2": r2(y_true[keep], y_pred[keep]),
    }


def evaluate_with_std(results: list[dict]) -> dict:
    """Aggregate per-seed metric dicts into mean +/- std."""
    maes = [r["MAE"] for r in results]
    rmses = [r["RMSE"] for r in results]
    r2s = [r["R2"] for r in results]
    return {
        "MAE": (float(np.mean(maes)), float(np.std(maes))),
        "RMSE": (float(np.mean(rmses)), float(np.std(rmses))),
        "R2": (float(np.mean(r2s)), float(np.std(r2s))),
    }


def wilcoxon_signed_rank(errors_a: np.ndarray, errors_b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test on point-wise absolute errors.

    The manuscript (Section 4.2) uses it to assess significance of the full
    model over the strongest baseline. Returns statistic and p-value; if the
    scipy dependency is unavailable, p is returned as None.
    """
    ea = np.abs(np.asarray(errors_a).reshape(-1))
    eb = np.abs(np.asarray(errors_b).reshape(-1))
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(ea, eb, zero_method="zsplit")
        return {"statistic": float(stat), "p": float(p)}
    except Exception:
        return {"statistic": float(np.sum(np.sign(ea - eb) * (ea - eb) ** 2)),
                "p": None, "note": "scipy unavailable"}
