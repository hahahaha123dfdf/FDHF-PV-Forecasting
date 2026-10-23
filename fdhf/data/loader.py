"""Data loading, chronological splitting, per-block cleaning, normalization and
windowing.

CRITICAL (manuscript Section 2, line 21): to avoid leakage the dataset is split
*before* any processing. Each block (train / val / test) is therefore cleaned
independently so that interpolation of a missing value inside the training block
can never use a value from the validation or test block.

    full data  = [ train_period (70%) | test (30%) ]
    train_period = [ actual_train (first 85%) | val (last 15%) ]

All transformation parameters (min/max, decomposition masks) are fitted on the
actual training subset only and then applied to val / test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: str, cfg: dict):
    """Load the SCADA CSV and return (datetime, target, features).

    Returns
    -------
    dt : ndarray (T,)            datetime (or integer index if absent)
    target : ndarray (T,)
    features : ndarray (T, C-1)
    """
    df = pd.read_csv(path)
    dt_col = cfg["data"]["datetime_col"]
    target_col = cfg["data"]["target_col"]
    feat_cols = cfg["data"]["feature_cols"]

    if dt_col in df.columns:
        dt = pd.to_datetime(df[dt_col])
    else:
        dt = pd.Series(pd.date_range("2018-06-30", periods=len(df), freq="15min"))

    target = df[target_col].to_numpy(dtype=float)
    feats = df[feat_cols].to_numpy(dtype=float)
    return dt.to_numpy(), target, feats


def validate_split(dt, capacity_days: float | None = 365.0):
    """Warn (do not crash) if the CSV does not look like the documented 1-year,
    15-min dataset so that a silent row-gap cannot shift the chronological split.

    The manuscript Table 1 expects 35,040 samples spanning 2018-06-30 to
    2019-06-30. A missing 15-min interval would move the 70% boundary off the
    stated date and silently invalidate reproducibility.
    """
    n = len(dt)
    warnings = []
    expected = 35040
    if n != expected:
        warnings.append(
            f"CSV has {n} rows, expected {expected} (1 year @15min). "
            f"The 70/30 chronological split will NOT land on 2019-03-12 as in "
            f"Table 1. Re-sample or gap-fill the data first."
        )
    if np.issubdtype(dt.dtype, np.datetime64):
        span = (dt[-1] - dt[0])
        if span.days < 300:
            warnings.append(
                f"Data span is {span.days} days (< ~365). Scarce-subset day counts "
                f"(39/77/116) quoted in the manuscript will not match."
            )
    for w in warnings:
        import warnings as _w
        _w.warn(w)
    return warnings


# --------------------------------------------------------------------------- #
# Per-block cleaning (operates INSIDE one block only -> no cross-block leakage)
# --------------------------------------------------------------------------- #
def _interp(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy()


def clean_block(series: np.ndarray, capacity_mw: float | None = None, dt=None):
    """Clean a single (C, T) block: remove physically invalid PV points and
    linearly interpolate missing values *within the block only*.

    Channel 0 is the PV target; values < 0 or > capacity are marked invalid.

    If ``dt`` is supplied it must have one entry per time step of ``series``;
    any entirely-NaN time step that is dropped from ``series`` is dropped from
    ``dt`` as well, so the returned datetime stays aligned with the cleaned
    block. Returns ``(series, dt_kept)`` when ``dt`` is given, else ``series``.
    """
    series = series.astype(float).copy()
    if capacity_mw is not None:
        series[0, (series[0] < 0) | (series[0] > capacity_mw)] = np.nan
    # drop columns entirely NaN
    bad = np.all(np.isnan(series), axis=1)
    if bad.any():
        series = series[~bad]
        if dt is not None:
            dt = np.asarray(dt)[~bad]
    for j in range(series.shape[0]):
        series[j] = _interp(series[j])
    if dt is not None:
        return series, np.asarray(dt)
    return series


# --------------------------------------------------------------------------- #
# Chronological split (operates on the raw multivariate series (C, T))
# --------------------------------------------------------------------------- #
def split_series(series: np.ndarray, cfg: dict, dt=None) -> dict:
    """Split a (C, T) multivariate series chronologically.

    Returns dict with keys: actual_train, val, test, scarce{15,30,45}.
    Blocks are RAW (uncleaned) -- cleaning happens per-block afterwards.

    If ``dt`` (a (T,) datetime array) is supplied, the same chronological
    slicing is applied and the aligned datetime blocks are returned under the
    ``"dt"`` key. This is essential for the night mask used by the evaluation
    metrics: the test-window timestamps must be relative to the *test block*,
    not the full series, otherwise solar-elevation<0 is evaluated at the wrong
    instants.
    """
    C, T = series.shape
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio_in_train = cfg["data"]["val_ratio_in_train"]

    n_train = int(round(T * train_ratio))
    train_period = series[:, :n_train]
    test = series[:, n_train:]

    n_val = int(round(n_train * val_ratio_in_train))
    actual_train = train_period[:, : n_train - n_val]
    val = train_period[:, n_train - n_val:]

    out = {"actual_train": actual_train, "val": val, "test": test, "scarce": {}}
    for r in cfg["data"]["scarce_ratios"]:
        n_sub = int(round(n_train * r))
        out["scarce"][r] = train_period[:, :n_sub]

    if dt is not None:
        dt = np.asarray(dt)
        out["dt"] = {
            "actual_train": dt[: n_train - n_val],
            "val": dt[n_train - n_val: n_train],
            "test": dt[n_train:],
            "scarce": {r: dt[: int(round(n_train * r))]
                       for r in cfg["data"]["scarce_ratios"]},
        }
    return out


# --------------------------------------------------------------------------- #
# Normalization (fit on actual training subset only; per-channel for 2-D input)
# --------------------------------------------------------------------------- #
class MinMaxNormalizer(BaseEstimator, TransformerMixin):
    """Min/max normalizer.

    For a 1-D array it stores scalar min/max. For a 2-D array of shape (C, T)
    it stores a *per-channel* min/max so that variables on different scales
    (e.g. irradiance in W/m^2 vs pressure in hPa) are normalised independently,
    matching Equation (10) which is applied per sequence.
    """

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.ndim <= 1:
            x = x.reshape(1, -1)
        self.min_ = np.nanmin(x, axis=1)          # (C,)
        self.max_ = np.nanmax(x, axis=1)          # (C,)
        tiny = (self.max_ - self.min_) == 0
        self.max_[tiny] = self.min_[tiny] + 1e-8
        self.n_channels_ = x.shape[0]
        return self

    def _reshape(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return x.reshape(1, -1), True
        return x, False

    def transform(self, x: np.ndarray) -> np.ndarray:
        x, was_1d = self._reshape(x)
        rng = (self.max_ - self.min_).reshape(-1, 1)
        mn = self.min_.reshape(-1, 1)
        out = (x - mn) / rng
        if x.shape[0] == 1 and (was_1d or x.shape[0] == self.n_channels_):
            if was_1d:
                return out.reshape(-1)
        return out

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x, was_1d = self._reshape(x)
        rng = (self.max_ - self.min_).reshape(-1, 1)
        mn = self.min_.reshape(-1, 1)
        out = x * rng + mn
        if was_1d:
            return out.reshape(-1)
        return out


# --------------------------------------------------------------------------- #
# Windowing (sliding window over a 1-D target + optional covariates)
# --------------------------------------------------------------------------- #
def build_windows(target: np.ndarray, covariates: np.ndarray | None,
                  input_len: int, pred_len: int):
    """Sliding-window dataset.

    target : (T,)
    covariates : (C, T) or None
    Returns X : (N, input_len, 1+C), y : (N, pred_len)
    """
    T = target.shape[0]
    if covariates is not None:
        feat = np.concatenate([target[None, :], covariates], axis=0)   # (1+C, T)
    else:
        feat = target[None, :]
    F = feat.shape[0]
    Xs, ys = [], []
    for i in range(0, T - input_len - pred_len + 1):
        Xs.append(feat[:, i:i + input_len].T)        # (input_len, F)
        ys.append(target[i + input_len:i + input_len + pred_len])
    X = np.stack(Xs)        # (N, input_len, F)
    y = np.stack(ys)        # (N, pred_len)
    return X, y
