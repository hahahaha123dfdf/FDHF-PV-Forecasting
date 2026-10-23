"""Statistical baselines: Persistence and Seasonal-naive (manuscript Table 7)."""

from __future__ import annotations

import numpy as np

from .base import BaseForecaster


class Persistence(BaseForecaster):
    """Persistence: forecast the last observed value for all future steps."""

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (N, input_len) -> repeat last value pred_len times
        last = X[:, -1:]
        return np.repeat(last, self.pred_len, axis=1)

    def __init__(self, pred_len: int = 24):
        self.pred_len = pred_len


class SeasonalNaive(BaseForecaster):
    """Seasonal-naive: use the value from exactly one day (96 steps) earlier.

    Falls back to persistence for the warm-up region where no full day of
    history exists.
    """

    def __init__(self, input_len: int = 96, pred_len: int = 24, season: int = 96):
        self.input_len = input_len
        self.pred_len = pred_len
        self.season = season

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.pred_len))
        for i in range(X.shape[0]):
            # last `season` steps of the input window shifted back by `season`
            seg = X[i]
            if seg.shape[0] >= self.season + self.pred_len:
                out[i] = seg[-(self.season + self.pred_len):-self.season]
            else:
                out[i] = seg[-1]
        return out
