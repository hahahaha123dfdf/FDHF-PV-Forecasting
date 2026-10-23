"""Baseline forecasters referenced by the manuscript comparison tables.

The manuscript reports a large comparison suite (Tables 2, 4, 6, 7, 9, 10):
  - Table 2 : decomposition strategies (VMD+EN, CEEMDAN+EN, per-channel VMD+EN,
              MVMD+EN)  -> see vmd_lstm / decomposition wrappers
  - Table 4 : Persistence, CNN, LSTM, GRU, DLinear, Transformer, Informer,
              VMD-LSTM
  - Table 6 : TimeGPT, GPT4TS  (PROPRIETARY API -> not reproducible offline; see api.py)
  - Table 7 : linear extrapolation, seasonal-naive, ARIMA
  - Table 9 : distributional metrics (MMD / W1 / Pearson / NN-dist) + DCGAN
  - Table 10: jittering, bootstrapping, DCGAN, no-augmentation

Offline-feasible models are implemented here. TimeGPT / GPT4TS require paid APIs
and are provided only as explicit stubs that raise (never silently fake results).
"""

from __future__ import annotations

import numpy as np


class BaseForecaster:
    """Common interface: fit on (X, y) arrays, predict returns (N, pred_len)."""

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def make_pv_windows(pv: np.ndarray, input_len: int, pred_len: int):
    """Sliding windows over a 1-D PV series -> (X, y) for direct forecasters."""
    pv = np.asarray(pv, dtype=float)
    T = pv.shape[0]
    Xs, ys = [], []
    for i in range(0, T - input_len - pred_len + 1):
        Xs.append(pv[i:i + input_len])
        ys.append(pv[i + input_len:i + input_len + pred_len])
    return np.stack(Xs), np.stack(ys)
