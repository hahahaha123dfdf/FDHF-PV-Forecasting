"""VMD-LSTM baseline (manuscript Table 4) and MVMD/EN decomposition wrappers
used by Table 2 Strategy 3 (per-channel VMD + EN) and Strategy 4 (MVMD + EN).

torch is imported lazily. The decomposition itself reuses the vendored MVMD.
"""

from __future__ import annotations

import numpy as np

from .base import BaseForecaster
from ..decomposition.mvmd import mvmd


def vmd_decompose(signal_1d: np.ndarray, K: int = 6, alpha: float = 2000.0):
    """Single-channel VMD -> sum of IMFs (used for the per-channel VMD baseline)."""
    sig = np.asarray(signal_1d, dtype=float)[None, :]   # (1, T)
    modes, _, _ = mvmd(sig, num_modes=K, alpha=alpha)
    return modes[:, 0, :]                               # (K, T)


class VMDLSTM(BaseForecaster):
    """Decompose PV with VMD, sum IMFs, then forecast with LSTM (Table 4)."""

    def __init__(self, K: int = 6, alpha: float = 2000.0, **lstm_kw):
        self.K = K
        self.alpha = alpha
        self.lstm_kw = lstm_kw
        self.lstm = None

    def fit(self, X, y):
        from .ml import make_lstm
        self.lstm = make_lstm(**self.lstm_kw)
        self.lstm.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lstm.predict(X)
