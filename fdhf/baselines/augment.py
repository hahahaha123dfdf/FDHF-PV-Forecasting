"""Data-augmentation baselines for the high-frequency component
(manuscript Table 10): jittering and bootstrapping.

Both produce *consistent* (input, target) pairs -- a jittered/bootstrapped
window keeps its own true target -- unlike the previous WGAN-GP bug.
"""

from __future__ import annotations

import numpy as np


def jittering(X: np.ndarray, y: np.ndarray, n_extra: int, noise_std: float = 0.02):
    """Add small Gaussian noise to real windows; keep their true labels."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    idx = np.random.choice(n, size=n_extra, replace=True)
    noise = np.random.normal(0, noise_std, size=X[idx].shape)
    Xj = X[idx] + noise
    return Xj, y[idx]


def bootstrapping(X: np.ndarray, y: np.ndarray, n_extra: int):
    """Bootstrap-resample real windows (with their true labels)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    idx = np.random.choice(n, size=n_extra, replace=True)
    return X[idx], y[idx]
