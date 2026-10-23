"""Empirical Mode Decomposition (EMD).

Used by the FDHF ablation Case 1a ("MVMD-EN removed; frequency separation on
the raw power signal by sample entropy only"). Implements cubic-spline sifting
to split a signal into a small number of intrinsic mode functions (IMFs).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _is_imf(c: np.ndarray, r: np.ndarray, tol: float) -> bool:
    # Stop when the sifting residual is below a relative tolerance.
    return np.mean(np.abs(r)) < tol * np.mean(np.abs(c)) or np.allclose(c, r, atol=1e-8)


def emd(x: np.ndarray, max_imfs: int = 6, max_sift: int = 50,
        tol: float = 0.05) -> list[np.ndarray]:
    """Return a list of IMFs (each length == len(x)) plus the final residue.

    The last returned array is the monotone residue; callers may drop it.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    t = np.arange(n)
    residue = x.copy()
    imfs = []
    while len(imfs) < max_imfs and np.std(residue) > 1e-10:
        h = residue.copy()
        for _ in range(max_sift):
            # upper / lower envelopes via cubic splines
            # local maxima / minima
            up = (np.diff(np.sign(np.diff(h))) < 0).nonzero()[0] + 1
            down = (np.diff(np.sign(np.diff(h))) > 0).nonzero()[0] + 1
            if len(up) < 2 or len(down) < 2:
                break
            env_up = CubicSpline(t[up], h[up])(t)
            env_down = CubicSpline(t[down], h[down])(t)
            mean_env = 0.5 * (env_up + env_down)
            h_new = h - mean_env
            if _is_imf(h, h_new, tol):
                h = h_new
                break
            h = h_new
        imfs.append(h.copy())
        residue = residue - h
    imfs.append(residue.copy())
    return imfs
