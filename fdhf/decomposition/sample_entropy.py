"""Sample Entropy (SampEn) implementation.

Sample entropy measures the complexity/irregularity of a time series and is
used in the FDHF framework to separate low-frequency (smooth, low entropy)
from high-frequency (volatile, high entropy) intrinsic mode components.

Reference: Richman & Moorman (2000), "Physiological time-series analysis
using approximate entropy and sample entropy".
"""

from __future__ import annotations

import numpy as np


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Compute sample entropy of a 1-D signal.

    Parameters
    ----------
    x : 1-D array
        Input time series.
    m : int
        Embedding dimension.
    r : float, optional
        Tolerance (typically 0.1-0.25 * std(x)). If None, r = 0.2 * std(x).

    Returns
    -------
    float
        Sample entropy. When no matches are found (B == 0) the exact SampEn is
        -log(0) = +inf, i.e. maximal irregularity, so we return a large finite
        value. This is essential for the frequency separation: a no-match IMF is
    *high* frequency and must NOT be forced into the low-frequency (Time-LLM)
    branch.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.shape[0]
    if n < m + 2:
        return 0.0

    if r is None:
        r = 0.2 * np.std(x)
    if r == 0:
        r = 1e-8

    # Build m- and (m+1)-dimensional embedded templates (phase-space vectors)
    def _count_matches(dim: int) -> int:
        templates = np.array([x[i:i + dim] for i in range(n - dim + 1)])
        count = 0
        for i in range(len(templates)):
            # Chebyshev distance between template i and all others
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist <= r) - 1  # exclude self-match
        return count

    a = _count_matches(m)      # matches for dimension m
    b = _count_matches(m + 1)  # matches for dimension m+1

    if a == 0 or b == 0:
        # No match -> SampEn diverges (maximal complexity) -> large finite value.
        return 1e6
    return float(-np.log(b / a))
