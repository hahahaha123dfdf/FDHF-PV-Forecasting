"""Hyper-parameter selection on the validation set only.

The manuscript states that K, alpha, the Elastic-Net parameters and the
sample-entropy threshold S are all *selected on the validation set* (Sections
2 and 3.1) and that Table 3 reports the test-time sensitivity of S. This module
provides a lightweight harness that scans S (reproducing Table 3) and the EN
grid, using fast ridge surrogates on the decomposed validation windows so the
search is cheap. The selected configuration is then used to (re)build the full
pipeline.

NOTE: this is a *harness*, not the full expensive re-training. It measures how S
and the EN penalty affect the downstream forecastability of the separated
components via ridge surrogates, which is sufficient for threshold selection.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .decomposition.mvmd_en import MVMDEN


def _surrogate_val_rmse(mvmd_en: MVMDEN, val_block: np.ndarray,
                        input_len: int, pred_len: int):
    """Train ridge surrogates on the (causal) val components and return RMSE of
    the summed prediction vs true PV. Cheap proxy for downstream error."""
    low, high, cov = mvmd_en.decompose_series(val_block)
    pv = val_block[0]
    # whole-block windows (val proxies; good enough for threshold ranking)
    from .data.loader import build_windows
    Xl, yl = build_windows(low, None, input_len, pred_len)
    Xh, yh = build_windows(high, (cov if cov.size else None), input_len, pred_len)
    ypv = build_windows(pv, None, input_len, pred_len)[1]
    Xl = Xl.reshape(Xl.shape[0], -1); yl = yl.reshape(yl.shape[0], -1)
    Xh = Xh.reshape(Xh.shape[0], -1); yh = yh.reshape(yh.shape[0], -1)

    rl = Ridge(alpha=1.0).fit(Xl, yl); rh = Ridge(alpha=1.0).fit(Xh, yh)
    pred = rl.predict(Xl) + rh.predict(Xh)
    return float(np.sqrt(np.mean((ypv - pred) ** 2)))


def scan_sample_entropy_threshold(cfg: dict, train_signal: np.ndarray,
                                  val_block: np.ndarray,
                                  S_grid=None) -> dict:
    """Reproduce Table 3: scan S, select on validation. Returns {S: val_rmse}."""
    if S_grid is None:
        S_grid = [0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20]
    d = cfg["decomposition"]
    out = {}
    for S in S_grid:
        mv = MVMDEN(
            K=d["mvmd"]["K"], alpha=d["mvmd"]["alpha"], tolerance=d["mvmd"]["tolerance"],
            init=d["mvmd"]["init"], max_iter=d["mvmd"]["max_iter"],
            en_alpha=d["elastic_net"]["alpha"], l1_ratio=d["elastic_net"]["l1_ratio"],
            se_threshold=S, se_m=d["freq_separation"]["m"],
            se_r_factor=d["freq_separation"]["r_factor"],
        )
        mv.fit(train_signal)
        out[S] = _surrogate_val_rmse(mv, val_block, cfg["data"]["input_len"],
                                     cfg["data"]["pred_len"])
    return out


def grid_search_elastic_net(cfg: dict, train_signal: np.ndarray, val_block: np.ndarray):
    """Scan l1_ratio x alpha_EN grid (manuscript Section 3.1, line 36)."""
    d = cfg["decomposition"]
    l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    a_grid = [1e-3, 1e-2, 1e-1, 1.0]
    out = {}
    for l1 in l1_grid:
        for a in a_grid:
            mv = MVMDEN(
                K=d["mvmd"]["K"], alpha=d["mvmd"]["alpha"], tolerance=d["mvmd"]["tolerance"],
                init=d["mvmd"]["init"], max_iter=d["mvmd"]["max_iter"],
                en_alpha=a, l1_ratio=l1,
                se_threshold=d["freq_separation"]["sample_entropy_threshold"],
                se_m=d["freq_separation"]["m"], se_r_factor=d["freq_separation"]["r_factor"],
            )
            mv.fit(train_signal)
            out[(l1, a)] = _surrogate_val_rmse(mv, val_block, cfg["data"]["input_len"],
                                              cfg["data"]["pred_len"])
    return out
