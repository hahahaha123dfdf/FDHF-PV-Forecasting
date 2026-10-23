"""XGBoost predictor for the high-frequency PV component.

XGBoost models the highly non-linear, noisy high-frequency residuals
(manuscript Section 3.3). Windows are flattened and (optionally) concatenated
with exogenous covariate windows before being fed to the gradient-boosted
tree ensemble.
"""

from __future__ import annotations

import time

import numpy as np
import xgboost as xgb


class XGBoostForecaster:
    def __init__(self, cfg: dict, seed: int = 0):
        x = cfg["high_freq_branch"]["xgboost"]
        self.params = dict(
            n_estimators=x["n_estimators"],
            max_depth=x["max_depth"],
            learning_rate=x["learning_rate"],
            subsample=x["subsample"],
            colsample_bytree=x["colsample_bytree"],
            n_jobs=x["n_jobs"],
            objective="reg:squarederror",
            random_state=seed,
        )
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X: (N, input_len[, + cov]) flattened; y: (N, pred_len)
        t0 = time.perf_counter()
        Xf = X.reshape(X.shape[0], -1)
        yf = y.reshape(y.shape[0], -1)
        self.model.fit(Xf, yf)
        self.last_train_seconds = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        Xf = X.reshape(X.shape[0], -1)
        out = self.model.predict(Xf).reshape(X.shape[0], -1)
        self.last_inference_seconds = time.perf_counter() - t0
        self.last_inference_samples = X.shape[0]
        return out

    # ------------------------------------------------------------------ #
    def complexity(self) -> dict:
        """Tree-ensemble size (the tree analogue of a parameter count)."""
        booster = self.model.get_booster()
        try:
            trees = booster.num_boosted_rounds()
        except Exception:
            trees = int(self.params["n_estimators"])
        n_leaves = 0
        try:
            for feat in booster.get_dump(with_stats=False):
                n_leaves += feat.count("leaf=")
        except Exception:
            pass
        return {"n_trees": int(trees), "n_leaves": int(n_leaves),
                "max_depth": int(self.params["max_depth"])}
