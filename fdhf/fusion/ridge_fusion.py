"""Ridge-regression fusion of the low- and high-frequency branches.

The final forecast is a weighted combination of the Time-LLM (low-frequency)
and XGBoost (high-frequency) predictions learned by a ridge regressor
(manuscript Section 3 / Conclusion).

Fusion weights are NEVER fit on the test set. When ``out_of_fold`` is True the
ridge alpha is selected by 5-fold cross-validation on the provided branch
predictions (the manuscript's "out-of-fold" requirement, Section 4.2); when
False the fixed alpha is used and the regressor is fit directly on the
validation-set branch predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


class RidgeFusion:
    def __init__(self, cfg: dict):
        f = cfg["fusion"]
        self.alpha = f["alpha"]
        self.out_of_fold = f.get("out_of_fold", False)
        self.model = None

    def fit(self, low_pred: np.ndarray, high_pred: np.ndarray, y_true: np.ndarray):
        # low_pred, high_pred, y_true: (N, pred_len)
        X = np.stack([low_pred, high_pred], axis=-1).reshape(len(low_pred), -1)
        Y = y_true.reshape(len(y_true), -1)
        if self.out_of_fold:
            # 5-fold CV over the provided branch predictions; the fixed alpha is
            # one of the candidate regularisation strengths.
            self.model = RidgeCV(alphas=[self.alpha, self.alpha / 10,
                                        self.alpha * 10, 0.1, 1.0, 10.0], cv=5)
        else:
            self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, Y)
        return self

    def predict(self, low_pred: np.ndarray, high_pred: np.ndarray) -> np.ndarray:
        X = np.stack([low_pred, high_pred], axis=-1).reshape(len(low_pred), -1)
        return self.model.predict(X).reshape(low_pred.shape[0], -1)
