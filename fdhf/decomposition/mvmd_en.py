"""MVMD-EN: Multivariate VMD + Elastic-Net variable selection + sample-entropy
frequency separation.

This module wraps the vendored MVMD implementation (fdhf/decomposition/mvmd.py) and adds the two
supervised/structural steps described in the manuscript (Section 3.1 and the
"sample-entropy-based frequency separation" step in Section 2):

1. **Elastic Net (EN) screening** -- fits ``ElasticNet`` on the high-dimensional
   IMF-channel matrix (target = PV power) and keeps only IMFs whose coefficient
   is non-zero, dropping redundant / noise-dominated modes.
2. **Sample-entropy frequency separation** -- each *selected* PV-channel IMF is
   assigned to the low-frequency branch (smooth, SE < S) or the high-frequency
   branch (volatile, SE >= S). The two branches are reconstructed by summation.

All decomposition parameters (K, alpha, tolerance, init) are fit on the actual
training subset only; the EN mask and the low/high IMF assignment are frozen and
then applied to validation / test periods.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet

from .mvmd import mvmd
from .sample_entropy import sample_entropy
from .emd import emd


class MVMDEN:
    def __init__(
        self,
        K: int = 6,
        alpha: float = 2000.0,
        tolerance: float = 1e-3,
        init: int = 1,
        max_iter: int = 500,
        en_alpha: float = 0.01,
        l1_ratio: float = 0.5,
        se_threshold: float = 0.13,
        se_m: int = 2,
        se_r_factor: float = 0.2,
        target_channel: int = 0,
        multivariate: bool = True,
    ):
        self.K = K
        self.alpha = alpha
        self.tolerance = tolerance
        self.init = init
        self.max_iter = max_iter
        self.en_alpha = en_alpha
        self.l1_ratio = l1_ratio
        self.se_threshold = se_threshold
        self.se_m = se_m
        self.se_r_factor = se_r_factor
        self.target_channel = target_channel
        self.multivariate = multivariate

        # Fitted state
        self.selected_mask_: np.ndarray | None = None   # (K, C) bool
        self.low_imf_mask_: np.ndarray | None = None    # (K,) bool for PV channel
        self.n_channels_: int | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, signal_train: np.ndarray):
        """Fit EN screening on the actual training subset.

        Parameters
        ----------
        signal_train : ndarray (C, T)
            Multivariate training signal (channel 0 = PV target).
        """
        signal_train = np.asarray(signal_train, dtype=float)
        C, T = signal_train.shape
        self.n_channels_ = C

        # 1) Decompose training signal
        modes, _, _ = mvmd(
            signal_train, num_modes=self.K, alpha=self.alpha,
            init=self.init, tolerance=self.tolerance, max_iter=self.max_iter,
        )  # modes: (K, C, T)

        # 2) EN feature selection: every (IMF, channel) column is a candidate
        #    feature; target is the raw PV power.
        X = modes.reshape(self.K * C, T).T           # (T, K*C)
        y = signal_train[self.target_channel]         # (T,)
        en = ElasticNet(
            alpha=self.en_alpha, l1_ratio=self.l1_ratio,
            max_iter=5000, tol=1e-4, random_state=0,
        ).fit(X, y)
        coef = en.coef_.reshape(self.K, C)
        self.selected_mask_ = coef != 0

        # 3) Frequency separation on the selected PV-channel IMFs
        pv_imfs = modes[self.selected_mask_[:, self.target_channel], self.target_channel, :]
        # pv_imfs may be 1-D if only one IMF selected; normalise to 2-D
        pv_imfs = np.atleast_2d(pv_imfs)
        if pv_imfs.shape[0] == 0:
            raise RuntimeError("EN removed all PV IMFs; relax en_alpha / l1_ratio.")
        se = np.array([
            sample_entropy(imf, m=self.se_m, r=self.se_r_factor * np.std(imf))
            for imf in pv_imfs
        ])
        low_local = se < self.se_threshold
        # Map back to the (K,) PV-channel mask
        self.low_imf_mask_ = np.zeros(self.K, dtype=bool)
        sel_idx = np.where(self.selected_mask_[:, self.target_channel])[0]
        self.low_imf_mask_[sel_idx] = low_local
        # Guard: ensure the low-frequency branch is never empty. If every
        # selected PV IMF is classified high-frequency, assign the single
        # smoothest (lowest-entropy) one to the low branch.
        if self.low_imf_mask_.sum() == 0 and len(sel_idx) > 0:
            self.low_imf_mask_[sel_idx[np.argmin(se)]] = True

        return self

    # ------------------------------------------------------------------ #
    # Transformation
    # ------------------------------------------------------------------ #
    def _decompose(self, signal: np.ndarray):
        signal = np.asarray(signal, dtype=float)
        if self.multivariate:
            return mvmd(signal, num_modes=self.K, alpha=self.alpha,
                        init=self.init, tolerance=self.tolerance,
                        max_iter=self.max_iter)[0]  # (K, C, T)
        # Ablation 2c: decompose each channel independently (single-channel VMD)
        C = signal.shape[0]
        per_channel = [
            mvmd(signal[c:c + 1], num_modes=self.K, alpha=self.alpha,
                  init=self.init, tolerance=self.tolerance,
                  max_iter=self.max_iter)[0]   # (K, 1, T)
            for c in range(C)
        ]
        return np.concatenate(per_channel, axis=1)               # (K, C, T)

    def decompose_series(self, signal: np.ndarray):
        """Decompose a *full* series (used for the training period only).

        Returns
        -------
        low_freq : (T,) PV low-frequency component
        high_freq : (T,) PV high-frequency component
        covariates : (C_cov, T) selected meteorological IMFs (channels != target)
        """
        if self.selected_mask_ is None:
            raise RuntimeError("Call fit() before transform.")
        signal = np.asarray(signal, dtype=float)
        modes = self._decompose(signal)              # (K, C, T)
        low, high = self._reconstruct(modes)
        cov = self._covariates(modes)
        return low, high, cov

    def decompose_window(self, history: np.ndarray):
        """Causal, window-by-window decomposition for validation / test.

        Parameters
        ----------
        history : (C, L) available historical window (L = input_len).

        Returns
        -------
        low_freq : (L,) high-frequency-excluded PV target window
        high_freq : (L,) low-frequency-excluded PV target window
        covariates : (C_cov, L)
        """
        modes = self._decompose(history)
        low, high = self._reconstruct(modes)
        cov = self._covariates(modes)
        return low, high, cov

    # ------------------------------------------------------------------ #
    # Case 1a: no MVMD-EN. Frequency separation on the RAW power signal by
    # sample entropy only (manuscript Case 1a, line 340). EMD supplies the
    # intrinsic modes; SampEn decides low vs high. No EN screening.
    # ------------------------------------------------------------------ #
    def separate_raw_window(self, pv_window: np.ndarray):
        imfs = emd(np.asarray(pv_window, dtype=float), max_imfs=self.K)
        # NOTE: emd() returns the intrinsic modes AND the final monotone residue
        # as the last element. We keep it: (a) the full set of IMFs + residue
        # reconstructs pv *exactly*, so low + high == pv; (b) the residue is the
        # lowest-frequency trend and its sample entropy is ~0, so it is naturally
        # routed to the low-frequency branch. Dropping it (the previous behaviour)
        # broke reconstruction by the residue's energy.
        if len(imfs) == 0:
            return pv_window.copy(), np.zeros_like(pv_window)
        se = np.array([
            sample_entropy(imf, m=self.se_m, r=self.se_r_factor * np.std(imf))
            for imf in imfs
        ])
        low = np.zeros_like(pv_window)
        high = np.zeros_like(pv_window)
        for imf, s in zip(imfs, se):
            if s < self.se_threshold:
                low += imf
            else:
                high += imf
        # Guard: if every mode (incl. residue) classified high, keep the
        # smoothest one in the low branch and re-derive high by subtraction so
        # the reconstruction stays exact.
        if np.allclose(low, 0):
            low = imfs[int(np.argmin(se))]
            high = pv_window - low
        return low, high

    def separate_raw_series(self, pv_series: np.ndarray):
        """Whole-series raw separation (used only to build ground-truth targets
        for evaluation; never fed to the model)."""
        return self.separate_raw_window(np.asarray(pv_series, dtype=float))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reconstruct(self, modes: np.ndarray):
        pv_imfs = modes[:, self.target_channel, :]        # (K, T)
        low = pv_imfs[self.low_imf_mask_].sum(axis=0)
        high = pv_imfs[~self.low_imf_mask_].sum(axis=0)
        return low, high

    def _covariates(self, modes: np.ndarray):
        # Selected meteorological IMF channels (exclude target channel)
        sel = self.selected_mask_.copy()
        sel[:, self.target_channel] = False
        if not sel.any():
            return np.empty((0, modes.shape[-1]))
        return modes[sel].reshape(-1, modes.shape[-1])
