"""FDHF end-to-end pipeline.

Frequency-Domain Decoupled Hybrid Forecasting framework (manuscript Sections 2 & 3).

Leakage protocol (manuscript Section 2, line 21) -- implemented here:
  1. Load raw data.
  2. Chronological split FIRST (train_period | test), then (actual_train | val).
  3. Clean each block INDEPENDENTLY (outlier removal + interpolation within the
     block only) so training interpolation can never use val/test values.
  4. MVMD-EN is fit on the actual training subset only; the EN mask and the
     low/high assignment are frozen.
  5. Training windows use whole-block decomposition (all within train -> no leak).
  6. Validation / test INPUTS use causal, window-by-window decomposition
     (decompose_window / separate_raw_window) using only the 96-step history.
     The corresponding targets are taken from a whole-block decomposition of the
     block and used ONLY as ground-truth labels for evaluation -- never fed to
     the model.
  7. min/max normalisation is fit on the training subset and applied per-channel.
  8. Fusion weights are fit on val (or via 5-fold OOF on the branch predictions),
     never on test. Night-time zero-power points (solar elevation < 0) are kept in
     the data but excluded from the metrics.

Ablation presets from Experiment 4 (Table 8) are supported through `ablation`.
"""

from __future__ import annotations

import numpy as np

from .data.loader import (
    MinMaxNormalizer, build_windows, clean_block, load_raw, split_series,
    validate_split,
)
from .data.metrics import evaluate, night_mask
from .decomposition.mvmd_en import MVMDEN
from .profiling import (
    Stopwatch, count_parameters, device_name, peak_memory, reset_peak_memory,
)


# --------------------------------------------------------------------------- #
# Ablation presets (Table 8)
# --------------------------------------------------------------------------- #
ABLATIONS = {
    None:   dict(use_mvmd_en=True, use_en=True, use_pap=True,
                 use_wgan_gp=True, multivariate=True, ridge=True, backbone="gpt2"),
    "1a":   dict(use_mvmd_en=False),                        # w/o MVMD-EN
    "1b":   dict(use_en=False),                             # w/o EN only
    "2a":   dict(use_pap=False),                            # w/o PaP
    "2b":   dict(use_wgan_gp=False),                        # w/o WGAN-GP
    "2c":   dict(multivariate=False),                       # MVMD -> VMD
    "3":    dict(ridge=False),                              # w/o ridge (sum)
    "4":    dict(backbone="bert"),                          # GPT-2 -> BERT
}


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


class FDHFPipeline:
    def __init__(self, cfg: dict, experiment: str = "full", ablation: str | None = None):
        self.cfg = cfg
        self.experiment = experiment
        self.ab = {**ABLATIONS[None], **ABLATIONS.get(ablation, {})}
        self.input_len = cfg["data"]["input_len"]
        self.pred_len = cfg["data"]["pred_len"]
        self.lat = cfg["data"].get("station_lat", 36.644)
        self.lon = cfg["data"].get("station_lon", 113.642)
        self.capacity = cfg["data"].get("capacity_mw", 20.0)
        # normalizers (fit on training subset)
        self.norm_low = self.norm_high = self.norm_cov = self.norm_pv = None
        # last-run test predictions (for statistics / Wilcoxon)
        self.test_y_true = self.test_y_pred = self.test_dt = None

    # ------------------------------------------------------------------ #
    def run(self, path: str, capacity_mw: float | None = None, seed: int = 0):
        set_seed(seed)
        if capacity_mw is None:
            capacity_mw = self.capacity

        sw = Stopwatch()
        reset_peak_memory()

        # 1) Load raw
        with sw.time("load"):
            dt, target, features = load_raw(path, self.cfg)
        validate_split(dt)
        series = np.vstack([target[None, :], features.T])     # (C, T), ch0 = PV

        # 2) Chronological split (raw blocks) -- keep datetimes aligned so the
        #    night mask in the metrics is evaluated at the correct instants.
        with sw.time("split"):
            splits = split_series(series, self.cfg, dt)

        # 3) Per-block cleaning (no cross-block interpolation leakage)
        train_sub = self._training_subset(splits)
        dt_blocks = splits.get("dt", {})
        blocks = {}
        dt_test = None
        with sw.time("clean"):
            for name in ["actual_train", "val", "test"]:
                if name in dt_blocks:
                    blocks[name], _dt = clean_block(
                        splits[name], capacity_mw, dt=dt_blocks[name])
                else:
                    blocks[name] = clean_block(splits[name], capacity_mw)
                if name == "test":
                    dt_test = _dt if name in dt_blocks else None
            for r, blk in splits["scarce"].items():
                blocks[f"scarce_{r}"] = clean_block(blk, capacity_mw)

        # 4) Fit MVMD-EN on the (cleaned) training subset only
        with sw.time("decomposition"):
            mvmd_en = self._make_mvmd_en()
            if mvmd_en is not None:
                mvmd_en.fit(blocks["actual_train"])
                if not self.ab["use_en"]:
                    mvmd_en.selected_mask_ = np.ones_like(mvmd_en.selected_mask_)

        # 5-7) Build windows + normalisers
        with sw.time("windowing"):
            train_win = self._build_train_windows(mvmd_en, blocks["actual_train"])
        with sw.time("windowing"):
            val_win = self._build_causal_windows(mvmd_en, blocks["val"])
        with sw.time("windowing"):
            test_win = self._build_causal_windows(mvmd_en, blocks["test"])

        # 6) Train branches
        with sw.time("train"):
            low_model, high_model, wgan = self._train_branches(train_win, mvmd_en)

        # 7) Fuse (val or OOF)
        with sw.time("fusion"):
            fusion = self._fit_fusion(val_win, low_model, high_model, train_win)

        # 8) Evaluate on test (with night mask + physical clip)
        with sw.time("inference"):
            result = self._evaluate(test_win, low_model, high_model, fusion,
                                    blocks["test"], dt_test)

        result["cost"] = self._cost_report(sw, low_model, high_model, wgan,
                                           train_win, test_win)
        return result

    # ------------------------------------------------------------------ #
    def _cost_report(self, sw, low_model, high_model, wgan,
                     train_win, test_win) -> dict:
        """Training time, inference latency, peak GPU memory and the
        trainable / frozen parameter split of the LLM branch."""
        rep = {
            "device": device_name(),
            "timings_s": sw.as_dict(),
            "peak_gpu_memory": peak_memory(),
        }
        if low_model is not None and hasattr(low_model, "model"):
            rep["llm_params"] = count_parameters(low_model.model)
            if hasattr(low_model, "last_train_seconds"):
                rep["timings_s"]["train_time_llm_branch"] = \
                    low_model.last_train_seconds
            if hasattr(low_model, "last_inference_seconds"):
                rep["timings_s"]["inference_llm_branch"] = \
                    low_model.last_inference_seconds
        if wgan is not None and getattr(wgan, "gen", None) is not None:
            g = count_parameters(wgan.gen)
            d = count_parameters(wgan.disc)
            rep["wgan_gp_params"] = {
                "generator": g, "discriminator": d,
                "total": g["total"] + d["total"],
            }
        if high_model is not None and hasattr(high_model, "complexity"):
            rep["xgboost_complexity"] = high_model.complexity()
            if hasattr(high_model, "last_train_seconds"):
                rep["timings_s"]["train_time_xgboost"] = high_model.last_train_seconds
            if hasattr(high_model, "last_inference_seconds"):
                rep["timings_s"]["inference_xgboost"] = high_model.last_inference_seconds
        rep["n_train_windows"] = int(train_win["X_low"].shape[0])
        rep["n_test_windows"] = int(test_win["X_low"].shape[0])
        return rep

    # ------------------------------------------------------------------ #
    def _training_subset(self, splits):
        if self.experiment == "full":
            return splits["actual_train"]
        r = float(self.experiment.replace("scarce", "")) / 100
        return splits["scarce"][r]

    def _make_mvmd_en(self):
        if not self.ab["use_mvmd_en"]:
            return None
        d = self.cfg["decomposition"]
        return MVMDEN(
            K=d["mvmd"]["K"], alpha=d["mvmd"]["alpha"], tolerance=d["mvmd"]["tolerance"],
            init=d["mvmd"]["init"], max_iter=d["mvmd"]["max_iter"],
            en_alpha=d["elastic_net"]["alpha"], l1_ratio=d["elastic_net"]["l1_ratio"],
            se_threshold=d["freq_separation"]["sample_entropy_threshold"],
            se_m=d["freq_separation"]["m"], se_r_factor=d["freq_separation"]["r_factor"],
            multivariate=self.ab["multivariate"],
        )

    # ------------------------------------------------------------------ #
    # Window builders
    # ------------------------------------------------------------------ #
    def _build_train_windows(self, mvmd_en, block):
        low, high, cov = self._decompose_block(mvmd_en, block)
        pv = block[0]
        # fit normalisers on the training decomposed series
        self.norm_low = MinMaxNormalizer().fit(low)
        self.norm_high = MinMaxNormalizer().fit(high)
        self.norm_cov = MinMaxNormalizer().fit(cov) if cov.size else None
        self.norm_pv = MinMaxNormalizer().fit(pv)
        low_n = self.norm_low.transform(low)
        high_n = self.norm_high.transform(high)
        cov_n = self.norm_cov.transform(cov) if cov.size else cov
        pv_n = self.norm_pv.transform(pv)
        # keep normalised training decomposed series for WGAN segment building
        self.train_high_n = high_n
        self.train_cov_n = cov_n if cov.size else np.empty((0, high_n.shape[0]))

        X_low, y_low = build_windows(low_n, None, self.input_len, self.pred_len)
        X_high, y_high = build_windows(high_n, cov_n, self.input_len, self.pred_len)
        y_pv = build_windows(pv_n, None, self.input_len, self.pred_len)[1]
        # Time-LLM expects (N, V, T); keep X_low as (N, input_len, 1) here.
        X_low = X_low.transpose(0, 2, 1)                    # (N, 1, input_len)
        return {"X_low": X_low, "y_low": y_low,
                "X_high": X_high, "y_high": y_high, "y_pv": y_pv}

    def _build_causal_windows(self, mvmd_en, block):
        """Validation / test windows.

        INPUTS: causal, window-by-window decomposition using only the 96-step
                history (decompose_window / separate_raw_window).
        TARGETS: from a whole-block decomposition of the block, used ONLY as
                 ground-truth labels for evaluation (never fed to the model).
        """
        low_t, high_t, cov_t = self._decompose_block(mvmd_en, block)
        pv_t = block[0]
        low_t_n = self.norm_low.transform(low_t)
        high_t_n = self.norm_high.transform(high_t)
        cov_t_n = self.norm_cov.transform(cov_t) if cov_t.size else cov_t
        pv_t_n = self.norm_pv.transform(pv_t)
        # ground-truth targets (whole-block, normalised)
        _, y_low = build_windows(low_t_n, None, self.input_len, self.pred_len)
        _, y_high = build_windows(high_t_n, cov_t_n, self.input_len, self.pred_len)
        y_pv = build_windows(pv_t_n, None, self.input_len, self.pred_len)[1]

        # causal inputs
        C, T = block.shape
        n = T - self.input_len - self.pred_len + 1
        X_low_list, X_high_list = [], []
        for i in range(n):
            hist = block[:, i:i + self.input_len]
            if mvmd_en is not None:
                lw, hw, cw = mvmd_en.decompose_window(hist)
            else:
                lw, hw = self._separate_raw(hist[0])
                cw = hist[1:]
            lw_n = self.norm_low.transform(lw)
            hw_n = self.norm_high.transform(hw)
            if cw.size:
                cw_n = self.norm_cov.transform(cw)
            else:
                cw_n = np.empty((0, self.input_len))
            X_low_list.append(lw_n)                         # (input_len,)
            X_high_list.append(np.concatenate([hw_n[None, :], cw_n], axis=0))  # (1+cov, input_len)
        X_low = np.stack(X_low_list)[:, None, :]             # (N, 1, input_len)
        X_high = np.stack(X_high_list).transpose(0, 2, 1)    # (N, input_len, 1+cov)
        return {"X_low": X_low, "y_low": y_low,
                "X_high": X_high, "y_high": y_high, "y_pv": y_pv}

    def _separate_raw(self, pv_window):
        """Case 1a frequency separation: EMD + sample-entropy on the raw PV."""
        from .decomposition.mvmd_en import MVMDEN
        fs = self.cfg["decomposition"]["freq_separation"]
        tmp = MVMDEN(se_threshold=fs["sample_entropy_threshold"],
                     se_m=fs["m"], se_r_factor=fs["r_factor"])
        return tmp.separate_raw_window(pv_window)

    def _decompose_block(self, mvmd_en, block):
        if mvmd_en is not None:
            return mvmd_en.decompose_series(block)
        pv = block[0]
        low, high = self._separate_raw(pv)
        cov = block[1:]
        return low, high, cov

    # ------------------------------------------------------------------ #
    # Branch training
    # ------------------------------------------------------------------ #
    def _train_branches(self, train_win, mvmd_en):
        # --- LOW branch: Time-LLM (or BERT for Case 4) ---
        low_model = None
        backbone = self.ab["backbone"]
        if backbone == "gpt2":
            from .lowfreq.time_llm import TimeLLMForecaster
            low_model = TimeLLMForecaster(self.cfg)
            low_model.fit(train_win["X_low"], train_win["y_low"])
        elif backbone == "bert":
            from .lowfreq.time_llm import BertForecaster
            low_model = BertForecaster(self.cfg)
            low_model.fit(train_win["X_low"], train_win["y_low"])
        else:
            raise RuntimeError(
                f"Backbone '{backbone}' is not implemented. Only 'gpt2' and "
                f"'bert' are supported (manuscript Case 4 swaps GPT-2 for BERT)."
            )

        # --- HIGH branch: WGAN-GP + XGBoost ---
        from .highfreq.wgan_gp import WGAN_GP
        from .highfreq.xgboost_model import XGBoostForecaster
        wgan = None
        Xh = train_win["X_high"]
        yh = train_win["y_high"]
        if self.ab["use_wgan_gp"]:
            wgan = WGAN_GP(self.cfg)
            # Build contiguous segments [input_len + pred_len] from the TRAIN
            # high-frequency series so the synthetic (input, target) pair is
            # internally consistent (no real-label reuse -- fixes the previous
            # bug where synthetic inputs were paired with real target labels).
            seg = self._build_aug_segments()                    # (N, L, F)
            wgan.fit(seg)
            n_syn = Xh.shape[0]
            syn_seg = wgan.augment(n_syn)                       # (n_syn, L, F)
            syn_X, syn_y = [], []
            for k in range(n_syn):
                sx, sy = WGAN_GP.split_segment(syn_seg[k], self.input_len)
                syn_X.append(sx); syn_y.append(sy)
            syn_X = np.stack(syn_X)                             # (n_syn, input_len, F)
            syn_y = np.stack(syn_y)                             # (n_syn, pred_len)
            Xh = np.concatenate([Xh, syn_X], axis=0)
            yh = np.concatenate([yh, syn_y], axis=0)
        high_model = XGBoostForecaster(self.cfg, seed=self.cfg["training"].get("seed", 0))
        high_model.fit(Xh, yh)
        return low_model, high_model, wgan

    def _build_aug_segments(self):
        """Assemble (N, L, F) high-frequency segments for WGAN training.

        Each segment = [input window | target window] along time, channels =
        high-freq target + covariates, sliced from the *normalised training
        decomposed series*. This guarantees the synthesised target continues the
        synthesised input (self-consistent pair).
        """
        L = self.input_len + self.pred_len
        high_n = self.train_high_n                                # (T_train,)
        cov_n = self.train_cov_n                                  # (C_cov, T_train) or empty
        n = high_n.shape[0] - L + 1
        segs = []
        for i in range(n):
            ch0 = high_n[i:i + L]                                 # (L,)
            if cov_n.size:
                cc = cov_n[:, i:i + L]                            # (C_cov, L)
                seg = np.concatenate([ch0[None, :], cc], axis=0)   # (1+C_cov, L)
            else:
                seg = ch0[None, :]
            segs.append(seg.T)                                    # (L, F)
        return np.stack(segs)                                     # (N, L, F)

    # ------------------------------------------------------------------ #
    def _fit_fusion(self, val_win, low_model, high_model, train_win):
        low_val = (low_model.predict(val_win["X_low"]) if low_model
                   else np.zeros_like(val_win["y_low"]))
        high_val = high_model.predict(val_win["X_high"])
        if self.ab["ridge"]:
            from .fusion.ridge_fusion import RidgeFusion
            fusion = RidgeFusion(self.cfg)
            fusion.fit(low_val, high_val, val_win["y_pv"])
            return fusion
        return None  # Case 3: simple summation (low + high = PV reconstruction)

    # ------------------------------------------------------------------ #
    def _evaluate(self, test_win, low_model, high_model, fusion, test_block, dt):
        low_pred = (low_model.predict(test_win["X_low"]) if low_model
                    else np.zeros_like(test_win["y_low"]))
        high_pred = high_model.predict(test_win["X_high"])
        if fusion is not None:
            pred_norm = fusion.predict(low_pred, high_pred)
        else:
            # Case 3: low + high reconstructs PV (the two branches predict
            # complementary components). NOTE: the manuscript Table 8 labels this
            # "simple average"; summation is the mathematically correct
            # reconstruction here -- flag for the paper wording.
            pred_norm = low_pred + high_pred
        pred = self.norm_pv.inverse_transform(pred_norm)
        y_true = self.norm_pv.inverse_transform(test_win["y_pv"])

        # datetime for the target steps (for the night mask)
        C, T = test_block.shape
        n = T - self.input_len - self.pred_len + 1
        dt_target = np.concatenate(
            [dt[i + self.input_len:i + self.input_len + self.pred_len] for i in range(n)]
        ) if dt is not None else None

        self.test_y_true = y_true.reshape(-1)
        self.test_y_pred = pred.reshape(-1)
        self.test_dt = dt_target

        # Metrics protocol (manuscript Section 4.1): physical MW scale, night
        # (solar elevation < 0) excluded, optional clamp to [0, capacity].
        metrics_cfg = self.cfg.get("metrics", {})
        night_exclude = metrics_cfg.get("night_exclude",
                                       self.cfg["data"].get("night_exclude", True))
        clip_to_capacity = metrics_cfg.get("clip_to_capacity", True)
        clip = (0.0, self.capacity) if clip_to_capacity else None
        metrics = evaluate(y_true, pred,
                          dt=dt_target if night_exclude else None,
                          lat=self.lat, lon=self.lon, clip=clip)
        return metrics
