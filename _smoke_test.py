"""Smoke test for the non-DL parts of FDHF (no torch/xgboost required)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import fdhf  # noqa: ensures package imports without torch
from fdhf.data.loader import (MinMaxNormalizer, build_windows, clean_block,
                              split_series)
from fdhf.data.metrics import (evaluate, night_mask, r2, rmse, solar_elevation_deg)
from fdhf.decomposition.mvmd import mvmd
from fdhf.decomposition.sample_entropy import sample_entropy
from fdhf.decomposition.emd import emd
from fdhf.decomposition.mvmd_en import MVMDEN
from fdhf.fusion.ridge_fusion import RidgeFusion


def make_config():
    return {
        "data": {"input_len": 48, "pred_len": 12, "station_lat": 36.644,
                 "station_lon": 113.642, "capacity_mw": 20.0,
                 "train_ratio": 0.7, "val_ratio_in_train": 0.15,
                 "scarce_ratios": [0.15]},
        "decomposition": {
            "mvmd": {"K": 4, "alpha": 2000, "tolerance": 1e-3, "init": 1, "max_iter": 50},
            "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
            "freq_separation": {"sample_entropy_threshold": 0.13, "m": 2, "r_factor": 0.2},
        },
        "fusion": {"alpha": 1.0, "out_of_fold": True},
    }


def make_series(T=2000, C=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # non-negative PV in [~1, ~9] MW (stays within rated capacity 20 MW so the
    # outlier mask does not fire -- we are testing independence, not cleaning)
    pv = 5.0 + 4.0 * np.sin(2 * np.pi * t / 96.0) + 0.1 * rng.normal(0, 1, T)
    cov = rng.normal(0, 1, (C - 1, T)) * np.array([1.0, 100.0, 0.01])[:, None]
    return np.vstack([pv[None, :], cov])


def test_clean_block_independent():
    cfg = make_config()
    series = make_series(T=400)
    split = split_series(series, cfg)
    tr = split["actual_train"]; va = split["val"]
    tr_c = clean_block(tr, 20.0); va_c = clean_block(va, 20.0)
    assert not np.isnan(tr_c).any() and not np.isnan(va_c).any()
    whole = clean_block(np.concatenate([tr, va], axis=1), 20.0)
    assert np.allclose(whole[:, tr.shape[1]:], va_c)
    print("[OK] per-block cleaning independent, no NaN")


def test_causal_decomposition():
    series = make_series(T=600)
    mv = MVMDEN(K=4, alpha=2000, max_iter=50, se_threshold=0.13)
    mv.fit(series)
    L = make_config()["data"]["input_len"]
    window = series[:, 100:100 + L]
    lw, hw, cw = mv.decompose_window(window)
    modes = mv._decompose(window)
    low2, high2 = mv._reconstruct(modes)
    assert np.allclose(lw, low2) and np.allclose(hw, high2)
    assert lw.shape[0] == L and hw.shape[0] == L
    print("[OK] causal decompose_window uses only the history window")


def test_solar_elevation_night():
    import pandas as pd
    night = pd.to_datetime(["2018-06-30 02:00:00"])
    noon = pd.to_datetime(["2018-06-30 12:00:00"])
    e_n = solar_elevation_deg(night, 36.644, 113.642)
    e_d = solar_elevation_deg(noon, 36.644, 113.642)
    assert e_n[0] < 0 < e_d[0], (e_n, e_d)
    nm = night_mask(night, 36.644, 113.642)
    assert bool(nm[0]) is True
    print(f"[OK] solar elevation night={e_n[0]:.1f}  day={e_d[0]:.1f}")


def test_sample_entropy_fix():
    # Constant signal -> every template matches -> SampleEn ~ 0 (lowest
    # complexity, correctly classified LOW frequency). Finite-N boundary
    # effects give a small positive value (~0.04); the point is that it is
    # well below the 0.13 separation threshold, NOT that it equals exactly 0.
    s_const = sample_entropy(np.ones(50), r=0.1)
    assert 0.0 <= s_const < 0.1, s_const
    # No-match signal: strictly monotonic so no two length-m windows are within
    # r -> SampleEn diverges -> large finite value (maximal complexity, must be
    # classified HIGH frequency, kept out of the low-frequency branch).
    s_nomatch = sample_entropy(np.arange(20.0) * 100.0, m=2, r=1.0)
    assert s_nomatch > 1e5, s_nomatch
    # Random signal -> intermediate entropy.
    rng = np.random.default_rng(1)
    s2 = sample_entropy(rng.normal(0, 1, 60), r=0.2)
    assert 0 < s2 < 50, s2
    print(f"[OK] sample_entropy fix: const={s_const:.4f}  no-match={s_nomatch:.0f}  random={s2:.3f}")


def test_per_channel_norm():
    x = np.array([[0.0, 10.0, 20.0], [0.0, 1.0, 2.0]])
    n = MinMaxNormalizer().fit(x)
    xn = n.transform(x)
    assert np.allclose(xn[0], [0, 0.5, 1.0]) and np.allclose(xn[1], [0, 0.5, 1.0])
    print("[OK] per-channel normalisation independent per channel")


def test_oof_fusion():
    rng = np.random.default_rng(0)
    lo = rng.normal(0, 1, (200, 12)); hi = rng.normal(0, 1, (200, 12))
    y = lo + hi + rng.normal(0, 0.1, (200, 12))
    for oof in (True, False):
        f = RidgeFusion({"fusion": {"alpha": 1.0, "out_of_fold": oof}})
        f.fit(lo, hi, y)
        p = f.predict(lo, hi)
        assert p.shape == (200, 12)
    print("[OK] ridge fusion out_of_fold True/False both run")


def test_emd_separation():
    rng = np.random.default_rng(2)
    t = np.arange(300)
    # A period-200 sine is genuinely low-complexity (SampEn ~ 0.08 < 0.13);
    # white noise is high-complexity (SampEn ~ 2.3 > 0.13).
    low = np.sin(2 * np.pi * t / 200.0)
    high = rng.normal(0, 1, 300)
    pv = low + high
    mv = MVMDEN(se_threshold=0.13)
    lw, hw = mv.separate_raw_window(pv)
    # EMD (+ residue) must reconstruct the original exactly.
    assert np.allclose(lw + hw, pv, atol=1e-6), np.abs(lw + hw - pv).max()
    se_low = sample_entropy(low, r=0.2 * np.std(low))
    se_high = sample_entropy(high, r=0.2 * np.std(high))
    assert se_low < 0.13 < se_high, (se_low, se_high)
    print(f"[OK] EMD separation: SE_low={se_low:.3f}  SE_high={se_high:.3f}")


def test_pipeline_windows():
    cfg = make_config()
    series = make_series(T=700)
    mv = MVMDEN(K=4, alpha=2000, max_iter=50, se_threshold=0.13)
    mv.fit(series)
    from fdhf.pipeline import FDHFPipeline
    pipe = FDHFPipeline(cfg, experiment="full", ablation=None)
    pipe._make_mvmd_en = lambda: mv
    trw = pipe._build_train_windows(mv, series[:, :400])
    assert trw["X_low"].shape[0] == trw["y_low"].shape[0]
    cw = pipe._build_causal_windows(mv, series[:, 400:])
    assert cw["X_low"].shape[0] == cw["y_low"].shape[0] == cw["y_pv"].shape[0]
    print("[OK] pipeline train + causal window builders produce aligned shapes")


if __name__ == "__main__":
    test_clean_block_independent()
    test_causal_decomposition()
    test_solar_elevation_night()
    test_sample_entropy_fix()
    test_per_channel_norm()
    test_oof_fusion()
    test_emd_separation()
    test_pipeline_windows()
    print("\nALL SMOKE TESTS PASSED")
