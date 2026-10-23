"""Measure the computational cost of the FDHF framework.

Reports the four quantities needed for the manuscript's efficiency discussion:

  1. LLM parameter count (total / frozen / trainable)
  2. Training time per branch and end-to-end
  3. Inference latency (per test window and for the whole test set)
  4. Peak GPU memory

Run it on the machine that is described in the manuscript's experimental
section, e.g.

    python benchmark_cost.py --csv data/pv_scada.csv --config configs/default.yaml

The script prints a Markdown-ready table and also writes the same numbers to
``benchmark_cost.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdhf.data.loader import load_raw, split_series, clean_block  # noqa: E402
from fdhf.lowfreq.prompt_bank import build_pap_prompt, compute_window_stats  # noqa: E402
from fdhf.profiling import (  # noqa: E402
    count_parameters, device_name, peak_memory, reset_peak_memory,
)


def _fmt_seconds(x: float) -> str:
    if x < 90:
        return f"{x:.1f} s"
    if x < 5400:
        return f"{x / 60:.1f} min"
    return f"{x / 3600:.2f} h"


def benchmark_llm_branch(cfg: dict, n_windows: int, device: str,
                         batch_size: int | None = None) -> dict:
    """Time the Time-LLM branch and count its parameters."""
    import torch
    from fdhf.lowfreq.time_llm import TimeLLMForecaster

    lf = dict(cfg["low_freq_branch"])
    input_len = cfg["data"]["input_len"]
    pred_len = cfg["data"]["pred_len"]
    if batch_size:
        lf["batch_size"] = int(batch_size)
    bs = lf["batch_size"]

    # synthetic windows with the real shape; the cost depends on tensor
    # dimensions, not on the physical values
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_windows, 1, input_len)).astype(np.float32)
    y = rng.normal(size=(n_windows, pred_len)).astype(np.float32)

    cfg = dict(cfg)
    cfg["low_freq_branch"] = lf
    cfg["training"] = dict(cfg["training"])
    cfg["training"]["device"] = device

    reset_peak_memory()
    model = TimeLLMForecaster(cfg)

    params = count_parameters(model.model)
    backbone = count_parameters(model.model.llm)

    # ---- training: one epoch, extrapolated to the configured epoch count --
    epochs = int(lf["max_epochs"])
    cfg1 = dict(cfg)
    cfg1["low_freq_branch"] = dict(lf)
    cfg1["low_freq_branch"]["max_epochs"] = 1
    model.cfg = cfg1
    model.epochs = 1

    t0 = time.perf_counter()
    model.fit(X, y)
    one_epoch_s = time.perf_counter() - t0
    per_epoch_s = one_epoch_s  # measured on the full window count
    total_train_s = per_epoch_s * epochs

    # ---- inference latency -----------------------------------------------
    t0 = time.perf_counter()
    model.predict(X)
    infer_s = time.perf_counter() - t0
    per_window_ms = infer_s / max(1, n_windows) * 1000.0

    mem = peak_memory()
    return {
        "device": device_name(),
        "llm_params": params,
        "llm_backbone_params": backbone,
        "n_windows": n_windows,
        "batch_size": bs,
        "epochs": epochs,
        "train_one_epoch_s": one_epoch_s,
        "train_total_s": total_train_s,
        "inference_s": infer_s,
        "inference_per_window_ms": per_window_ms,
        "peak_gpu_memory": mem,
    }


def benchmark_wgan(cfg: dict, n_segments: int, device: str) -> dict:
    """Time the WGAN-GP augmentation stage and count its parameters."""
    import torch
    from fdhf.highfreq.wgan_gp import WGAN_GP

    g = cfg["high_freq_branch"]["wgan_gp"]
    L = cfg["data"]["input_len"] + cfg["data"]["pred_len"]
    F = 1 + len(cfg["data"]["feature_cols"])
    rng = np.random.default_rng(1)
    seg = rng.normal(size=(n_segments, L, F)).astype(np.float32)

    cfg = dict(cfg)
    cfg["training"] = dict(cfg["training"])
    cfg["training"]["device"] = device
    cfg["high_freq_branch"] = dict(cfg["high_freq_branch"])
    cfg["high_freq_branch"]["wgan_gp"] = dict(g)
    # one epoch for the timing measurement; the configured value is reported
    cfg["high_freq_branch"]["wgan_gp"]["epochs"] = 1

    reset_peak_memory()
    wgan = WGAN_GP(cfg)
    t0 = time.perf_counter()
    wgan.fit(seg)
    one_epoch_s = time.perf_counter() - t0
    per_epoch_s = one_epoch_s
    total_s = per_epoch_s * int(g["epochs"])

    gen = count_parameters(wgan.gen)
    disc = count_parameters(wgan.disc)
    return {
        "n_segments": n_segments,
        "epochs": int(g["epochs"]),
        "train_one_epoch_s": one_epoch_s,
        "train_total_s": total_s,
        "generator_params": gen,
        "discriminator_params": disc,
        "peak_gpu_memory": peak_memory(),
    }


def benchmark_xgboost(cfg: dict, n_windows: int) -> dict:
    """Time the XGBoost stage and report the tree-ensemble size."""
    try:
        from fdhf.highfreq.xgboost_model import XGBoostForecaster
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        return {"available": False, "reason": f"xgboost not installed: {exc}"}

    input_len = cfg["data"]["input_len"]
    pred_len = cfg["data"]["pred_len"]
    F = 1 + len(cfg["data"]["feature_cols"])
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n_windows, input_len, F)).astype(np.float32)
    y = rng.normal(size=(n_windows, pred_len)).astype(np.float32)

    model = XGBoostForecaster(cfg, seed=0)
    t0 = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.predict(X)
    infer_s = time.perf_counter() - t0
    out = {
        "available": True,
        "n_windows": n_windows,
        "train_s": train_s,
        "inference_s": infer_s,
        "inference_per_window_ms": infer_s / max(1, n_windows) * 1000.0,
    }
    out.update(model.complexity())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="real SCADA CSV (optional)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--windows", type=int, default=2048,
                    help="number of training windows used for the timing run")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the configured batch size (memory-limited GPUs)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override Time-LLM epochs (default: config max_epochs)")
    ap.add_argument("--wgan-epochs", type=int, default=None,
                    help="override WGAN-GP epochs (default: config epochs)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: config)")
    ap.add_argument("--out", default="benchmark_cost.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["low_freq_branch"] = dict(cfg["low_freq_branch"])
        cfg["low_freq_branch"]["max_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["low_freq_branch"] = dict(cfg["low_freq_branch"])
        cfg["low_freq_branch"]["batch_size"] = int(args.batch_size)
    if args.wgan_epochs is not None:
        cfg["high_freq_branch"] = dict(cfg["high_freq_branch"])
        cfg["high_freq_branch"]["wgan_gp"] = dict(cfg["high_freq_branch"]["wgan_gp"])
        cfg["high_freq_branch"]["wgan_gp"]["epochs"] = int(args.wgan_epochs)

    device = args.device or cfg["training"]["device"]
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[warn] CUDA not available -- falling back to CPU")
                device = "cpu"
        except Exception:
            device = "cpu"

    report = {"device": device, "device_name": device_name()}

    print("=" * 68)
    print("FDHF computational-cost benchmark")
    print("=" * 68)
    print(f"device          : {report['device_name']} ({device})")
    print(f"windows         : {args.windows}")
    print(f"batch size      : {args.batch_size or cfg['low_freq_branch']['batch_size']}")
    print(f"epochs          : {cfg['low_freq_branch']['max_epochs']}")
    print()

    print("[1/3] Time-LLM branch (frozen GPT-2 + trainable adapters)")
    report["time_llm"] = benchmark_llm_branch(cfg, args.windows, device,
                                              args.batch_size)
    print("[2/3] WGAN-GP augmentation stage")
    report["wgan_gp"] = benchmark_wgan(cfg, args.windows, device)
    print("[3/3] XGBoost stage")
    report["xgboost"] = benchmark_xgboost(cfg, args.windows)
    print()

    llm = report["time_llm"]
    wgan = report["wgan_gp"]
    xgb = report["xgboost"]

    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print("LLM (GPT-2-Small backbone + trainable adapters)")
    print(f"  backbone total      : {llm['llm_backbone_params']['total']:>14,} params")
    print(f"  backbone frozen     : {llm['llm_backbone_params']['frozen']:>14,} params")
    print(f"  trainable adapters  : {llm['llm_params']['trainable']:>14,} params")
    print(f"  full model total    : {llm['llm_params']['total']:>14,} params")
    print(f"  training (1 epoch)  : {_fmt_seconds(llm['train_one_epoch_s'])}")
    print(f"  training (all {llm['epochs']})   : {_fmt_seconds(llm['train_total_s'])}")
    print(f"  inference latency   : {llm['inference_per_window_ms']:.3f} ms/window")
    print(f"  peak GPU memory     : {llm['peak_gpu_memory']['allocated_bytes'] / 1024**2:.1f} MiB")
    print()
    if wgan.get("generator_params"):
        print("WGAN-GP")
        print(f"  generator           : {wgan['generator_params']['total']:>14,} params")
        print(f"  discriminator       : {wgan['discriminator_params']['total']:>14,} params")
        print(f"  training (1 epoch)  : {_fmt_seconds(wgan['train_one_epoch_s'])}")
        print(f"  training (all {wgan['epochs']}): {_fmt_seconds(wgan['train_total_s'])}")
        print()
    if xgb.get("available"):
        print("XGBoost")
        print(f"  trees               : {xgb['n_trees']:>14,}")
        print(f"  leaves              : {xgb['n_leaves']:>14,}")
        print(f"  training            : {_fmt_seconds(xgb['train_s'])}")
        print(f"  inference latency   : {xgb['inference_per_window_ms']:.3f} ms/window")
        print()
    else:
        print(f"XGBoost: skipped ({xgb.get('reason')})")
        print()

    total_train = llm["train_total_s"] + wgan.get("train_total_s", 0.0) + \
        xgb.get("train_s", 0.0)
    print(f"END-TO-END training time (sum of stages): {_fmt_seconds(total_train)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
