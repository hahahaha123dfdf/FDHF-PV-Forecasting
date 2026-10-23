"""Measure how the Time-LLM branch scales with batch size.

Peak GPU memory for a fixed model grows close to linearly with the batch
size, so a sweep over several batch sizes yields a line that can be
extrapolated to the batch size used in the manuscript even when the local
card is too small to run that batch size directly.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdhf.lowfreq.time_llm import TimeLLMForecaster  # noqa: E402
from fdhf.profiling import count_parameters, peak_memory, reset_peak_memory  # noqa: E402


def run_one(cfg, bs, n_windows, device):
    lf = dict(cfg["low_freq_branch"])
    lf["batch_size"] = bs
    lf["max_epochs"] = 1
    cfg = dict(cfg)
    cfg["low_freq_branch"] = lf
    cfg["training"] = dict(cfg["training"])
    cfg["training"]["device"] = device

    input_len = cfg["data"]["input_len"]
    pred_len = cfg["data"]["pred_len"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_windows, 1, input_len)).astype(np.float32)
    y = rng.normal(size=(n_windows, pred_len)).astype(np.float32)

    reset_peak_memory()
    model = TimeLLMForecaster(cfg)
    model.epochs = 1
    t0 = time.perf_counter()
    model.fit(X, y)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.predict(X)
    infer_s = time.perf_counter() - t0
    mem = peak_memory()
    return {
        "batch_size": bs,
        "n_windows": n_windows,
        "train_s": train_s,
        "per_window_train_ms": train_s / n_windows * 1000.0,
        "inference_s": infer_s,
        "per_window_infer_ms": infer_s / n_windows * 1000.0,
        "peak_allocated_mib": mem["allocated_bytes"] / 1024 ** 2,
        "peak_reserved_mib": mem["reserved_bytes"] / 1024 ** 2,
    }


def main():
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = cfg["training"]["device"]
    n_windows = 256

    print("=" * 74)
    print("Time-LLM branch: batch-size scaling")
    print("=" * 74)
    print(f"{'batch':>7} {'peak alloc MiB':>16} {'peak resv MiB':>15} "
          f"{'train ms/win':>14} {'infer ms/win':>14}")
    rows = []
    for bs in [2, 4, 8, 16, 24]:
        try:
            r = run_one(cfg, bs, n_windows, device)
            rows.append(r)
            print(f"{bs:>7} {r['peak_allocated_mib']:>16.1f} "
                  f"{r['peak_reserved_mib']:>15.1f} "
                  f"{r['per_window_train_ms']:>14.3f} "
                  f"{r['per_window_infer_ms']:>14.3f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>7} {'OOM':>16}")
            torch.cuda.empty_cache()
    print()

    # Peak memory is dominated by a fixed term (frozen backbone + optimizer
    # state + CUDA context) plus a term linear in the batch size. The two
    # largest feasible batch sizes give the most reliable slope, so the
    # extrapolation is anchored on the upper end of the sweep.
    if len(rows) >= 2:
        x = np.array([r["batch_size"] for r in rows], dtype=float)
        y = np.array([r["peak_allocated_mib"] for r in rows], dtype=float)
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        fixed = y[-1] - slope * x[-1]
        print(f"peak-allocated model:  MiB = {fixed:.1f} (fixed) "
              f"+ {slope:.1f} * batch")
        for target in (32, 64):
            print(f"  extrapolated peak at batch {target:>3}: "
                  f"{fixed + slope * target:>8.1f} MiB")
        print()

    # per-window throughput is roughly batch-independent -> scale to the
    # manuscript's actual training set (20,849 samples -> 20,730 windows)
    if rows:
        best = rows[-1]
        n_train_windows = 20849 - 96 - 24 + 1
        n_test_windows = 10512 - 96 - 24 + 1
        epoch_s = best["per_window_train_ms"] / 1000.0 * n_train_windows
        print(f"throughput from batch {best['batch_size']} "
              f"({best['per_window_train_ms']:.3f} ms/window train)")
        print(f"  manuscript train set  : {n_train_windows:,} windows")
        print(f"  -> 1 epoch            : {epoch_s / 60:.1f} min")
        print(f"  -> 10 epochs          : {epoch_s * 10 / 3600:.2f} h")
        print(f"  -> test inference     : "
              f"{best['per_window_infer_ms'] / 1000.0 * n_test_windows:.1f} s "
              f"for {n_test_windows:,} windows")


if __name__ == "__main__":
    main()
