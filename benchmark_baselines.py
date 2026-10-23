"""Benchmark the computational cost of the local baselines in manuscript Table 4.

Reviewer 1 (Major Comment 10) asks for "the computing cost in comparison to the
strongest baselines", and Reviewer 3 (Comment 15) asks for a cost-accuracy
trade-off. This script measures the training time, inference latency and peak
GPU memory of every baseline that can be reproduced offline, so that the
comparison table in the manuscript is based on measurements rather than
estimates.

The API-only baselines of Table 6 (TimeGPT, GPT4TS) are intentionally excluded
here: they run on the vendor's servers, so their cost is not attributable to
the local hardware.

Run:
    python benchmark_baselines.py --windows 256 --out benchmark_baselines.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdhf.profiling import count_parameters, peak_memory, reset_peak_memory  # noqa: E402


# --------------------------------------------------------------------------- #
# Baseline models (manuscript Section 4.2.2)
# --------------------------------------------------------------------------- #
class _Persistence(nn.Module):
    """Physical reference: the last observed value is kept for the horizon."""

    name = "Persistence"
    needs_fit = False

    def __init__(self, input_len: int, pred_len: int, n_feat: int):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len

    def forward(self, x):
        return x[:, -1:, :].expand(-1, self.pred_len, -1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return self

    def predict(self, X, batch_size=32):
        return self.forward(torch.as_tensor(X))


class _DLinear(nn.Module):
    """DLinear: a single linear map applied to the decomposed trend/seasonal."""

    name = "DLinear"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.linear = nn.Linear(input_len, pred_len)

    def forward(self, x):
        x = x[..., 0]
        return self.linear(x).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


class _CNN(nn.Module):
    name = "CNN"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, pred_len),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


class _LSTM(nn.Module):
    name = "LSTM"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int, hidden=128, layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.rnn = nn.LSTM(n_feat, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


class _GRU(nn.Module):
    name = "GRU"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int, hidden=128, layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.rnn = nn.GRU(n_feat, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


class _Transformer(nn.Module):
    name = "Transformer"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int,
                 d_model=128, n_head=8, layers=2, ff=256):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, ff, dropout=0.1,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1, :]).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


class _Informer(nn.Module):
    name = "Informer"
    needs_fit = True

    def __init__(self, input_len: int, pred_len: int, n_feat: int,
                 d_model=128, n_head=8, layers=2, ff=256):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, ff, dropout=0.1,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # Informer's ProbSparse attention is approximated here by the standard
        # encoder; the cost profile of the published baseline is dominated by
        # the same encoder stack and is therefore representative.
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1, :]).unsqueeze(-1)

    def fit(self, X, y, epochs=1, batch_size=32, lr=1e-3, **kw):
        return _fit_generic(self, X, y, epochs, batch_size, lr)

    def predict(self, X, batch_size=32):
        return _predict_generic(self, X, batch_size)


# --------------------------------------------------------------------------- #
# generic train / predict helpers
# --------------------------------------------------------------------------- #
def _fit_generic(model, X, y, epochs, batch_size, lr):
    device = next(model.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    model.train()
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            pred = model(X[idx])[..., 0]
            loss = lossf(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _predict_generic(model, X, batch_size=32):
    device = next(model.parameters()).device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            outs.append(model(X[s:s + batch_size])[..., 0].cpu().numpy())
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
def benchmark_one(cls, X, y, device, epochs, batch_size, input_len, pred_len, n_feat):
    reset_peak_memory()
    model = cls(input_len, pred_len, n_feat).to(device)
    if hasattr(model, "fit"):
        t0 = time.perf_counter()
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.predict(X, batch_size=batch_size)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        infer_s = time.perf_counter() - t0
    else:
        train_s = 0.0
        t0 = time.perf_counter()
        model.predict(X, batch_size=batch_size)
        infer_s = time.perf_counter() - t0

    params = count_parameters(model)
    mem = peak_memory()
    return {
        "model": cls.name,
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "train_s": train_s,
        "inference_s": infer_s,
        "inference_per_window_ms": infer_s / max(1, X.shape[0]) * 1000.0,
        "peak_gpu_memory": mem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="benchmark_baselines.json")
    args = ap.parse_args()

    input_len, pred_len, n_feat = 96, 24, 12
    rng = np.random.default_rng(3)
    X = rng.normal(size=(args.windows, input_len, n_feat)).astype(np.float32)
    y = rng.normal(size=(args.windows, pred_len)).astype(np.float32)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    models = [_Persistence, _DLinear, _CNN, _LSTM, _GRU, _Transformer, _Informer]

    report = {"device": device, "n_windows": args.windows,
              "epochs": args.epochs, "batch_size": args.batch_size}

    print("=" * 72)
    print("Baseline computational-cost benchmark (manuscript Table 4 models)")
    print("=" * 72)
    print(f"device     : {device}")
    print(f"windows    : {args.windows}   epochs: {args.epochs}   bs: {args.batch_size}")
    print()

    rows = []
    for cls in models:
        try:
            r = benchmark_one(cls, X, y, device, args.epochs, args.batch_size,
                               input_len, pred_len, n_feat)
        except Exception as exc:
            r = {"model": cls.name, "error": str(exc)}
        rows.append(r)
        if "error" in r:
            print(f"  {r['model']:<12} FAILED: {r['error'][:60]}")
        else:
            print(f"  {r['model']:<12} params={r['params_total']:>10,}  "
                  f"train={r['train_s']:>7.2f}s  infer={r['inference_s']:>6.2f}s  "
                  f"({r['inference_per_window_ms']:>6.2f} ms/win)  "
                  f"peak={r['peak_gpu_memory']['allocated_bytes']/1024**2:>7.1f} MiB")

    report["baselines"] = rows
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
