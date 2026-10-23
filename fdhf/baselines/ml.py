"""Deep-learning baselines (manuscript Table 4): LSTM, GRU, CNN, DLinear.

torch is imported lazily inside the class so the module is importable without
torch installed. These are the manuscript's direct multi-horizon forecasters
trained on the raw PV series (no decomposition).
"""

from __future__ import annotations

import numpy as np

from .base import BaseForecaster


class _TorchForecaster(BaseForecaster):
    """Shared training / prediction loop for the torch baselines."""

    def __init__(self, kind: str, input_len: int = 96, pred_len: int = 24,
                 hidden: int = 64, epochs: int = 20, batch_size: int = 32,
                 lr: float = 1e-3, device: str = "cpu"):
        self.kind = kind
        self.input_len = input_len
        self.pred_len = pred_len
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.net = None

    def _build_net(self, n_feat: int):
        import torch.nn as nn
        if self.kind == "lstm":
            return nn.LSTM(n_feat, self.hidden, batch_first=True)
        if self.kind == "gru":
            return nn.GRU(n_feat, self.hidden, batch_first=True)
        if self.kind == "cnn":
            return nn.Sequential(
                nn.Conv1d(n_feat, self.hidden, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
        if self.kind == "dlinear":
            return nn.Sequential(nn.Linear(self.input_len, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.pred_len))
        raise ValueError(self.kind)

    def fit(self, X, y):
        import torch
        X = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
        y = torch.tensor(np.asarray(y, dtype=float), dtype=torch.float32)
        n_feat = X.shape[-1] if X.ndim == 3 else 1
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        net = self._build_net(n_feat).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        net.train()
        for _ in range(self.epochs):
            for s in range(0, X.shape[0], self.batch_size):
                xb, yb = X[s:s + self.batch_size], y[s:s + self.batch_size]
                opt.zero_grad()
                if self.kind == "dlinear":
                    p = net(xb.squeeze(-1))
                elif self.kind == "cnn":
                    p = net(xb.transpose(1, 2)).squeeze(-1)
                else:
                    out, _ = net(xb)
                    p = out[:, -1]
                p = p.reshape(yb.shape) if p.shape == yb.shape else p
                loss_fn(p, yb).backward(); opt.step()
        self.net = net
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
        if X.ndim == 2:
            X = X.unsqueeze(-1)
        self.net.eval()
        with torch.no_grad():
            if self.kind == "dlinear":
                p = self.net(X.squeeze(-1))
            elif self.kind == "cnn":
                p = self.net(X.transpose(1, 2)).squeeze(-1)
            else:
                out, _ = self.net(X); p = out[:, -1]
        return p.cpu().numpy().reshape(X.shape[0], -1)


def make_lstm(**kw):   return _TorchForecaster("lstm", **kw)
def make_gru(**kw):    return _TorchForecaster("gru", **kw)
def make_cnn(**kw):    return _TorchForecaster("cnn", **kw)
def make_dlinear(**kw): return _TorchForecaster("dlinear", **kw)
