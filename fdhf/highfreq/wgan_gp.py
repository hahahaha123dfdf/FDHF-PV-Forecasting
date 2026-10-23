"""WGAN-GP for high-frequency-component data augmentation.

Implements Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017) with
**GRU** generator and discriminator (manuscript Section 3.3, line 50: single-layer
GRUs, hidden 64, latent 32, lambda = 10).

IMPORTANT (manuscript Table 10 / Section 4.2.5): WGAN-GP augments the
*high-frequency component*. It is trained on contiguous high-frequency segments
that already contain BOTH the input window and the following target window
(L = input_len + pred_len). Each synthetic segment is then split internally into
a consistent (input, target) pair -- the synthesised target is the actual
continuation of the synthesised input, never a copy of a real label. This fixes
the previous bug where synthetic inputs were paired with real labels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _GRUGenerator(nn.Module):
    def __init__(self, latent: int, hidden: int, length: int, feat: int):
        super().__init__()
        self.length = length
        self.feat = feat
        self.hidden = hidden
        self.fc_z = nn.Linear(latent, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, feat)

    def forward(self, z):
        h0 = self.fc_z(z).unsqueeze(0)              # (1, B, hidden)
        steps = torch.zeros(z.size(0), self.length, self.hidden, device=z.device)
        out, _ = self.gru(steps, h0)
        return self.out(out)                         # (B, L, F)


class _GRUDiscriminator(nn.Module):
    def __init__(self, hidden: int, feat: int):
        super().__init__()
        self.gru = nn.GRU(feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])                       # (B, 1)


class WGAN_GP:
    def __init__(self, cfg: dict):
        g = cfg["high_freq_branch"]["wgan_gp"]
        self.hidden = g["hidden_size"]
        self.latent = g["latent_dim"]
        self.epochs = g["epochs"]
        self.batch_size = g["batch_size"]
        self.lambda_gp = g["lambda_gp"]
        self.n_critic = g["n_critic"]
        self.lr_g = g["lr_g"]
        self.lr_d = g["lr_d"]
        self.device = cfg["training"]["device"]
        self.length = None    # set on fit (= input_len + pred_len)
        self.feat = None
        self.gen = self.disc = None

    # ------------------------------------------------------------------ #
    def fit(self, segments: np.ndarray):
        """Train on real high-frequency segments of shape (N, L, F).

        L = input_len + pred_len; F = 1 (high-freq target) + n_covariates.
        """
        segments = torch.tensor(np.asarray(segments, dtype=float),
                                dtype=torch.float32).to(self.device)
        N, L, F = segments.shape
        self.length, self.feat = L, F
        self.gen = _GRUGenerator(self.latent, self.hidden, L, F).to(self.device)
        self.disc = _GRUDiscriminator(self.hidden, F).to(self.device)
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(0.5, 0.9))
        gp = self.lambda_gp

        for _ in range(self.epochs):
            idx = torch.randperm(N)
            for s in range(0, N, self.batch_size):
                real = segments[idx[s:s + self.batch_size]]
                if real.size(0) < 2:
                    continue
                # ---- critic ----
                z = torch.randn(real.size(0), self.latent, device=self.device)
                fake = self.gen(z).detach()
                d_real = self.disc(real)
                d_fake = self.disc(fake)
                grad_pen = self._gradient_penalty(real, fake)
                loss_d = d_fake.mean() - d_real.mean() + gp * grad_pen
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                # ---- generator ----
                if (s // self.batch_size) % self.n_critic == 0:
                    z = torch.randn(real.size(0), self.latent, device=self.device)
                    fake = self.gen(z)
                    loss_g = -self.disc(fake).mean()
                    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        return self

    def _gradient_penalty(self, real, fake):
        a = torch.rand(real.size(0), 1, 1, device=self.device)
        interp = (a * real + (1 - a) * fake).requires_grad_(True)
        # The penalty differentiates the critic w.r.t. its input, which is a
        # double backward through the GRU. The cuDNN RNN backend does not
        # implement double backward, so it is disabled for this computation.
        with torch.backends.cudnn.flags(enabled=False):
            d_interp = self.disc(interp)
            grads = torch.autograd.grad(d_interp.sum(), interp,
                                        create_graph=True, retain_graph=True)[0]
        return ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def augment(self, n: int) -> np.ndarray:
        """Return (n, L, F) synthetic high-frequency segments."""
        z = torch.randn(n, self.latent, device=self.device)
        return self.gen(z).cpu().numpy()

    @staticmethod
    def split_segment(seg: np.ndarray, input_len: int):
        """Split a (L, F) segment into (X, y): X = first input_len steps,
        y = the following (L - input_len) steps. Guarantees an internally
        consistent (input, target) pair."""
        return seg[:input_len], seg[input_len:]
