"""DCGAN baseline for high-frequency augmentation (manuscript Table 10).

Produces consistent (input, target) augmentation pairs like the WGAN-GP branch
segments, but uses the standard DCGAN (convolutional) generator/discriminator
instead of the WGAN-GP gradient penalty. torch is imported lazily.
"""

from __future__ import annotations

import numpy as np


class DCGAN:
    def __init__(self, cfg: dict):
        g = cfg["high_freq_branch"]["wgan_gp"]
        self.length = None
        self.feat = None
        self.latent = g["latent_dim"]
        self.epochs = g["epochs"]
        self.batch_size = g["batch_size"]
        self.device = cfg["training"]["device"]
        self.gen = self.disc = None

    def fit(self, segments: np.ndarray):
        import torch
        import torch.nn as nn
        segments = torch.tensor(np.asarray(segments, dtype=float),
                                dtype=torch.float32).to(self.device)
        N, L, F = segments.shape
        self.length, self.feat = L, F

        class Gen(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(self.latent, 64), nn.ReLU(),
                    nn.Linear(64, L * F), nn.Tanh())
            def forward(self, z):
                return self.net(z).view(-1, L, F)

        class Dis(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(L * F, 64), nn.LeakyReLU(0.2),
                    nn.Linear(64, 1), nn.Sigmoid())
            def forward(self, x):
                return self.net(x.reshape(x.shape[0], -1))

        self.gen, self.disc = Gen().to(self.device), Dis().to(self.device)
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=2e-4)
        bce = nn.BCELoss()
        for _ in range(self.epochs):
            for s in range(0, N, self.batch_size):
                real = segments[s:s + self.batch_size]
                z = torch.randn(real.size(0), self.latent, device=self.device)
                fake = self.gen(z).detach()
                opt_d.zero_grad()
                (bce(self.disc(real), torch.ones(real.size(0), 1, device=self.device))
                 + bce(self.disc(fake), torch.zeros(real.size(0), 1, device=self.device))
                 ).backward(); opt_d.step()
                z = torch.randn(real.size(0), self.latent, device=self.device)
                opt_g.zero_grad()
                (bce(self.disc(self.gen(z)), torch.ones(real.size(0), 1, device=self.device))
                 ).backward(); opt_g.step()
        return self

    def augment(self, n: int):
        import torch
        with torch.no_grad():
            z = torch.randn(n, self.latent, device=self.device)
            return self.gen(z).cpu().numpy()
