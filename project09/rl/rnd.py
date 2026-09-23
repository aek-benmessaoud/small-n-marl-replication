"""
rl/rnd.py — Random Network Distillation (Burda et al., 2018) for Project09.

RND provides an intrinsic novelty signal over the team's camera POSE state:

    pose_t = [ (x_c, y_c, theta_c) / (1000, 1000, 180) ] for c in cameras

A frozen random target network f maps pose -> features; a trainable predictor
f_hat is fitted (MSE) on the poses actually visited. The per-camera intrinsic
is the prediction error ||f(x) - f_hat(x)||^2 (mean over output dims); the
team intrinsic is the sum over cameras. Novel states (never/near-never seen)
keep a high error -> exploration bonus; familiar states give ~0.

Unlike the Chao-U shaping (§4.8) the bonus is a NON-STATIONARY potential:
the predictor keeps moving, so the shaping is not a fixed potential difference
and the Ng et al. (1999) invariance does not apply. It also targets COVERAGE
directly (novel camera poses = visiting new parts of the arena) instead of
angular diversity — the only untested family (§7-2).
"""

import numpy as np

import torch
import torch.nn as nn


class RNDTarget(nn.Module):
    """Fixed random feature net (frozen)."""

    def __init__(self, in_dim, out_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class RNDPredictor(nn.Module):
    """Trainable predictor trained to reproduce the target's features."""

    def __init__(self, in_dim, out_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class RND:
    """Target + predictor + optimizer. Deterministic when seeded."""

    def __init__(self, in_dim=3, out_dim=32, hidden=64, lr=1e-3, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.target = RNDTarget(in_dim, out_dim, hidden)
        self.predictor = RNDPredictor(in_dim, out_dim, hidden)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.out_dim = out_dim

    @staticmethod
    def _t(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32))

    def error(self, states):
        """Per-state prediction error: mean over output dims of (t - p)^2.
        states: (N, in_dim) -> (N,) float array."""
        x = self._t(states)
        with torch.no_grad():
            t = self.target(x)
            p = self.predictor(x)
        return ((t - p) ** 2).mean(dim=1).numpy().astype(np.float64)

    def update(self, states, n_steps=1):
        """Train the predictor on the visited states. Returns the mean loss."""
        x = self._t(states)
        with torch.no_grad():
            t = self.target(x).detach()
        total = 0.0
        for _ in range(n_steps):
            self.optimizer.zero_grad()
            loss = ((self.predictor(x) - t) ** 2).mean()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
        return total / max(n_steps, 1)
