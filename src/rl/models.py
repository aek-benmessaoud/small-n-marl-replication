"""models.py — Shared-parameter MAPPO networks for the MATE camera team.

All cameras share ONE policy network (parameter sharing, cooperative
team). The critic is centralized: it consumes the concatenation of all
camera observations and estimates the team value.

obs layout (MATE MultiCamera): ndarray (n_cam, obs_dim), obs_dim = 126.
actions: for each camera, a Box(2) in [-1, 1] (joint/position deltas).

Observations are statistically normalized per-dimension (fit from sampled
data) BEFORE the networks — MATE obs carry raw coordinates up to ~2000 and
its space bounds contain +-inf, so space-based normalization is impossible.
The normalizer is applied inside the models, so acting, critic evaluation and
the update path are all consistent.
"""

import numpy as np
import torch
import torch.nn as nn


class TanhNormal(torch.distributions.Distribution):
    """Gaussian squashed by tanh into (-1, 1), with the log-det-Jacobian
    correction. Prevents the action-mean saturation collapse seen with a raw
    linear mean + hard clip (mean explodes beyond the box, the clip pins it at
    the boundary, and no gradient escapes)."""

    has_rsample = True

    def __init__(self, loc, scale, eps=1e-6):
        super().__init__(validate_args=False)
        self.base = torch.distributions.Normal(loc, scale, validate_args=False)
        self.eps = eps

    @staticmethod
    def _atanh(x):
        return 0.5 * (torch.log1p(x) + (-torch.log1p(-x)))

    def sample(self, sample_shape=torch.Size()):
        u = self.base.sample(sample_shape)
        return torch.tanh(u)

    def rsample(self, sample_shape=torch.Size()):
        u = self.base.rsample(sample_shape)
        return torch.tanh(u)

    def log_prob(self, value):
        v = value.clamp(-1.0 + self.eps, 1.0 - self.eps)
        u = self._atanh(v)
        logp = self.base.log_prob(u)
        logp = logp - (torch.log(1.0 - v * v) + self.eps)
        return logp

    def entropy(self):
        # proxy: the pre-squash Gaussian entropy
        return self.base.entropy()

    @property
    def mean(self):
        return torch.tanh(self.base.mean)

    @property
    def stddev(self):
        return self.base.stddev


class ObsNormalizer(nn.Module):
    """Per-dimension mean/std normalizer, fit from a batch of observations.
    Identity until fit()."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self._fitted = False

    @property
    def fitted(self):
        return self._fitted

    def fit(self, obs):
        """obs: (N, dim) array/tensor of stacked per-camera observations."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        std[std < 1e-6] = 1.0
        self.mean.copy_(mean)
        self.std.copy_(std)
        self._fitted = True
        return self

    def forward(self, x):
        return (x - self.mean) / self.std


class MLPPolicy(nn.Module):
    """Gaussian policy shared by every camera: obs (batch, obs_dim) -> N(mu, sigma)."""

    def __init__(self, obs_dim, act_dim, hidden=(512, 256), log_std_init=-0.5,
                 normalizer=None):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.normalizer = normalizer or nn.Identity()

    def forward(self, obs):
        return self.net(self.normalizer(obs))

    def get_distribution(self, obs):
        raw = self.net(self.normalizer(obs))
        mean = torch.tanh(raw)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return TanhNormal(mean, std)

    def act(self, obs, deterministic=False):
        """obs: (batch, obs_dim) tensor or ndarray -> (batch, act_dim) actions."""
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            dist = self.get_distribution(obs)
            a = dist.mean if deterministic else dist.sample()
        return a.cpu().numpy()

    def evaluate_actions(self, obs, actions):
        """Team-level evaluation.

        obs     : (batch, n_cam, obs_dim) — flattened internally to
                  (batch*n_cam, obs_dim), since the policy is per-camera.
        actions : (batch, n_cam, act_dim).
        Returns (log_prob, entropy, mean) each of shape (batch,), the joint
        per-step quantities being SUMS over the n_cam cameras.
        """
        batch, n_cam = obs.shape[0], obs.shape[1]
        flat_obs = obs.reshape(batch * n_cam, -1)
        flat_act = actions.reshape(batch * n_cam, -1)
        dist = self.get_distribution(flat_obs)
        logp = dist.log_prob(flat_act).sum(dim=-1).reshape(batch, n_cam)
        ent = dist.entropy().sum(dim=-1).reshape(batch, n_cam)
        mean = dist.mean.reshape(batch, n_cam, -1)
        return logp.sum(dim=-1), ent.sum(dim=-1), mean


class CentralizedCritic(nn.Module):
    """V(s) from the concatenated team observation (n_cam * obs_dim)."""

    def __init__(self, n_cam, obs_dim, hidden=(512, 256), normalizer=None):
        super().__init__()
        layers = []
        prev = n_cam * obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.normalizer = normalizer or nn.Identity()
        self.n_cam = n_cam
        self.obs_dim = obs_dim

    def forward(self, team_obs):
        """team_obs: (batch, n_cam * obs_dim) -> (batch, 1)."""
        batch, _ = team_obs.shape
        x = team_obs.reshape(batch, self.n_cam, self.obs_dim)
        x = self.normalizer(x)          # per-camera, elementwise
        return self.net(x.reshape(batch, -1)).squeeze(-1)
