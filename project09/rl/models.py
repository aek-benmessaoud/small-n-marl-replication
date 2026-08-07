"""models.py — Shared-parameter MAPPO networks for the MATE camera team.

All cameras share ONE policy network (parameter sharing, cooperative
team). The critic is centralized: it consumes the concatenation of all
camera observations and estimates the team value.

obs layout (MATE MultiCamera): ndarray (n_cam, obs_dim), obs_dim = 126.
actions: for each camera, a Box(2) in [-1, 1] (joint/position deltas).
"""

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Gaussian policy shared by every camera: obs (batch, obs_dim) -> N(mu, sigma)."""

    def __init__(self, obs_dim, act_dim, hidden=(512, 256), log_std_init=-0.5):
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

    def forward(self, obs):
        return self.net(obs)

    def get_distribution(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0))
        return torch.distributions.Normal(mean, std)

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

    def __init__(self, n_cam, obs_dim, hidden=(512, 256)):
        super().__init__()
        layers = []
        prev = n_cam * obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, team_obs):
        """team_obs: (batch, n_cam * obs_dim) -> (batch, 1)."""
        return self.net(team_obs).squeeze(-1)
