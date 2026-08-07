"""
environment/reward.py — Chao-U intrinsic reward wrapper.

Wraps a MATE single-team environment (mate.MultiCamera output) and shapes the
per-step camera reward:

    r = coverage_reward + REWARD_LAMBDA * (U_t - U_{t+1}) / U_max

where U is the target-centric Chao-U remaining-work estimate and U_max is the
number of targets. The intrinsic term is POSITIVE when U decreases (targets
become localizable) and ZERO when U is stable — the corrected sign convention
(see config.py; the charter's U_{t+1} - U_t is inverted and NOT used).

The wrapper is leak-free by construction: it reads only what a central
observer could read (the environment's view mask + entity positions), i.e. the
team's joint observation channel. The CRLB oracle in estimators/validation.py
is EVALUATION-only and never feeds the decision signal.

Step order (locked):
  1. U_t = tracker.U()   (state BEFORE the action's effect)
  2. obs, r_env, done, info = env.step(action)   (env advances)
  3. tracker.observe(...)  (new view -> buffers updated)
  4. U_{t+1} = tracker.U()
  5. r_intrinsic = lambda * (U_t - U_{t+1}) / U_max
  6. r_shaped = r_env + r_intrinsic

On reset, the tracker is cleared; the reset-state view is observed so that U
at step 0 already reflects the initial configuration (matches how MAPPO rolls
out: obs from reset, then act, then step).
"""

import numpy as np

import gym

from project09.config import REWARD_LAMBDA
from project09.environment.mate_env import state_arrays
from project09.estimators.continuous import TargetTracker


class ChaoUReward(gym.Wrapper):
    """Add the Chao-U intrinsic term to the team camera reward."""

    def __init__(self, env, reward_lambda=REWARD_LAMBDA, variant="bias_cap",
                 ang_tol=None, cluster_cap=None):
        super().__init__(env)
        u = env.unwrapped
        self.num_targets = int(u.num_targets)
        self.reward_lambda = float(reward_lambda)
        self.variant = variant
        self.tracker = TargetTracker(
            self.num_targets, ang_tol=ang_tol, cluster_cap=cluster_cap)
        self.U_prev = None
        self.u_max = self.tracker.u_max()

    # ---- public diagnostics (used by evaluation/logger) --------------------
    def current_U(self):
        return self.tracker.U(variant=self.variant)

    def current_bundle(self):
        return (self.tracker.visit(), self.tracker.known(),
                self.tracker.obs(), self.tracker.underdetermined())

    def current_components(self):
        return self.tracker.components()

    # ---- gym API -----------------------------------------------------------
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.tracker.reset()
        camera_xy, target_xy, view = state_arrays(self.env)
        self.tracker.observe(camera_xy, target_xy, view)
        self.U_prev = self.tracker.U(variant=self.variant)
        return obs

    def step(self, action):
        U_t = self.U_prev
        obs, r_env, done, info = self.env.step(action)

        camera_xy, target_xy, view = state_arrays(self.env)
        self.tracker.observe(camera_xy, target_xy, view)
        U_next = self.tracker.U(variant=self.variant)

        r_intrinsic = (self.reward_lambda * (U_t - U_next) / self.u_max
                       if U_t is not None else 0.0)
        r_shaped = float(r_env) + float(r_intrinsic)

        self.U_prev = U_next

        if isinstance(info, (list, tuple)) and info:
            info = list(info)
            for d in info:
                if isinstance(d, dict):
                    d.setdefault("U", float(U_next))
                    d.setdefault("intrinsic_reward", float(r_intrinsic))
                    d.setdefault("F1", float(self.tracker.components()[0]))
                    d.setdefault("F2", float(self.tracker.components()[1]))

        return obs, r_shaped, done, info
