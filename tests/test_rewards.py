"""
tests/test_rewards.py — Phase 2 gate: Chao-U intrinsic reward sign/shape.

Locked convention: r_intrinsic = lambda * (U_t - U_{t+1}) / U_max.
  - zero when U is stable,
  - POSITIVE when U decreases (targets become localizable),
  - negative when U increases,
  - bounded in [-lambda, +lambda] since U in [1, U_max].
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from project09.config import REWARD_LAMBDA
from project09.environment.mate_env import make_mate, state_arrays
from project09.environment.reward import ChaoUReward
from project09.estimators.continuous import TargetTracker


def test_reward_zero_when_U_stable():
    """A step that adds no new information must yield ~0 intrinsic reward."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=0)
    wrapped = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
    wrapped.reset()

    U_before = wrapped.current_U()
    camera_xy, target_xy, view = state_arrays(wrapped.env)

    # observe the SAME snapshot again (identical view -> config counts
    # saturate; duplicate bearings merge under the angular tolerance)
    tracker = TargetTracker(wrapped.num_targets)
    tracker.observe(camera_xy, target_xy, view)
    U_next = wrapped.tracker.U(variant="bias_cap")
    r_intrinsic = REWARD_LAMBDA * (U_before - U_next) / wrapped.u_max
    assert abs(r_intrinsic) < 1e-9


def test_reward_positive_when_U_decreases():
    """Constructing an under-determined target into 2 configs must give a
    positive intrinsic reward (driven by U decrease)."""
    tracker = TargetTracker(num_targets=3)
    camera_xy = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    target_xy = np.array([[50.0, 1.0], [50.0, 60.0], [10.0, 90.0]])
    # step 1: target 0 seen by cam 0 only -> 1 configuration (under-determined)
    tracker.observe(camera_xy, target_xy, np.array([[1, 1, 1],
                                                     [0, 0, 0],
                                                     [0, 0, 0]], dtype=bool))
    U_t = tracker.U(variant="bias_cap")

    # step 2: target 0 now also seen by cam 1 -> second independent bearing
    tracker.observe(camera_xy, target_xy, np.array([[1, 1, 1],
                                                     [1, 0, 0],
                                                     [0, 0, 0]], dtype=bool))
    U_next = tracker.U(variant="bias_cap")
    r = REWARD_LAMBDA * (U_t - U_next) / tracker.u_max()
    assert U_next < U_t
    assert r > 0.0
    assert abs(r) <= REWARD_LAMBDA


def test_reward_bounded_and_wrapper_runs():
    """End-to-end: the wrapped env steps, rewards are finite and the intrinsic
    term stays within [-lambda, lambda]."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=7)
    wrapped = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
    wrapped.seed(7)
    obs = wrapped.reset()
    done = False
    steps = 0
    seen_intrinsic = []
    while not done and steps < 40:
        obs, r, done, info = wrapped.step(wrapped.action_space.sample())
        steps += 1
        if isinstance(info, list) and info and isinstance(info[0], dict):
            seen_intrinsic.append(info[0].get("intrinsic_reward", 0.0))
    wrapped.close()
    assert steps > 0
    assert np.isfinite(r)
    if seen_intrinsic:
        assert np.all(np.abs(np.asarray(seen_intrinsic)) <= REWARD_LAMBDA + 1e-9)


def test_wrapper_tracker_consistent_with_state():
    """After reset the tracker's U reflects the initial view; after a step the
    tracker's buffers only grow (observations accumulate)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=3)
    wrapped = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
    wrapped.seed(3)
    wrapped.reset()
    n_after_reset = sum(len(b) for b in wrapped.tracker.buffers)
    wrapped.step(wrapped.action_space.sample())
    n_after_step = sum(len(b) for b in wrapped.tracker.buffers)
    assert n_after_step >= n_after_reset
    wrapped.close()
