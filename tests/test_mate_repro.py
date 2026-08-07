"""
tests/test_mate_repro.py — Phase 0 gate: MATE environment reproducibility.

Checks that the pinned dependency stack (numpy 1.26.4 + gym 0.21 + MATE
submodule) actually runs a deterministic episode, and that the ground-truth
state channel the Project09 signal relies on is present.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import mate


def test_mate_imports():
    assert mate.__version__ == "0.1.0"
    for rid in ("MultiAgentTracking-v0", "MATE-4v2-9-v0",
                "MATE-4v8-9-v0", "MATE-8v8-9-v0"):
        spec = mate.make(rid)
        spec.close()
        assert spec is not None


def test_episode_deterministic_random():
    # NOTE: gym 0.21's Tuple.seed hangs on numpy 1.26 (np.random.choice with
    # np.iinfo(int).max + replace=False tries to allocate an enormous array).
    # Use a fixed zero joint action instead: still exercises the full env loop
    # and is fully deterministic, which is what the gate checks.
    def run(seed):
        env = mate.make("MultiAgentTracking-v0")
        env.seed(seed)
        sample = env.action_space.sample()
        zero_action = tuple(np.zeros_like(s) for s in sample)
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 200:
            obs, (cam_r, tar_r), done, info = env.step(zero_action)
            total += float(cam_r)
            steps += 1
        env.close()
        return round(total, 6), steps

    r1 = run(seed=0)
    r2 = run(seed=0)
    assert r1 == r2, "same seed must reproduce identical episodes"


def test_state_channel_present():
    """The decision signal reads view mask + entity positions."""
    env = mate.make("MultiAgentTracking-v0")
    env.seed(0)
    obs = env.reset()
    u = env.unwrapped
    n_cam, n_tar = u.num_cameras, u.num_targets
    assert u.camera_target_view_mask.shape == (n_cam, n_tar)
    cam = u.cameras[0]
    tar = u.targets[0]
    assert np.shape(cam.location) == (2,)
    assert np.shape(tar.location) == (2,)
    env.close()


def test_smoke_episode_runs():
    """A short random-policy episode completes with finite reward."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = mate.make("MATE-4v2-9-v0")
        env.seed(42)
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 100:
            obs, (cam_r, tar_r), done, info = env.step(env.action_space.sample())
            total += float(cam_r)
            steps += 1
        env.close()
    assert steps > 0
    assert np.isfinite(total)
