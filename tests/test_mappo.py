"""
tests/test_mappo.py — Phase 3 gate: MAPPO save/resume + smoke train.

Short training on MATE-4v2-9-v0 (4 cameras, 2 targets) verifying the
checkpoint/resume cycle, finite losses, and deterministic eval acting.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from project09.config import REWARD_LAMBDA
from project09.environment.mate_env import make_mate
from project09.environment.reward import ChaoUReward
from project09.rl.mappo import MAPPO


def _make_env(seed=0):
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=seed)
    env = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
    env.seed(seed)
    return env


def test_mappo_act_shapes():
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0)
    a, logp = mappo.act_batch(obs)
    assert a.shape == (n_cam, 2)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    assert np.isscalar(logp) or np.ndim(logp) == 0
    v = mappo.critic_value(obs)
    assert np.isfinite(v)
    env.close()


def test_mappo_train_resume(tmp_path):
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    ckpt = os.path.join(tmp_path, "ckpts")

    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0,
                  minibatch=8, epochs=2)
    total = mappo.train(env, max_total_steps=60, max_steps_per_episode=60,
                        horizon=60, checkpoint_dir=ckpt, checkpoint_interval=30)
    assert total >= 60
    assert os.path.exists(os.path.join(ckpt, "latest.pt"))

    # resume into a fresh instance
    mappo2 = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=1,
                   minibatch=8, epochs=2)
    ck = mappo2.load(os.path.join(ckpt, "latest.pt"))
    assert ck["step_count"] == mappo.step_count
    assert ck["step_count"] > 0
    mappo2.train(env, max_total_steps=mappo.step_count + 30,
                 max_steps_per_episode=60, horizon=60,
                 checkpoint_dir=ckpt, checkpoint_interval=30)
    env.close()


def test_mappo_deterministic_eval():
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0)
    a1, _ = mappo.act_batch(obs, deterministic=True)
    a2, _ = mappo.act_batch(obs, deterministic=True)
    assert np.allclose(a1, a2)
    env.close()


def test_mappo_update_finite():
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0,
                  minibatch=8, epochs=2)
    buffer = mappo.collect_rollout(env, horizon=40, max_steps=40)
    pg, vf = mappo.update(buffer)
    assert np.isfinite(pg) and np.isfinite(vf)
    env.close()


def test_obs_normalizer_fit():
    """fit_obs_normalizer must produce per-dim stats and keep acting finite."""
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0)
    norm = mappo.fit_obs_normalizer(env, samples=64, seed=0)
    assert norm.fitted
    assert np.all(np.isfinite(norm.std.cpu().numpy()))
    assert np.all(norm.std.cpu().numpy() > 0)
    a, _ = mappo.act_batch(obs)
    assert np.all(np.isfinite(a))
    env.close()


def test_mappo_stability_no_nan():
    """Regression: normalized obs must keep training finite (previous run
    diverged to NaN at ~13k steps with raw obs)."""
    env = _make_env()
    obs = env.reset()
    n_cam, obs_dim = obs.shape
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam, seed=0,
                  minibatch=16, epochs=2)
    mappo.fit_obs_normalizer(env, samples=128, seed=0)
    steps = 0
    while steps < 800:
        buffer = mappo.collect_rollout(env, horizon=100, max_steps=100)
        mappo.update(buffer)
        steps += len(buffer[0])
        assert np.all(np.isfinite(
            mappo.policy.log_std.detach().cpu().numpy())), "policy log_std NaN"
        for name, p in mappo.policy.named_parameters():
            assert np.all(np.isfinite(p.detach().cpu().numpy())), \
                f"policy param {name} NaN"
    env.close()
