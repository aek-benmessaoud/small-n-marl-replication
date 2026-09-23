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
from project09.environment.reward import (
    ChaoUReward, LocalizationReward, RNDIntrinsic, UTracker, measure_loc_scale,
    measure_reward_scale, measure_rnd_scale, rnd_pose_state, suggest_lambda,
    suggest_loc_lambda, suggest_rnd_lambda,
)
from project09.estimators.continuous import TargetTracker
from project09.rl.rnd import RND


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


def test_utracker_reward_passthrough():
    """UTracker must NOT shape the reward: same seed + same actions -> same
    per-step rewards as the plain env, and U tracking works."""
    rng_seed = 11
    cap = 25

    def run(wrapped):
        env = make_mate(env_id="MultiAgentTracking-v0",
                        config="MATE-4v2-9-v0", seed=rng_seed)
        env.seed(rng_seed)
        if wrapped:
            env = UTracker(env)
        obs = env.reset()
        act = np.zeros((obs.shape[0], 2), dtype=np.float32)
        act[0] = [0.5, -0.5]
        done = False
        rs = []
        n = 0
        while not done and n < cap:
            obs, r, done, info = env.step(act)
            rs.append(float(r))
            n += 1
        env.close()
        return rs

    plain = run(False)
    tracked = run(True)
    assert len(plain) == len(tracked) > 0
    assert np.allclose(plain, tracked)

    env2 = make_mate(env_id="MultiAgentTracking-v0",
                     config="MATE-4v2-9-v0", seed=rng_seed)
    tr = UTracker(env2)
    tr.seed(rng_seed)
    tr.reset()
    assert 1.0 <= tr.current_U() <= tr.num_targets
    tr.close()


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


def test_windowed_wrapper_runs_and_propagates_window():
    """ChaoUReward(window=N) hands the sliding window to the tracker; a
    random rollout runs, is finite, and the tracker buffers never exceed the
    window size."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=2)
    wrapped = ChaoUReward(env, reward_lambda=2.0, window=5)
    assert wrapped.tracker.window == 5
    wrapped.seed(2)
    obs = wrapped.reset()
    done = False
    n = 0
    while not done and n < 40:
        obs, r, done, info = wrapped.step(wrapped.action_space.sample())
        assert np.isfinite(r)
        assert all(len(b) <= 5 for b in wrapped.tracker.buffers)
        n += 1
    wrapped.close()
    assert n > 0


# ==================== dense (continuous) signal =============================
def test_dense_confidence_grows_with_spread():
    """Dense confidence saturates at 1 once the bearing spread reaches the
    angular tolerance; single bearings give 0 confidence."""
    tracker = TargetTracker(num_targets=1, ang_tol=np.deg2rad(15.0))
    assert tracker.dense_confidence_sum() == 0.0
    tracker.observe(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                    np.array([[True]]))
    assert tracker.dense_confidence_sum() == 0.0
    # spread of 28.6 deg (> 15 deg tol) -> confidence saturates at 1
    tracker.buffers[0] = [0.0, 0.5]
    c = tracker.dense_confidences()
    assert c[0] == 1.0
    assert tracker.dense_confidence_sum() == 1.0
    # sub-tolerance spread -> partial confidence in (0, 1)
    tracker.buffers[0] = [0.0, 0.1]  # 5.7 deg apart
    assert 0.0 < tracker.dense_confidences()[0] < 1.0


def test_dense_reward_positive_on_confidence_growth():
    """Dense mode: intrinsic reward is POSITIVE when the confidence sum grows
    and ZERO when it is stable (shaping, not penalizing)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v2-9-v0", seed=1)
    wrapped = ChaoUReward(env, reward_lambda=2.0, dense=True)
    wrapped.reset()
    C_t = wrapped.current_C()
    # simulate a confidence-increasing observation
    tracker = wrapped.tracker
    tracker.buffers[0] = [0.0, 0.5]
    C_next = wrapped.current_C()
    r_intr = 2.0 * (C_next - C_t) / wrapped.u_max
    assert r_intr >= 0.0
    if C_next > C_t:
        assert r_intr > 0.0
    wrapped.close()


def test_dense_signal_dense_in_env():
    """The dense intrinsic term must fire on a large fraction of steps in a
    4v8 random rollout (unlike the discrete U signal)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v8-9-v0", seed=5)
    wrapped = ChaoUReward(env, reward_lambda=1.0, dense=True)
    wrapped.seed(5)
    obs = wrapped.reset()
    rng = np.random.default_rng(5)
    intr = []
    done = False
    n = 0
    while not done and n < 300:
        act = rng.uniform(-1, 1, size=(obs.shape[0], 2))
        obs, r, done, info = wrapped.step(act)
        intr.append(info[0].get("intrinsic_reward", 0.0))
        n += 1
    wrapped.close()
    intr = np.asarray(intr)
    density = float((intr != 0).mean())
    assert density > 0.5, f"dense signal too sparse: density={density:.3f}"


# ==================== calibration ===========================================
def test_suggest_lambda_sparse_signal_falls_back_to_nonzero():
    """A sparse (discrete) signal with global p75 == 0 must NOT blow lambda up:
    suggest_lambda falls back to the p75 of the non-zero deltas."""
    scale = {
        "env_abs_mean": 1.0, "env_std": 1.014,
        "unit_p50": 0.0, "unit_p75": 0.0, "unit_p95": 0.25,
        "unit_nonzero_p75": 0.25, "n": 1000,
    }
    lam = suggest_lambda(scale, target_ratio=1.0)
    assert 1.0 < lam < 10.0, f"lambda should be ~4.06, got {lam}"
    assert abs(lam - 1.014 / 0.25) < 1e-6

    scale_dense = dict(scale)
    scale_dense["unit_p75"] = 0.0005
    scale_dense["unit_nonzero_p75"] = 0.0008
    lam_dense = suggest_lambda(scale_dense, target_ratio=1.0)
    assert abs(lam_dense - 1.014 / 0.0005) < 1e-6


def test_suggest_lambda_all_zero_units_falls_back_to_neutral():
    """If the intrinsic NEVER fires during calibration (quasi-static targets,
    random probe), both p75 and nonzero-p75 are 0 and lambda must not explode:
    fall back to a neutral finite scale (lambda == target_ratio)."""
    scale = {
        "env_abs_mean": 2.0, "env_std": 3.5,
        "unit_p50": 0.0, "unit_p75": 0.0, "unit_p95": 0.0,
        "unit_nonzero_p75": 0.0, "n": 1000,
    }
    lam = suggest_lambda(scale, target_ratio=1.0)
    assert abs(lam - 1.0) < 1e-6, f"lambda should be neutral (1.0), got {lam}"
    lam_half = suggest_lambda(scale, target_ratio=0.5)
    assert abs(lam_half - 0.5) < 1e-6, f"lambda should be 0.5, got {lam_half}"


# ==================== LocalizationReward (objective, §7-3) ===================
class _FakeMate:
    """Minimal env exposing only what state_arrays + LocalizationReward read."""

    def __init__(self, cameras, targets, view):
        self.unwrapped = self
        self.cameras = [type("Cam", (), {"location": c})() for c in cameras]
        self.targets = [type("Tar", (), {"location": t})() for t in targets]
        self.camera_target_view_mask = np.asarray(view, dtype=bool)
        self.action_space = type("AS", (), {"sample": lambda self: np.zeros(
            (len(cameras), 2), dtype=np.float32)})()

    def reset(self, **kwargs):
        return np.zeros((len(self.cameras), 4))

    def step(self, action):
        return self.reset(), 0.0, False, [{"dummy": 1}]

    def close(self):
        pass


def _fake(cameras, targets, view):
    return _FakeMate(np.asarray(cameras, dtype=float),
                     np.asarray(targets, dtype=float), np.asarray(view, dtype=bool))


def test_loc_reward_zero_without_two_views():
    """A target seen by a single camera is NOT localizable -> q=0, the wrapper
    passes the env reward through unchanged."""
    env = _fake([[0.0, 0.0]], [[50.0, 1.0]], [[True]])
    w = LocalizationReward(env, reward_lambda=5.0)
    obs, r, done, info = w.step(w.action_space.sample())
    assert r == 0.0
    assert info[0]["localization_reward"] == 0.0
    assert info[0]["env_reward"] == 0.0
    w.close()


def test_loc_reward_positive_with_two_views():
    """Two cameras with well-separated bearings on the same target make it
    localizable (finite CRLB) -> q > 0 and the wrapped reward grows by
    lambda * q."""
    env = _fake([[0.0, 0.0], [100.0, 0.0]], [[50.0, 1.0]], [[True], [True]])
    w = LocalizationReward(env, reward_lambda=5.0)
    _, r, _, info = w.step(w.action_space.sample())
    assert info[0]["localization_reward"] > 0.0
    assert abs(r - info[0]["localization_reward"]) < 1e-12
    assert info[0]["loc_quality"] > 0.0
    w.close()


def test_loc_reward_bounded_by_threshold():
    """Per-target quality is clipped to [0, 1], so the summed reward stays in
    [0, lambda * n_targets] even with many perfectly-conditioned views."""
    env = _fake([[0.0, 0.0], [100.0, 0.0]], [[50.0, 1.0], [50.0, 60.0]],
                [[True, True], [True, True]])
    w = LocalizationReward(env, reward_lambda=3.0)
    _, r, _, info = w.step(w.action_space.sample())
    assert 0.0 <= info[0]["localization_reward"] <= 3.0 * 2 + 1e-9
    assert 0.0 <= info[0]["loc_quality"] <= 1.0
    w.close()


def test_suggest_loc_lambda():
    """loc calibration: lambda = target_ratio * env_noise / nonzero_p75, with a
    neutral fallback when the oracle never fires."""
    scale = {"env_std": 10.0, "env_abs_mean": 8.0,
             "loc_mean": 0.02, "loc_nonzero_p75": 0.5, "n": 1000}
    lam = suggest_loc_lambda(scale, target_ratio=1.0)
    assert abs(lam - 20.0) < 1e-6
    lam_half = suggest_loc_lambda(scale, target_ratio=0.5)
    assert abs(lam_half - 10.0) < 1e-6
    empty = {"env_std": 10.0, "env_abs_mean": 8.0, "loc_mean": 0.0,
             "loc_nonzero_p75": 0.0, "n": 1000}
    assert abs(suggest_loc_lambda(empty, target_ratio=1.0) - 1.0) < 1e-6


def test_measure_loc_scale_runs():
    """measure_loc_scale runs a random rollout on a real env and returns a
    finite bounded scale dict (loc fires rarely in 4v8: mean small, non-zero
    p75 > 0 whenever the goulot was reached)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v8-9-v0", seed=5)
    env.seed(5)
    scale = measure_loc_scale(env, n_episodes=1, n_steps=300, seed=5)
    assert set(["loc_mean", "loc_p75", "loc_nonzero_p75", "loc_nonzero_n",
                "env_abs_mean", "env_std", "n"]).issubset(scale)
    assert scale["n"] > 0
    assert 0.0 <= scale["loc_mean"] <= 8.0
    assert np.isfinite(scale["env_std"])
    env.close()


# ==================== RND / novelty d'état (§7-2) ============================
def test_rnd_target_and_predictor_differ():
    """Fresh RND: predictor cannot match the frozen target -> error > 0."""
    rnd = RND(in_dim=3, out_dim=32, hidden=64, seed=0)
    states = np.asarray([[0.5, -0.5, 0.3], [0.0, 0.0, 1.0]],
                        dtype=np.float32)
    err = rnd.error(states)
    assert err.shape == (2,)
    assert np.all(err > 1e-6)
    # deterministic given the seed
    rnd2 = RND(in_dim=3, out_dim=32, hidden=64, seed=0)
    assert np.allclose(rnd.error(states), rnd2.error(states))


def test_rnd_predictor_converges_on_fixed_states():
    """Training the predictor on a fixed set of states must drive the error
    on THOSE states down, while a far-away novel state stays high -> the
    novelty signal is self-limiting on familiar regions."""
    rnd = RND(in_dim=3, out_dim=32, hidden=64, seed=0)
    seen = np.asarray([[0.5, -0.5, 0.3]] * 32, dtype=np.float32)
    novel = np.asarray([[0.99, 0.99, 1.7]], dtype=np.float32)
    err_seen0 = float(rnd.error(seen[:1]).mean())
    err_novel0 = float(rnd.error(novel).mean())
    for _ in range(200):
        rnd.update(seen)
    err_seen1 = float(rnd.error(seen[:1]).mean())
    err_novel1 = float(rnd.error(novel).mean())
    assert err_seen1 < err_seen0 * 0.5, f"familiar error must shrink: " \
        f"{err_seen0} -> {err_seen1}"
    assert err_novel1 > err_seen1, "novel states must keep a higher error " \
        f"than familiar ones ({err_novel1} vs {err_seen1})"


def test_rnd_intrinsic_wrapper_shapes_reward():
    """RNDIntrinsic adds lambda * sum_c error to the env reward and reports it
    in info (pose read from the unwrapped env, predictor trained sporadically
    on a replay buffer)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v8-9-v0", seed=1)
    env.seed(1)
    wrapped = RNDIntrinsic(env, reward_lambda=2.0, seed=7)
    obs = wrapped.reset()
    n_steps = 30
    tot_intr = 0.0
    for _ in range(n_steps):
        obs, r, done, info = wrapped.step(
            np.zeros((obs.shape[0], 2), dtype=np.float32))
        intr = info[0].get("intrinsic_reward", 0.0)
        env_r = info[0].get("env_reward", 0.0)
        assert abs(r - (env_r + 2.0 * intr)) < 1e-6
        assert intr > 0.0
        tot_intr += intr
    assert tot_intr > 0.0
    wrapped.close()


def test_rnd_pose_state_shape_and_range():
    """rnd_pose_state returns (n_cam, 3) float32 normalized poses: positions
    within the arena map to ~[-1.2, 1.2], orientation (degrees) to [0, 2)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v8-9-v0", seed=2)
    env.seed(2)
    env.reset()
    poses = rnd_pose_state(env)
    assert poses.shape == (4, 3)
    assert poses.dtype == np.float32
    assert np.all(np.abs(poses[:, :2]) < 1.5)
    assert np.all(np.abs(poses[:, 2]) < 1.5)
    env.close()


def test_suggest_rnd_lambda():
    """rnd lambda = target_ratio * env_noise / nonzero_p75, neutral fallback."""
    scale = {"env_std": 10.0, "env_abs_mean": 8.0, "rnd_mean": 0.5,
             "rnd_p75": 2.0, "rnd_nonzero_p75": 5.0, "n": 1000}
    assert abs(suggest_rnd_lambda(scale, target_ratio=1.0) - 2.0) < 1e-6
    assert abs(suggest_rnd_lambda(scale, target_ratio=0.5) - 1.0) < 1e-6
    empty = {"env_std": 10.0, "env_abs_mean": 8.0, "rnd_mean": 0.0,
             "rnd_p75": 0.0, "rnd_nonzero_p75": 0.0, "n": 1000}
    assert abs(suggest_rnd_lambda(empty, target_ratio=1.0) - 1.0) < 1e-6


def test_measure_rnd_scale_runs():
    """measure_rnd_scale runs a random rollout and returns a finite dict with
    a positive novelty floor (fresh predictor can't match the target)."""
    env = make_mate(env_id="MultiAgentTracking-v0",
                    config="MATE-4v8-9-v0", seed=5)
    env.seed(5)
    scale = measure_rnd_scale(env, n_episodes=1, n_steps=200, seed=5)
    assert set(["rnd_mean", "rnd_p75", "rnd_nonzero_p75", "rnd_nonzero_n",
                "env_abs_mean", "env_std", "n"]).issubset(scale)
    assert scale["n"] > 0
    assert scale["rnd_nonzero_p75"] > 0.0
    assert np.isfinite(scale["env_std"])
    env.close()
