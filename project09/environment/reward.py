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
from collections import deque

from project09.config import (  # noqa: E402
    QUALITY_SIGMA_BEARING_DEG, QUALITY_THRESHOLD, REWARD_LAMBDA,
)
from project09.environment.mate_env import state_arrays  # noqa: E402
from project09.estimators.continuous import TargetTracker  # noqa: E402
from project09.estimators.validation import per_target_bounds  # noqa: E402
from project09.rl.rnd import RND  # noqa: E402


# Camera-pose normalization for RND (MATE angles are in DEGREES; arena ~[-1000, 1000]).
def rnd_pose_state(env):
    """Per-camera normalized pose state for RND: (x/1000, y/1000, theta/180).
    Returns (num_cameras, 3) float32 array from the ground-truth state."""
    u = env.unwrapped
    poses = np.asarray([[c.location[0], c.location[1], c.orientation]
                        for c in u.cameras], dtype=np.float64)
    poses[:, 0] /= 1000.0
    poses[:, 1] /= 1000.0
    poses[:, 2] /= 180.0
    return poses.astype(np.float32)


class ChaoUReward(gym.Wrapper):
    """Add the Chao-U intrinsic term to the team camera reward.

    Two shaping modes (``dense``):
      * dense=False (default): r_intrinsic = lambda*(U_t - U_{t+1})/U_max from
        the discrete Chao-U remaining-work estimate. POSITIVE when U decreases.
      * dense=True: r_intrinsic = lambda*(C_{t+1} - C_t)/U_max from the
        continuous per-target circular variance C = sum_t Var_circ(bearings_t),
        which responds to EVERY new bearing (density ~90% vs ~0.5% discrete).
        POSITIVE when the angular diversity of the recorded bearings grows.

    ``window`` (steps) switches the tracker to sliding-window mode: only
    bearings recorded in the last `window` steps survive (older ones are
    evicted), so U reflects the SHORT-TIMESCALE angular diversity — compatible
    with moving targets instead of the episode-cumulative model. window=None
    (default) keeps the episode-cumulative semantics.
    """

    def __init__(self, env, reward_lambda=REWARD_LAMBDA, variant="bias_cap",
                 ang_tol=None, cluster_cap=None, dense=False, dense_window=None,
                 r_intr_clip=None, window=None, n_bins=None):
        super().__init__(env)
        u = env.unwrapped
        self.num_targets = int(u.num_targets)
        self.reward_lambda = float(reward_lambda)
        self.variant = variant
        self.dense = bool(dense)
        self.dense_window = dense_window
        self.r_intr_clip = (None if r_intr_clip is None
                            else float(r_intr_clip))
        self.window = (None if window is None else int(window))
        self.tracker = TargetTracker(
            self.num_targets, ang_tol=ang_tol, cluster_cap=cluster_cap,
            window=self.window, n_bins=n_bins)
        self.U_prev = None
        self.u_max = self.tracker.u_max()

    # ---- public diagnostics (used by evaluation/logger) --------------------
    def current_U(self):
        return self.tracker.U(variant=self.variant)

    def current_C(self):
        return self.tracker.variance_confidence_sum(window=self.dense_window)

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
        if self.dense:
            self.U_prev = self.current_C()
        else:
            self.U_prev = self.tracker.U(variant=self.variant)
        return obs

    def step(self, action):
        U_t = self.U_prev
        obs, r_env, done, info = self.env.step(action)

        camera_xy, target_xy, view = state_arrays(self.env)
        self.tracker.observe(camera_xy, target_xy, view)
        if self.dense:
            U_next = self.current_C()
            r_intrinsic = (self.reward_lambda * (U_next - U_t) / self.u_max
                           if U_t is not None else 0.0)
        else:
            U_next = self.tracker.U(variant=self.variant)
            r_intrinsic = (self.reward_lambda * (U_t - U_next) / self.u_max
                           if U_t is not None else 0.0)
        if self.r_intr_clip is not None:
            r_intrinsic = float(np.clip(r_intrinsic,
                                        -self.r_intr_clip, self.r_intr_clip))
        r_shaped = float(r_env) + float(r_intrinsic)

        self.U_prev = U_next

        if isinstance(info, (list, tuple)) and info:
            info = list(info)
            for d in info:
                if isinstance(d, dict):
                    d.setdefault("U", float(self.tracker.U(variant=self.variant)))
                    d.setdefault("intrinsic_reward", float(r_intrinsic))
                    if self.dense:
                        d.setdefault("C", float(U_next))
                    d.setdefault("F1", float(self.tracker.components()[0]))
                    d.setdefault("F2", float(self.tracker.components()[1]))

        return obs, r_shaped, done, info


class LocalizationReward(gym.Wrapper):
    """Add an EXPLICIT localization objective to the env reward (option §7-3).

        r = r_mate + REWARD_LAMBDA * sum_t q_t
        q_t = clip(QUALITY_THRESHOLD / bound_t, 0, 1)

    where bound_t is the INSTANTANEOUS CRLB bound of target t (bearing-only
    Fisher information from its current observing cameras, estimators/
    validation.py). q_t = 0 unless the target has >= 2 well-separated views
    (finite bound); q_t saturates to 1 when bound <= QUALITY_THRESHOLD and
    decays smoothly (THRESHOLD / bound) for worse geometry, giving a learning
    gradient toward multi-view, well-conditioned configurations.

    Unlike the Chao-U shaping (potential-based, §4.8 — the optimal policy is
    invariant), this term is a DIRECT additive objective: it CHANGES the optimal
    policy. The coverage <-> localization trade-off it induces is exactly what
    the campaign measures. The wrapper reads the joint state via state_arrays —
    the same leak-free channel the intrinsic tracker uses.
    """

    def __init__(self, env, reward_lambda=REWARD_LAMBDA,
                 quality_sigma_deg=QUALITY_SIGMA_BEARING_DEG,
                 quality_threshold=QUALITY_THRESHOLD):
        super().__init__(env)
        self.reward_lambda = float(reward_lambda)
        self.quality_sigma_deg = quality_sigma_deg
        self.quality_threshold = float(quality_threshold)

    def _loc_quality(self):
        """Per-step localization quality: (sum_t q_t, mean_t q_t)."""
        cam, tar, view = state_arrays(self.env)
        bounds = per_target_bounds(cam, tar, view,
                                   sigma_deg=self.quality_sigma_deg)
        q = np.where(np.isfinite(bounds) & (bounds > 0.0),
                     self.quality_threshold / bounds, 0.0)
        q = np.clip(q, 0.0, 1.0)
        return float(q.sum()), float(q.mean()) if q.size else 0.0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_env, done, info = self.env.step(action)
        loc, q_mean = self._loc_quality()
        r = float(r_env) + self.reward_lambda * loc
        if isinstance(info, (list, tuple)) and info:
            info = list(info)
            for d in info:
                if isinstance(d, dict):
                    d.setdefault("localization_reward",
                                 float(self.reward_lambda * loc))
                    d.setdefault("loc_quality", q_mean)
                    d.setdefault("env_reward", float(r_env))
        return obs, r, done, info


class RNDIntrinsic(gym.Wrapper):
    """Add a Random Network Distillation novelty bonus to the team reward.

        r = r_env + REWARD_LAMBDA * sum_c error(pose_c)

    where error is the RND predictor's prediction error on the camera pose
    (position + orientation, normalized) — high for states the team has never
    (or rarely) visited, ~0 for familiar ones. The novelty is a NON-STATIONARY
    potential (§4.8 does NOT apply) and targets COVERAGE directly (novel poses
    = new parts of the arena), the only untested intrinsic family (§7-2).

    The predictor is NOT updated every step on the current pose (that
    memorizes the slowly-moving trajectory and collapses the error to ~0).
    Instead it is trained sporadically on random minibatches from a replay
    buffer of past poses, so it LAGS the exploration: familiar poses keep a
    low error while genuinely novel poses keep a high one (the mechanism is
    validated by test_rnd_predictor_converges_on_fixed_states). The pose is
    read from the ground-truth state (env.unwrapped) — the same leak-free
    channel the intrinsic tracker uses; the policy never observes it.
    """

    def __init__(self, env, reward_lambda=REWARD_LAMBDA, seed=None,
                 update_every=16, buffer_size=400, batch_size=128,
                 in_dim=3, out_dim=32, hidden=64, lr=1e-3):
        super().__init__(env)
        self.reward_lambda = float(reward_lambda)
        self.rnd = RND(in_dim=in_dim, out_dim=out_dim, hidden=hidden, lr=lr,
                       seed=seed)
        self.update_every = int(update_every)
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.buffer = deque(maxlen=self.buffer_size)
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        self.buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_env, done, info = self.env.step(action)
        poses = rnd_pose_state(self.env)
        err = self.rnd.error(poses)
        intr = float(err.sum())
        r = float(r_env) + self.reward_lambda * intr
        self._step += 1
        self.buffer.extend(poses.tolist())
        if (self._step % self.update_every == 0
                and len(self.buffer) >= self.batch_size):
            idx = np.random.default_rng(self._step).integers(
                0, len(self.buffer), self.batch_size)
            batch = np.asarray([self.buffer[i] for i in idx],
                               dtype=np.float32)
            self.rnd.update(batch, n_steps=1)
        if isinstance(info, (list, tuple)) and info:
            info = list(info)
            for d in info:
                if isinstance(d, dict):
                    d.setdefault("intrinsic_reward", intr)
                    d.setdefault("env_reward", float(r_env))
        return obs, r, done, info


class ClipEnvReward(gym.Wrapper):
    """Clip the ENV reward component before any intrinsic shaping.

    The MATE tracking reward is a small controllable tracking bonus (+1 per
    tracked target per step, range [0, n_targets]) plus large negative spikes
    when a target delivers cargo (exogenous timing, magnitude up to -500+).
    These spikes dominate the GAE advantages and drown the learnable tracking
    signal, so MAPPO stays at the trivial constant-action level. Clipping to
    [-n_targets, n_targets] keeps the tracking signal and removes the
    exogenous delivery noise from the LEARNING signal; evaluation always uses
    the un-clipped reward (UTracker on a raw env).
    """

    def __init__(self, env, low=-2.0, high=2.0):
        super().__init__(env)
        self.low = float(low)
        self.high = float(high)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return obs, float(np.clip(r, self.low, self.high)), done, info


class UTracker(gym.Wrapper):
    """Track the Chao-U remaining-work estimate WITHOUT shaping the reward.

    Same tracker semantics as ChaoUReward but the env reward is passed through
    unchanged. Used by the no-intrinsic ablation arm so that both arms report
    comparable U / localization metrics during evaluation.
    """

    def __init__(self, env, variant="bias_cap", ang_tol=None, cluster_cap=None,
                 window=None, n_bins=None):
        super().__init__(env)
        u = env.unwrapped
        self.num_targets = int(u.num_targets)
        self.variant = variant
        self.window = (None if window is None else int(window))
        self.tracker = TargetTracker(
            self.num_targets, ang_tol=ang_tol, cluster_cap=cluster_cap,
            window=self.window, n_bins=n_bins)
        self.u_max = self.tracker.u_max()

    # ---- public diagnostics (same API as ChaoUReward) ----------------------
    def current_U(self):
        return self.tracker.U(variant=self.variant)

    def current_C(self):
        return self.tracker.variance_confidence_sum()

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
        return obs

    def step(self, action):
        obs, r, done, info = self.env.step(action)

        camera_xy, target_xy, view = state_arrays(self.env)
        self.tracker.observe(camera_xy, target_xy, view)
        U = self.tracker.U(variant=self.variant)
        C = self.tracker.variance_confidence_sum()

        if isinstance(info, (list, tuple)) and info:
            info = list(info)
            for d in info:
                if isinstance(d, dict):
                    d.setdefault("U", float(U))
                    d.setdefault("C", float(C))

        return obs, r, done, info


# ==================== intrinsic scale calibration ===========================
def measure_reward_scale(env, n_episodes=2, n_steps=500, seed=0):
    """Per-step |r_env| and the distribution of the UNIT intrinsic term over
    random rollouts of an ALREADY-wrapped env (the wrapper must be built with
    reward_lambda=1 so the intrinsic term is the unit scale).

    The unit deltas are heavily right-skewed (rare large angular-diversity
    jumps), so we report percentiles (p50/p75/p95) rather than just the mean —
    calibrating on the mean over-scales lambda and saturates the reward clip.
    """
    rng = np.random.default_rng(seed)
    env_abs, env_raw, unit = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < n_steps:
            act = rng.uniform(-1.0, 1.0, size=(obs.shape[0], 2))
            obs, r, done, info = env.step(act)
            r_intr = (info[0].get("intrinsic_reward", 0.0)
                      if isinstance(info, (list, tuple)) and info else 0.0)
            env_abs.append(abs(float(r) - float(r_intr)))
            env_raw.append(float(r) - float(r_intr))
            unit.append(abs(float(r_intr)))
            steps += 1
    env_abs = np.asarray(env_abs)
    env_raw = np.asarray(env_raw)
    unit = np.asarray(unit)
    unit_nonzero = unit[unit > 1e-12]
    return {"env_abs_mean": float(env_abs.mean()),
            "env_std": float(env_raw.std()),
            "unit_p50": float(np.percentile(unit, 50)),
            "unit_p75": float(np.percentile(unit, 75)),
            "unit_p95": float(np.percentile(unit, 95)),
            "unit_nonzero_p75": (float(np.percentile(unit_nonzero, 75))
                                 if unit_nonzero.size else 0.0),
            "n": int(unit.size)}


def suggest_lambda(scale, target_ratio=1.0, base_percentile=75):
    """reward_lambda so that the p{base_percentile} of |r_intr| matches
    target_ratio * noise_scale of the env reward.

    noise_scale = env reward per-step STD (the signal the shaping must compete
    with), falling back to mean|r_env| when the STD is unavailable. The env
    reward in tracking regimes is mostly zeros with rare huge spikes, so
    mean|r_env| vastly under-estimates the noise that advantages see; that is
    why the earlier mean-based calibration made the intrinsic term negligible.
    """
    unit = scale[f"unit_p{base_percentile}"]
    # Sparse signals (e.g. discrete Chao-U: most steps have a zero delta) give
    # a global p75 of 0, which would blow lambda up to ~1e9. Fall back to the
    # p75 of the NON-ZERO deltas so that the magnitude of each firing event is
    # calibrated to the env noise scale. Dense signals are unaffected
    # (global p75 > 0 already).
    if unit <= 0.0:
        unit = scale.get("unit_nonzero_p75", 0.0)
    # If the intrinsic NEVER fires during calibration (e.g. quasi-static
    # targets under a random probe), both percentiles are 0 and lambda would
    # still explode to ~1e9. Fall back to a neutral, finite scale: assume the
    # unit signal is comparable to the env noise (lambda = target_ratio).
    if unit <= 0.0:
        unit = scale.get("env_std", scale.get("env_abs_mean", 0.0))
    noise = scale.get("env_std", scale.get("env_abs_mean", 0.0))
    return float(target_ratio * noise / max(unit, 1e-9))


def default_intr_clip(scale, target_ratio=1.0, safety=2.5):
    """|intrinsic reward| cap = safety * target_ratio * env-noise-scale; passes
    the p{base_percentile} level unclipped and clips only the spike tail."""
    noise = scale.get("env_std", scale.get("env_abs_mean", 0.0))
    return float(safety * target_ratio * noise)


def measure_loc_scale(env, n_episodes=2, n_steps=500, seed=0):
    """Per-step localization-reward scale with reward_lambda=1.

    loc = sum_t q_t is bounded [0, n_targets] and sparse (>= 2-view geometry is
    the bottleneck in 4v8, ~0-2% of steps), so the global p75 is ~0; the scale
    that matters for lambda is the magnitude of each FIRING (loc_nonzero_p75).
    The env-reward noise is read from the wrapper's `env_reward` info field (so
    both signals come from the SAME rollout, with the learning clip applied).
    """
    env = LocalizationReward(env, reward_lambda=1.0)
    rng = np.random.default_rng(seed)
    loc, env_r = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done, st = False, 0
        while not done and st < n_steps:
            a = rng.uniform(-1.0, 1.0, size=(obs.shape[0], 2))
            obs, _, done, info = env.step(a)
            d = info[0] if isinstance(info, (list, tuple)) and info else {}
            loc.append(float(d.get("localization_reward", 0.0)))
            env_r.append(float(d.get("env_reward", 0.0)))
            st += 1
    env.close()
    loc = np.asarray(loc, dtype=np.float64)
    env_r = np.asarray(env_r, dtype=np.float64)
    nz = loc[loc > 1e-12]
    return {
        "loc_mean": float(loc.mean()) if loc.size else 0.0,
        "loc_p75": float(np.percentile(loc, 75)) if loc.size else 0.0,
        "loc_nonzero_p75": float(np.percentile(nz, 75)) if nz.size else 0.0,
        "loc_nonzero_n": int(nz.size),
        "env_abs_mean": float(np.abs(env_r).mean()) if env_r.size else 0.0,
        "env_std": float(env_r.std()) if env_r.size else 0.0,
        "n": int(loc.size),
    }


def suggest_loc_lambda(scale, target_ratio=1.0):
    """reward_lambda so that each localization FIRING (p75 of the non-zero
    sum_t q_t) matches target_ratio * env-reward noise. If the oracle never
    fires during calibration, falls back to the neutral scale
    (lambda = target_ratio)."""
    noise = scale.get("env_std", scale.get("env_abs_mean", 0.0))
    unit = scale.get("loc_nonzero_p75", 0.0)
    if unit <= 0.0:
        unit = scale.get("loc_mean", 0.0)
    if unit <= 0.0:
        return float(target_ratio)
    return float(target_ratio * noise / max(unit, 1e-9))


def measure_rnd_scale(env, n_episodes=2, n_steps=500, seed=0):
    """RND intrinsic scale with reward_lambda=1: per-step team novelty
    (sum_c error) distribution over random rollouts plus the env-reward noise
    from the SAME rollout (wrapper `env_reward` info field, learning clip
    applied). The predictor trains during calibration, so the scale reflects
    the steady-state error magnitude the learner will see."""
    env = RNDIntrinsic(env, reward_lambda=1.0, seed=seed)
    rng = np.random.default_rng(seed)
    intr, env_r = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done, st = False, 0
        while not done and st < n_steps:
            a = rng.uniform(-1.0, 1.0, size=(obs.shape[0], 2))
            obs, _, done, info = env.step(a)
            d = info[0] if isinstance(info, (list, tuple)) and info else {}
            intr.append(float(d.get("intrinsic_reward", 0.0)))
            env_r.append(float(d.get("env_reward", 0.0)))
            st += 1
    env.close()
    intr = np.asarray(intr, dtype=np.float64)
    env_r = np.asarray(env_r, dtype=np.float64)
    nz = intr[intr > 1e-12]
    return {
        "rnd_mean": float(intr.mean()) if intr.size else 0.0,
        "rnd_p75": float(np.percentile(intr, 75)) if intr.size else 0.0,
        "rnd_nonzero_p75": (float(np.percentile(nz, 75)) if nz.size else 0.0),
        "rnd_nonzero_n": int(nz.size),
        "env_abs_mean": float(np.abs(env_r).mean()) if env_r.size else 0.0,
        "env_std": float(env_r.std()) if env_r.size else 0.0,
        "n": int(intr.size),
    }


def suggest_rnd_lambda(scale, target_ratio=1.0):
    """reward_lambda so that the p75 of the per-step team novelty matches
    target_ratio * env-reward noise. RND error is always positive (random init
    can't be matched exactly), so the fallback only guards degenerate cases."""
    noise = scale.get("env_std", scale.get("env_abs_mean", 0.0))
    unit = scale.get("rnd_nonzero_p75", 0.0)
    if unit <= 0.0:
        unit = scale.get("rnd_p75", 0.0)
    if unit <= 0.0:
        unit = scale.get("rnd_mean", 0.0)
    if unit <= 0.0:
        return float(target_ratio)
    return float(target_ratio * noise / max(unit, 1e-9))
