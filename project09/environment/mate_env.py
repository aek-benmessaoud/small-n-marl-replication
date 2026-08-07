"""
environment/mate_env.py — MATE environment factory for Project09.

Builds the standard wrapped MATE environment used by both the MAPPO trainer
and the heuristic/evaluation harnesses:

    base = mate.make(env_id, config=regime, reward_type=...)
    env  = mate.MultiCamera(base, target_agent=GreedyTargetAgent(seed=...))

`MultiCamera` gives the RL-friendly single-team API:
  - reset()  -> obs (num_cameras, obs_dim)
  - step(a)  -> (obs, reward, done, infos), reward = shared scalar float,
                done = episode end, infos = list of per-camera dicts.

Ground-truth state access (evaluation / decision signal) is via
env.unwrapped: cameras[i].location (np.ndarray (2,)), targets[i].location,
camera_target_view_mask (num_cameras, num_targets) bool.
"""

import os
import warnings

import numpy as np

import mate

from project09.config import (
    DEFAULT_REGIME, REWARD_SPARSE, REGIME_IMBALANCED,
)


def _resolve_config(config):
    """Map a regime id ('MATE-4v8-9-v0') to its asset yaml path, or pass
    through an existing path / dict. gym.make's config kwarg expects a yaml
    filename resolvable from mate.ASSETS_DIR (NOT a gym id)."""
    if config is None:
        return mate.ASSETS_DIR / f"{DEFAULT_REGIME[:-3]}.yaml"
    if isinstance(config, (dict, os.PathLike)):
        return config
    if isinstance(config, str):
        if os.path.exists(config):
            return config
        name = config
        if name.endswith("-v0"):
            name = name[:-3]
        if not name.lower().endswith((".yaml", ".yml", ".json")):
            name = f"{name}.yaml"
        return mate.ASSETS_DIR / name
    return config


def make_mate(env_id=None, config=None, reward_type=None, seed=None,
              opponent_agent_factory=None, wrappers=None):
    """Create the wrapped MATE environment.

    env_id   : gym id; default 'MultiAgentTracking-v0'.
    config   : regime config ('MATE-4v8-9-v0' style id or asset name/yaml path);
               default DEFAULT_REGIME's asset.
    reward_type : 'dense' | 'sparse'; default from REWARD_SPARSE config.
    seed     : env seed (applied via env.seed).
    opponent_agent_factory : callable() -> TargetAgentBase used as the moving
               target policy; default GreedyTargetAgent(seed=0) (official
               MAPPO example convention).
    wrappers : extra callables applied after MultiCamera (in order).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = _resolve_config(config)
        env = mate.make(
            env_id or "MultiAgentTracking-v0",
            config=config,
            reward_type=("dense" if not REWARD_SPARSE else "sparse")
            if reward_type is None else reward_type,
        )
        if seed is not None:
            env.seed(seed)

        factory = opponent_agent_factory or (lambda: mate.GreedyTargetAgent(seed=0))
        env = mate.MultiCamera(env, target_agent=factory())

        for w in (wrappers or []):
            env = w(env)

    return env


def state_arrays(env):
    """Current camera/target positions + view mask from the unwrapped env.

    Returns (camera_xy, target_xy, view_mask):
      camera_xy : (num_cameras, 2) float array
      target_xy : (num_targets, 2) float array
      view_mask : (num_cameras, num_targets) bool — which camera sees which
                  target (field of view + line of sight + sight range).
    """
    u = env.unwrapped
    camera_xy = np.array([[c.location[0], c.location[1]] for c in u.cameras],
                         dtype=np.float64)
    target_xy = np.array([[t.location[0], t.location[1]] for t in u.targets],
                         dtype=np.float64)
    return camera_xy, target_xy, np.asarray(u.camera_target_view_mask, dtype=bool)


def num_cameras_targets(env):
    u = env.unwrapped
    return int(u.num_cameras), int(u.num_targets)


def num_entities(env):
    """(num_cameras, num_targets, num_obstacles) from the unwrapped env."""
    u = env.unwrapped
    return int(u.num_cameras), int(u.num_targets), int(u.num_obstacles)
