"""heuristics.py — Deterministic baseline policies for the MATE camera team.

All policies consume ONLY the observation (n_cam, obs_dim), except
ChaoUAwarePolicy which additionally reads the shared TargetTracker belief
maintained by the reward wrapper (the same internal estimate the intrinsic
reward is built from — mirrors Project08's frontier_richness consuming the
env's own U estimate).

Each policy exposes: act(obs, tracker=None) -> (n_cam, 2) actions in [-1, 1].
"""

import numpy as np

from .decode import decode_team_obs


def _steer_to(origin, target):
    """Normalized steering vector from origin to target (both length-2)."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.zeros(2)
    return d / n


class RandomCameraPolicy:
    """Uniform random actions in [-1, 1] (trivial baseline)."""

    def __init__(self, num_cameras=0, num_targets=0, num_obstacles=0, seed=0):
        self._rng = np.random.default_rng(seed)

    def act(self, obs, tracker=None):
        n_cam = obs.shape[0]
        return self._rng.uniform(-1.0, 1.0, size=(n_cam, 2))


class GreedyTrackingPolicy:
    """Each camera moves toward the nearest visible target; when nothing is
    visible it sweeps toward the terrain center. Observation-only."""

    def __init__(self, num_cameras, num_targets, num_obstacles=0):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles

    def act(self, obs, tracker=None):
        n_cam, n_tar = self.num_cameras, self.num_targets
        dec = decode_team_obs(obs, n_cam, n_tar, self.num_obstacles)
        actions = np.zeros((n_cam, 2))
        for i, d in enumerate(dec):
            vis = d["target_masks"]
            if np.any(vis):
                nearest = d["targets"][np.flatnonzero(vis)[0]]
                actions[i] = _steer_to(d["self_location"], nearest)
            else:
                actions[i] = _steer_to(d["self_location"], [0.0, 0.0])
        return actions


class TeamMemoryTrackingPolicy(GreedyTrackingPolicy):
    """Greedy tracking that additionally remembers a fixed number of targets
    it saw recently (beyond the current FOV), steering toward the remembered
    nearest one first. Observation-only."""

    def __init__(self, num_cameras, num_targets, num_obstacles=0, memory=8):
        super().__init__(num_cameras, num_targets, num_obstacles=num_obstacles)
        self.memory = memory
        self._memories = []   # per-camera list of (location, age)

    def reset(self):
        self._memories = []

    def act(self, obs, tracker=None):
        n_cam, n_tar = self.num_cameras, self.num_targets
        dec = decode_team_obs(obs, n_cam, n_tar, self.num_obstacles)
        while len(self._memories) < n_cam:
            self._memories.append([])
        actions = np.zeros((n_cam, 2))
        for i, d in enumerate(dec):
            vis_idx = np.flatnonzero(d["target_masks"])
            candidates = list(d["targets"][vis_idx])
            mem = self._memories[i]
            self._memories[i] = mem[-self.memory:]
            candidates += [m[0] for m in self._memories[i]]
            if candidates:
                target = min(candidates,
                             key=lambda t: np.linalg.norm(d["self_location"] - t))
                actions[i] = _steer_to(d["self_location"], target)
            else:
                actions[i] = _steer_to(d["self_location"], [0.0, 0.0])
        return actions


class ChaoUAwarePolicy:
    """Intrinsic-informed heuristic: from the shared TargetTracker belief,
    find under-determined (seen, 1 configuration) targets and steer each
    camera to gain angular diversity (move perpendicular to the latest
    bearing). When no under-determined target is known, fall back to greedy
    tracking of visible targets. Mirrors Project08's frontier_richness which
    consumed the same U estimate the reward uses."""

    def __init__(self, num_cameras, num_targets, num_obstacles=0, seed=0):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self._greedy = GreedyTrackingPolicy(num_cameras, num_targets,
                                            num_obstacles=num_obstacles)
        self._rng = np.random.default_rng(seed)

    def act(self, obs, tracker=None):
        n_cam = self.num_cameras
        if tracker is None:
            return self._greedy.act(obs)
        undet = np.flatnonzero(tracker.underdetermined())
        if len(undet) == 0:
            return self._greedy.act(obs)

        dec = decode_team_obs(obs, n_cam, self.num_targets, self.num_obstacles)
        actions = np.zeros((n_cam, 2))
        for i, d in enumerate(dec):
            pos = tracker.target_pos(undet[0])
            if np.any(np.isnan(pos)):
                actions[i] = self._greedy.act(obs)[i]
                continue
            bearing = tracker.last_bearing(undet[0])
            if bearing is None:
                actions[i] = _steer_to(d["self_location"], pos)
                continue
            perp = np.array([-np.sin(bearing), np.cos(bearing)])
            if self._rng.random() < 0.5:
                perp = -perp
            actions[i] = perp
        return actions
