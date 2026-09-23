"""
estimators/continuous.py — Continuous target-centric localization tracker.

THE transposition: in Project08 the "unit" was a grid cell and the knowledge
bundle was {visit, known, obs} over cells. Here the unit is a MOVING TARGET:
every camera that observes target t adds a bearing observation (the angle from
camera to target) to t's buffer. The tracker is fed each step from the MATE
ground-truth camera->target view mask (decision signal = what the team can
actually see).

It exposes the SAME knowledge bundle that estimators/richness.py expects, so
the Chao-U machinery is reused byte-identical:

  - visit[t]  := number of independent angular configurations of target t
                 (greedy cluster centers of the raw bearings, sorted order),
  - known[t]  := target t observed at least once,
  - obs[t]    := False for all targets (parity dummy),
  - underdetermined[t] := visit[t] <= 1  (rank-deficient, cannot localize).

Bearing convention (must match MATE's geometry): arctan2(y_target - y_camera,
x_target - x_camera), radians in (-pi, pi].
"""

import numpy as np

from project09.estimators.angular import greedy_cluster_centers
from project09.estimators.richness import chao_u, chao_u_components


def bearing_from(camera_xy, target_xy):
    """Bearing camera -> target, radians in (-pi, pi]."""
    return float(np.arctan2(target_xy[1] - camera_xy[1],
                            target_xy[0] - camera_xy[0]))


class TargetTracker:
    """Accumulates per-target bearing buffers and derives the richness bundle.

    Parameters:
        num_targets: number of targets (fixed by the environment config).
        ang_tol: clustering tolerance in radians (ANG_TOL_DEG).
        cluster_cap: max configurations per target.
        n_bins: if set, discretize the full circle in `n_bins` equal angular
            bins and count OCCUPIED BINS as the independent configurations of
            each target (replaces greedy clustering). Bin-sensitivity knob
            (roadmap step 6): 8/12/16/24 bins ↔ 45/30/22.5/15 deg per bin.
    """

    def __init__(self, num_targets, ang_tol=None, cluster_cap=None, window=None,
                 n_bins=None):
        from project09.config import ANG_TOL_DEG, CLUSTER_CAP
        self.num_targets = int(num_targets)
        self.ang_tol = (np.deg2rad(ANG_TOL_DEG) if ang_tol is None
                        else ang_tol)
        self.cluster_cap = (CLUSTER_CAP if cluster_cap is None else cluster_cap)
        self.n_bins = (None if n_bins is None else int(n_bins))
        if self.n_bins is not None and self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        # Sliding-window mode: only bearings recorded in the last `window`
        # observe() calls count (None = episode-cumulative, the default).
        self.window = None if window is None else int(window)
        self.buffers = [[] for _ in range(self.num_targets)]
        self.buffer_steps = [[] for _ in range(self.num_targets)]
        self._step = 0
        self.positions = np.full((self.num_targets, 2), np.nan)
        self._config_count_cache = None

    # ---- observation intake ------------------------------------------------
    def observe(self, camera_xy, target_xy, view_bits):
        """Add one bearing per (camera, target) pair with view_bits[c,t] True.

        camera_xy : (n_cam, 2) array of camera positions.
        target_xy : (n_targets, 2) array of target positions.
        view_bits : (n_cam, n_targets) bool — camera_target_view_mask.
        """
        view_bits = np.asarray(view_bits, dtype=bool)
        cam = np.asarray(camera_xy, dtype=np.float64)
        tar = np.asarray(target_xy, dtype=np.float64)
        n_cam, n_tar = view_bits.shape
        assert cam.shape[0] == n_cam and tar.shape[0] == n_tar
        for c in range(n_cam):
            for t in np.flatnonzero(view_bits[c]):
                self.buffers[t].append(bearing_from(cam[c], tar[t]))
                self.buffer_steps[t].append(self._step)
                self.positions[t] = tar[t]
        self._step += 1
        if self.window is not None:
            cutoff = self._step - self.window
            for t in range(n_tar):
                bs = self.buffer_steps[t]
                if bs and bs[0] < cutoff:
                    # drop the leading contiguous run older than the window
                    n_drop = int(np.searchsorted(np.asarray(bs, dtype=np.int64),
                                                 cutoff, side="left"))
                    del self.buffers[t][:n_drop]
                    del self.buffer_steps[t][:n_drop]
        self._config_count_cache = None

    def target_pos(self, t):
        """Most recently observed position of target t (nan if never seen)."""
        return self.positions[t].copy()

    # ---- dense (continuous) confidence --------------------------------------
    def angular_spread(self, t, window=None):
        """Max pairwise circular separation of the (optionally windowed)
        recorded bearings of target t, radians in [0, pi].

        Computed as min(max_gap, 2pi - max_gap) over the sorted bearings,
        where max_gap is the largest empty arc (O(N log N), no O(N^2) blow-up
        on long buffers). 0 if fewer than 2 bearings recorded.
        """
        b = self.buffers[t]
        if window is not None:
            b = b[-window:]
        if len(b) < 2:
            return 0.0
        a = np.sort(np.asarray(b, dtype=np.float64))
        gaps = np.diff(a)
        wrap = (a[0] + 2.0 * np.pi) - a[-1]
        max_gap = float(max(gaps.max(), wrap))
        return float(min(max_gap, 2.0 * np.pi - max_gap))

    def dense_confidences(self, window=None):
        """Per-target continuous confidence in [0, 1].

        Confidence grows with the angular spread of the recorded bearings and
        saturates at 1 once the spread reaches the clustering tolerance (the
        same criterion that creates a second configuration in the binary
        bundle). This is a dense, always-differentiable relaxation of the
        discrete known/unknown signal.
        """
        conf = np.zeros(self.num_targets, dtype=np.float64)
        for t in range(self.num_targets):
            if len(self.buffers[t]) >= 2:
                conf[t] = min(1.0, self.angular_spread(t, window=window)
                              / self.ang_tol)
        return conf

    def dense_confidence_sum(self, window=None):
        """Total dense confidence over targets (in [0, num_targets])."""
        return float(self.dense_confidences(window=window).sum())

    def angular_variance(self, t, window=None):
        """Circular variance 1 - |sum exp(i*theta)| / N of the (windowed)
        bearings of target t, in [0, 1]. Responds to EVERY new bearing, so it
        is a dense shaping signal (unlike the max-pairwise spread which only
        moves when a bearing lands outside the current arc)."""
        b = self.buffers[t]
        if window is not None:
            b = b[-window:]
        if not b:
            return 0.0
        b = np.asarray(b, dtype=np.float64)
        n = b.size
        R = np.abs(np.exp(1j * b).sum()) / n
        return float(1.0 - R)

    def variance_confidences(self, window=None):
        """Per-target circular variance in [0, 1] (0 unseen / single bearing)."""
        conf = np.zeros(self.num_targets, dtype=np.float64)
        for t in range(self.num_targets):
            if len(self.buffers[t]) >= 1:
                conf[t] = self.angular_variance(t, window=window)
        return conf

    def variance_confidence_sum(self, window=None):
        """Total circular-variance confidence over targets."""
        return float(self.variance_confidences(window=window).sum())

    def last_bearing(self, t):
        """Most recent raw bearing to target t, radians (None if never seen)."""
        if not self.buffers[t]:
            return None
        return self.buffers[t][-1]

    # ---- derived bundle ----------------------------------------------------
    def _bin_config_count(self, bearings):
        """Count DISTINCT occupied angular bins of `bearings` under the
        n_bins discretization. 0 for an empty buffer (never seen)."""
        if not bearings:
            return 0
        a = np.asarray(bearings, dtype=np.float64)
        a = (a + np.pi) % (2.0 * np.pi)          # normalize to [0, 2pi)
        idx = np.floor(a / (2.0 * np.pi) * self.n_bins).astype(np.int64)
        return int(np.unique(idx).size)

    def config_count(self, t):
        """Number of independent configurations of target t (>= 0)."""
        if self._config_count_cache is not None:
            return self._config_count_cache[t]
        if self.n_bins is not None:
            return self._bin_config_count(self.buffers[t])
        return len(greedy_cluster_centers(
            self.buffers[t], tol=self.ang_tol, cap=self.cluster_cap))

    def config_counts(self):
        """Array of independent-configuration counts, one per target."""
        if self._config_count_cache is None:
            if self.n_bins is not None:
                self._config_count_cache = np.array([
                    self._bin_config_count(b) for b in self.buffers
                ], dtype=np.int64)
            else:
                self._config_count_cache = np.array([
                    len(greedy_cluster_centers(b, tol=self.ang_tol,
                                               cap=self.cluster_cap))
                    for b in self.buffers
                ], dtype=np.int64)
        return self._config_count_cache

    def visit(self):
        return self.config_counts().astype(np.float64)

    def known(self):
        return (self.config_counts() > 0)

    def obs(self):
        return np.zeros(self.num_targets, dtype=bool)

    def underdetermined(self):
        """Known-but-under-determined targets: observed (>= 1 configuration)
        with <= 1 independent configuration (single bearing is rank-deficient).
        Targets NEVER observed are excluded — the signal starts flat, exactly
        mirroring Project08's get_total_undetermined()."""
        cc = self.config_counts()
        return (cc >= 1) & (cc <= 1)

    # ---- richness outputs --------------------------------------------------
    def U(self, variant="bias_cap", total_unknown=None):
        """Chao-U remaining-work estimate over targets.

        total_unknown defaults to the number of under-determined targets
        (cap = units with <= 1 configuration, matching Project08 semantics).
        """
        if total_unknown is None:
            total_unknown = int(np.sum(self.underdetermined()))
        return chao_u(self.visit(), self.known(), self.obs(),
                      total_unknown=total_unknown, variant=variant)

    def components(self):
        """(F1, F2) — targets with exactly 1 / 2 configurations."""
        return chao_u_components(self.visit(), self.known(), self.obs())

    def u_max(self):
        """Max possible remaining-work estimate (reward normalizer)."""
        return float(self.num_targets)

    def reset(self):
        self.buffers = [[] for _ in range(self.num_targets)]
        self.buffer_steps = [[] for _ in range(self.num_targets)]
        self._step = 0
        self.positions = np.full((self.num_targets, 2), np.nan)
        self._config_count_cache = None
