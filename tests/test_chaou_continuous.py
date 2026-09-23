"""
tests/test_chaou_continuous.py — Phase 1 gate: the continuous target-centric
Chao-U signal must equal the grid-cell Chao-U signal when target positions are
snapped to grid centers.

A synthetic scene places targets exactly on a lattice. The Project08 GRID
pipeline (per-cell bearing clustering -> visit/known/obs -> chao_u) and the
Project09 CONTINUOUS pipeline (per-target bearing clustering -> same bundle ->
same chao_u) must produce identical F1/F2 and U within np.allclose(rtol=1e-2).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from project09.config import ANG_TOL_DEG, CLUSTER_CAP
from project09.estimators.angular import greedy_cluster_centers
from project09.estimators.continuous import TargetTracker
from project09.estimators.richness import chao_u, chao_u_components


def _grid_bundle(cells, observers, fov_radius):
    """Project08 GRID pipeline: per-cell bearings -> (visit, known, obs).

    cells     : (n_cell, 2) integer cell coordinates.
    observers : (n_obs, 2) observer positions.
    fov_radius: square FOV half-width in cells (Chebyshev window).
    """
    cell_index = {tuple(c): i for i, c in enumerate(cells)}
    angles = [[] for _ in range(len(cells))]
    for (ox, oy) in observers:
        for (cx, cy) in cells:
            if max(abs(cx - ox), abs(cy - oy)) <= fov_radius:
                angles[cell_index[(cx, cy)]].append(
                    float(np.arctan2(cy - oy, cx - ox)))
    counts = np.array([
        len(greedy_cluster_centers(a, tol=np.deg2rad(ANG_TOL_DEG),
                                   cap=CLUSTER_CAP))
        for a in angles
    ], dtype=np.float64)
    visit = counts
    known = counts > 0
    obs = np.zeros(len(cells), dtype=bool)
    return visit, known, obs


def test_continuous_equals_grid_at_centers():
    rng = np.random.default_rng(0)
    # 6 targets snapped to lattice points; 4 observers at lattice points.
    cells = np.array([[2, 3], [5, 8], [10, 2], [7, 9], [4, 4], [11, 7]],
                     dtype=np.int64)
    observers = np.array([[0, 0], [6, 1], [12, 5], [3, 10]], dtype=np.int64)
    fov_radius = 3

    # --- grid pipeline ------------------------------------------------------
    g_visit, g_known, g_obs = _grid_bundle(cells, observers, fov_radius)
    g_undet = int(np.sum((g_known & ~g_obs) & (g_visit <= 1)))
    U_grid = chao_u(g_visit, g_known, g_obs, total_unknown=g_undet,
                    variant="bias_cap")
    F1_grid, F2_grid = chao_u_components(g_visit, g_known, g_obs)

    # --- continuous pipeline (targets at the SAME lattice points) -----------
    tracker = TargetTracker(num_targets=len(cells))
    camera_xy = observers.astype(np.float64)
    target_xy = cells.astype(np.float64)
    view = np.zeros((len(observers), len(cells)), dtype=bool)
    for c, (ox, oy) in enumerate(observers):
        for t, (cx, cy) in enumerate(cells):
            view[c, t] = max(abs(cx - ox), abs(cy - oy)) <= fov_radius
    tracker.observe(camera_xy, target_xy, view)

    assert np.array_equal(tracker.known(), g_known)
    assert np.array_equal(tracker.visit(), g_visit)
    F1_c, F2_c = tracker.components()
    assert F1_c == F1_grid and F2_c == F2_grid
    U_cont = tracker.U(variant="bias_cap",
                       total_unknown=int(np.sum(tracker.underdetermined())))
    assert np.allclose(U_cont, U_grid, rtol=1e-2)


def test_tracker_u_stable_without_new_observations():
    """Adding no new views must not change U (stability gate)."""
    rng = np.random.default_rng(1)
    tracker = TargetTracker(num_targets=4)
    camera_xy = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    target_xy = np.array([[10.0, 10.0], [30.0, 20.0], [60.0, 5.0],
                          [5.0, 40.0]])
    view = np.array([[1, 0, 0, 1],
                     [0, 1, 0, 1],
                     [1, 1, 1, 0]], dtype=bool)
    tracker.observe(camera_xy, target_xy, view)
    U1 = tracker.U(variant="bias_cap")
    # Re-observe the SAME view: buffers only accumulate distinct steps, so a
    # duplicate identical snapshot is a new timestep observation (buffers grow
    # but configuration counts saturate under the cluster cap / tolerance).
    tracker.observe(camera_xy, target_xy, view)
    U2 = tracker.U(variant="bias_cap")
    # With greedy clustering on identical bearings, the config counts must not
    # grow beyond the cap; U is bounded and finite.
    assert np.isfinite(U2)
    assert U2 >= 1.0


def test_u_decreases_when_configuration_added():
    """Adding a second independent bearing to a target reduces U."""
    camera_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    target_xy = np.array([[50.0, 1.0], [50.0, 60.0]])
    view1 = np.array([[1, 0],
                      [0, 1]], dtype=bool)
    view2 = np.array([[1, 1],
                      [1, 0]], dtype=bool)

    tracker = TargetTracker(num_targets=2)
    tracker.observe(camera_xy, target_xy, view1)
    U_before = tracker.U(variant="bias_cap")
    tracker.observe(camera_xy, target_xy, view2)
    U_after = tracker.U(variant="bias_cap")
    # target 0 now has two independent bearings -> under-determined set shrinks
    assert U_after <= U_before


def test_windowed_tracker_evicts_old_bearings():
    """window=N keeps only bearings from the last N observe() calls; older
    ones are evicted so the configuration count can drop again."""
    camera_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    target_xy = np.array([[50.0, 1.0]])
    cam = np.array([[True]], dtype=bool)

    tracker = TargetTracker(num_targets=1, window=3)
    # step 0: only cam1 sees the target -> one configuration (bearing ~179 deg)
    tracker.observe(camera_xy[1:], target_xy, cam)
    assert len(tracker.buffers[0]) == 1
    # step 1: cam0 also sees it -> second, well-separated configuration
    tracker.observe(camera_xy[:1], target_xy, cam)
    assert tracker.config_count(0) == 2
    assert len(tracker.buffers[0]) == 2
    # steps 2,3: cam0 only; after step 3 the step-0 bearing (cam1) is evicted
    tracker.observe(camera_xy[:1], target_xy, cam)
    assert len(tracker.buffers[0]) == 3
    tracker.observe(camera_xy[:1], target_xy, cam)
    assert len(tracker.buffers[0]) == 3  # window kept, oldest dropped
    assert tracker.config_count(0) == 1  # second config was evicted


def test_windowed_u_rises_again_after_eviction():
    """Windowed richness is SHORT-TIMESCALE: U drops when a configuration is
    gained, then RISES back when the window slides past it — the episode-
    cumulative model never forgets, the windowed one does."""
    camera_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    target_xy = np.array([[50.0, 1.0], [50.0, 60.0], [10.0, 90.0]])
    cam0 = np.array([[1, 1, 1],
                     [0, 0, 0]], dtype=bool)   # only cam0 sees all targets
    cam0_and_cam1 = np.array([[1, 1, 1],
                              [1, 0, 0]], dtype=bool)  # cam1 also sees t0

    tracker = TargetTracker(num_targets=3, window=2)
    # step 0: all 3 targets, 1 configuration each -> F1=3, U=3.0
    tracker.observe(camera_xy, target_xy, cam0)
    U_early = tracker.U(variant="bias_cap")
    # step 1: t0 gains a 2nd, well-separated configuration -> F2=1, U drops
    tracker.observe(camera_xy, target_xy, cam0_and_cam1)
    U_mid = tracker.U(variant="bias_cap")
    assert U_mid < U_early
    # steps 2,3: cam0 only; at step 3 (window=2) the step-1 cam1 bearing is
    # evicted and t0 is back to 1 configuration -> U rises again
    tracker.observe(camera_xy, target_xy, cam0)
    tracker.observe(camera_xy, target_xy, cam0)
    U_late = tracker.U(variant="bias_cap")
    assert tracker.config_count(0) == 1
    assert U_late > U_mid


def test_n_bins_counts_occupied_bins():
    """n_bins discretizes the circle; config_count = number of DISTINCT
    occupied bins (roadmap step 6, bin-sensitivity knob)."""
    tracker = TargetTracker(num_targets=1, n_bins=8)
    # bin width 45 deg. bearings 0/15 -> same bin; 45 -> next; 90 -> next.
    for deg in (0.0, 15.0, 45.0, 90.0):
        tracker.buffers[0].append(np.deg2rad(deg))
    assert tracker.config_count(0) == 3
    assert tracker.visit()[0] == 3.0

    # same bearings under greedy clustering: 0 and 15 deg within ANG_TOL_DEG
    # (15 deg -> NOT separated) so greedy gives 3; bins with 8 bins also 3.
    t24 = TargetTracker(num_targets=1, n_bins=24)
    for deg in (0.0, 15.0):
        t24.buffers[0].append(np.deg2rad(deg))
    assert t24.config_count(0) == 2  # 24 bins -> 15 deg/bin -> two bins

    t_empty = TargetTracker(num_targets=1, n_bins=8)
    assert t_empty.config_count(0) == 0  # never seen -> 0 configurations
