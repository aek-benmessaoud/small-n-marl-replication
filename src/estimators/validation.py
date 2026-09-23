"""
estimators/validation.py — CRLB oracle for target localization (evaluation
only, decoupled from the decision signal).

For every target, the cameras that observe it at the FINAL state contribute
bearings; the Cramer-Rao lower bound on the target's position estimate is

    J = sum_k u_k u_k^T / (sigma^2 d_k^2)
    bound = sqrt(trace(J^{-1}))

where u_k is the unit vector from camera k to the target and d_k the distance.
Targets with fewer than 2 linearly independent directions (J singular / < 2
observing cameras) have +inf bound — they are "under-determined".

Also ports the Gauss-Newton bearing-only triangulation for empirical error
validation (Spearman correlations), Project08's Phase-1a pattern.
"""

import numpy as np
from scipy.stats import spearmanr

from src.config import QUALITY_SIGMA_BEARING_DEG, QUALITY_THRESHOLD


def true_bearing(observer_xy, target_xy):
    """Bearing observer -> target, radians in (-pi, pi]."""
    ox, oy = observer_xy
    tx, ty = target_xy
    return np.arctan2(ty - oy, tx - ox)


def crlb_bound(observer_xy, target_xy, sigma_deg=QUALITY_SIGMA_BEARING_DEG):
    """Per-target CRLB bound sqrt(trace(J^-1)).

    observer_xy : (n, 2) observing camera positions (ground truth).
    target_xy   : (2,) target position.
    Returns +inf if < 2 observers or J singular (under-determined).
    """
    obs = np.asarray(observer_xy, dtype=np.float64)
    tar = np.asarray(target_xy, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] < 2:
        return float("inf")
    sigma = np.deg2rad(sigma_deg)
    d = obs - tar
    d2 = np.sum(d * d, axis=1)
    if np.any(d2 < 1e-12):
        return float("inf")
    u = d / np.sqrt(d2[:, None])
    J = np.einsum("ki,kj,k->ij", u, u, 1.0 / (sigma * sigma * d2))
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if det < 1e-12:
        return float("inf")
    Jinv = np.linalg.inv(J)
    return float(np.sqrt(np.trace(Jinv)))


def per_target_bounds(camera_xy, target_xy, view_mask, sigma_deg=None):
    """CRLB bound per target from the current view.

    camera_xy : (n_cam, 2)
    target_xy : (n_tar, 2)
    view_mask : (n_cam, n_tar) bool — which cameras see which targets.
    Returns (n_tar,) float array (+inf for unobserved/under-determined).
    """
    if sigma_deg is None:
        sigma_deg = QUALITY_SIGMA_BEARING_DEG
    view_mask = np.asarray(view_mask, dtype=bool)
    n_tar = target_xy.shape[0]
    bounds = np.full(n_tar, np.inf, dtype=np.float64)
    for t in range(n_tar):
        obs = camera_xy[view_mask[:, t]]
        bounds[t] = crlb_bound(obs, target_xy[t], sigma_deg=sigma_deg)
    return bounds


def well_localized(bounds, threshold=QUALITY_THRESHOLD):
    """Fraction of targets with finite bound <= threshold."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(bounds) & (bounds <= threshold)))


def gauss_newton_localize(observers, noisy_bearings, init=None, iters=12,
                          tol=1e-9):
    """Gauss-Newton bearing-only triangulation (Project08 port).

    observers      : (n, 2) true observer positions.
    noisy_bearings : measured bearings (radians), same length.
    Returns (x, y) estimate, or (nan, nan) if rank deficient.
    """
    obs = np.asarray(observers, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] < 2:
        return (float("nan"), float("nan"))
    n = obs.shape[0]
    th = np.asarray(noisy_bearings, dtype=np.float64).reshape(n)

    if init is None:
        p = obs.mean(axis=0).copy()
    else:
        p = np.asarray(init, dtype=np.float64).copy().reshape(2)

    for _ in range(iters):
        dx = p[0] - obs[:, 0]
        dy = p[1] - obs[:, 1]
        d2 = dx * dx + dy * dy
        if np.any(d2 < 1e-12):
            return (float("nan"), float("nan"))
        J = np.empty((n, 2))
        J[:, 0] = -dy / d2
        J[:, 1] = dx / d2
        residual = (np.arctan2(dy, dx) - th) % (2.0 * np.pi)
        residual = np.where(residual > np.pi, residual - 2.0 * np.pi, residual)

        H = J.T @ J
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if det < 1e-12:
            return (float("nan"), float("nan"))
        g = J.T @ residual
        step = np.linalg.solve(H, g)
        p = p - step
        if float(np.max(np.abs(step))) < tol:
            break
    return float(p[0]), float(p[1])


def spearman(x, y):
    """(rho, p) Spearman rank correlation; (nan, 1.0) if degenerate."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 3:
        return float("nan"), 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)
