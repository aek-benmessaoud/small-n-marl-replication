"""
analysis/stats.py — Paired statistics (Project08 port, locked protocol).

  - Paired Wilcoxon (zero_method="wilcox") per regime.
  - Holm-Bonferroni across regimes (global).
  - Delta = matched-pairs statistic m/n^2 (positive = A better).
  - Median relative gain of A over B (sign flipped for lower-is-better).
  - Fisher combined p across regimes (global statement).
"""

import numpy as np
from scipy import stats


def paired_stats(a, b):
    """(p, delta) paired Wilcoxon + matched-pairs delta."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return float("nan"), 0.0
    d = a - b
    if np.all(np.abs(d) < 1e-12):
        return 1.0, 0.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return 1.0, 0.0
    n = len(a)
    m = 0.0
    for i in range(n):
        m += np.sum(a[i] > b) - np.sum(a[i] < b)
    return float(p), m / (n * n)


def holm_bonferroni(ps):
    """Holm-Bonferroni corrected p-values (monotone, >= raw)."""
    n = len(ps)
    order = np.argsort(ps)
    out = [None] * n
    for rank, idx in enumerate(order):
        out[idx] = min(1.0, ps[idx] * (n - rank))
    for i in range(n - 1, 0, -1):
        out[order[i - 1]] = min(out[order[i - 1]], out[order[i]])
    return out


def gain_pct(va, vb, lower_better=False):
    """Median relative gain of A over B; positive = A better."""
    va = np.asarray(va, dtype=float)
    vb = np.asarray(vb, dtype=float)
    if lower_better:
        num, den = vb - va, va
    else:
        num, den = va - vb, vb
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = num / np.where(den == 0, np.nan, den)
    return float(np.nanmedian(rel) * 100.0)


def fisher_combined(ps):
    """Fisher combined p across the given (raw) p-values."""
    ps = np.asarray(ps, dtype=float)
    ps = ps[np.isfinite(ps) & (ps > 0)]
    if len(ps) < 2:
        return float("nan")
    chi2 = -2.0 * np.sum(np.log(np.maximum(ps, 1e-300)))
    return float(1.0 - stats.chi2.cdf(chi2, 2 * len(ps)))
