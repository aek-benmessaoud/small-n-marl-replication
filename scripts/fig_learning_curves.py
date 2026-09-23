"""scripts/fig_learning_curves.py — Fig. A: training curves for the 8v8 campaign.

Aggregates the per-run `curve` blocks (saved every 10k steps) over the 16
paired seeds for the intrinsic vs no_intrinsic arms, and plots:
  (left) eval_return            mean +/- SEM over seeds,
  (right) C_end (windowed UTracker confidence) mean +/- SEM.
Outputs a two-panel figure.

Usage:
    .venv\\Scripts\\python.exe scripts\\fig_learning_curves.py --tag chao8v8 \\
        --out results/fig_learning_curves.pdf --dpi 200
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [10000, 20000, 30000]
COLORS = {"intrinsic": "#1f77b4", "no_intrinsic": "#d62728"}


def collect(tag):
    """Return {'intrinsic': {step: mean, sem}, 'no_intrinsic': {...}} for the
    eval_return and C_end series across all seeds."""
    rets = {"intrinsic": {s: [] for s in STEPS},
            "no_intrinsic": {s: [] for s in STEPS}}
    cends = {"intrinsic": {s: [] for s in STEPS},
             "no_intrinsic": {s: [] for s in STEPS}}
    n_matched = 0
    for seed in range(16):
        for arm in ("intrinsic", "no_intrinsic"):
            fs = sorted(glob.glob(
                f"results/campaign_{tag}_s{seed:02d}_{arm}/**/results.json",
                recursive=True))
            if not fs:
                continue
            with open(fs[0]) as f:
                j = json.load(f)
            curve = j.get("curve", {})
            steps = curve.get("step", [])
            returns = curve.get("eval_return", [])
            ce = curve.get("C_end", [])
            # align by step index
            idx = {int(s): i for i, s in enumerate(steps)}
            for s in STEPS:
                if s in idx:
                    i = idx[s]
                    if i < len(returns):
                        rets[arm][s].append(float(returns[i]))
                    if i < len(ce):
                        cends[arm][s].append(float(ce[i]))
            if arm == "no_intrinsic":
                n_matched += 1
    summary = {}
    for arm in ("intrinsic", "no_intrinsic"):
        summary[arm] = {
            "return": {s: (_m(rets[arm][s]), _se(rets[arm][s]))
                       for s in STEPS},
            "C_end": {s: (_m(cends[arm][s]), _se(cends[arm][s]))
                      for s in STEPS},
        }
    return summary, n_matched


def _m(x):
    return float(np.mean(x)) if x else float("nan")


def _se(x):
    return float(np.std(x) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def plot_panel(ax, series, ylabel, title):
    for arm, color in COLORS.items():
        xs = [s for s in STEPS if not np.isnan(series[arm][s][0])]
        ys = [series[arm][s][0] for s in xs]
        ses = [series[arm][s][1] for s in xs]
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        ses = np.asarray(ses)
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=color,
                label=f"intrinsic" if arm == "intrinsic" else "no\\_intrinsic")
        ax.fill_between(xs, ys - ses, ys + ses, color=color, alpha=0.15)
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--regime-label", default="8v8")
    ap.add_argument("--out", default="results/fig_learning_curves.pdf")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    summary, n_matched = collect(args.tag)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.1))
    rets = {arm: summary[arm]["return"] for arm in summary}
    cends = {arm: summary[arm]["C_end"] for arm in summary}
    plot_panel(ax1, rets, "eval return", "Tracking reward (true env)")
    plot_panel(ax2, cends, "C$_{end}$ (windowed confidence)",
               "End-of-episode circular-variance confidence")
    fig.suptitle(f"8v8 training curves, mean $\\pm$ SEM over {n_matched} seeds",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved -> {args.out}  (matched seeds: {n_matched})")
    print(f"  intr return @10k/20k/30k:  "
          f"{[round(summary['intrinsic']['return'][s][0],0) for s in STEPS]}")
    print(f"  noi  return @10k/20k/30k:  "
          f"{[round(summary['no_intrinsic']['return'][s][0],0) for s in STEPS]}")
    print(f"  intr C_end @10k/20k/30k:   "
          f"{[round(summary['intrinsic']['C_end'][s][0],3) for s in STEPS]}")
    print(f"  noi  C_end @10k/20k/30k:   "
          f"{[round(summary['no_intrinsic']['C_end'][s][0],3) for s in STEPS]}")


if __name__ == "__main__":
    main()