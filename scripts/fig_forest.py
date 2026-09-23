"""scripts/fig_forest.py — Fig. C (optional): per-seed paired deltas (8v8).

Forest plot of per-seed paired differences (intrinsic - no_intrinsic) for:
  * windowed end-of-episode confidence  C_end(w)  (from eval_final)
  * sequential localization             seq_loc   (from eval_localization)
with the pooled mean + 90%/95% seed-bootstrap CI shown as reference bands.

Usage:
    .venv\\Scripts\\python.exe scripts\\fig_forest.py --tag chao8v8 \\
        --out results/fig_forest.pdf --dpi 200
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = list(range(16))


def load_cendw(tag):
    """Per-seed mean windowed C_end per arm (from eval_final jsons)."""
    cend = {"intrinsic": {}, "no_intrinsic": {}}
    for seed in SEEDS:
        for arm, suf in (("intrinsic", "intr"), ("no_intrinsic", "noi")):
            fp = glob.glob(
                f"results/campaign_{tag}/eval_final/{tag}_s{seed:02d}_{suf}.json")
            if not fp:
                continue
            with open(fp[0]) as f:
                j = json.load(f)
            cend[arm][seed] = float(np.mean(j["cend_w"]))
    return cend


def load_seqloc(tag, regime):
    """Per-seed seq_loc per arm (from eval_localization seq ep5 json)."""
    seq = {"intrinsic": {}, "no_intrinsic": {}}
    fp = glob.glob(
        f"results/campaign_{tag}/eval_localization/{regime}_seq_ep5.json")
    if not fp:
        return seq
    with open(fp[0]) as f:
        rows = json.load(f)
    for r in rows:
        arm = "intrinsic" if r["arm"] == "intrinsic" else "no_intrinsic"
        seq[arm][int(r["seed"])] = float(r["seq_loc"])
    return seq


def bootstrap_ci(diffs, n_boot=20000, alpha=0.10):
    rng = np.random.default_rng(0)
    d = np.asarray(diffs)
    m = []
    for _ in range(n_boot):
        m.append(np.mean(rng.choice(d, size=len(d), replace=True)))
    m = np.array(m)
    lo = np.percentile(m, 100 * alpha / 2)
    hi = np.percentile(m, 100 * (1 - alpha / 2))
    return lo, hi


def panel(ax, per_arm, metric_label, ci90, ci95):
    intr = np.array([per_arm["intrinsic"][s] for s in SEEDS])
    noi = np.array([per_arm["no_intrinsic"][s] for s in SEEDS])
    diffs = intr - noi
    ax.axvline(0.0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.axvspan(ci95[0], ci95[1], color="#1f77b4", alpha=0.10, zorder=1)
    ax.axvspan(ci90[0], ci90[1], color="#1f77b4", alpha=0.14, zorder=1)
    y = np.arange(len(SEEDS))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in diffs]
    ax.hlines(y, xmin=np.minimum(diffs, 0), xmax=np.maximum(diffs, 0),
              color="#444444", lw=0.8, zorder=2)
    ax.scatter(diffs, y, c=colors, s=28, edgecolors="#333333", linewidths=0.4,
               zorder=3)
    # pooled mean marker
    m = np.mean(diffs)
    ax.axvline(m, color="#7f7f7f", lw=1.0, zorder=2)
    ax.text(m, -1.8, f"mean {m:+.3f}", fontsize=8, ha="center",
            va="center", color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([str(s) for s in SEEDS], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(-2.5, len(SEEDS) - 0.5)
    ax.set_xlabel(f"{metric_label}  deltas (intr $-$ noi)")
    ax.set_title(metric_label, fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--regime", default="MATE-8v8-9-v0")
    ap.add_argument("--out", default="results/fig_forest.pdf")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    cendw = load_cendw(args.tag)
    seqloc = load_seqloc(args.tag, args.regime)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.4))

    d_c = np.array([cendw["intrinsic"][s] - cendw["no_intrinsic"][s]
                    for s in SEEDS])
    d_s = np.array([seqloc["intrinsic"][s] - seqloc["no_intrinsic"][s]
                    for s in SEEDS])
    ci90_c = bootstrap_ci(d_c)
    ci95_c = bootstrap_ci(d_c, alpha=0.05)
    ci90_s = bootstrap_ci(d_s)
    ci95_s = bootstrap_ci(d_s, alpha=0.05)
    panel(ax1, cendw, "C$_{end}$ (windowed)", ci90_c, ci95_c)
    panel(ax2, seqloc, "seq\\_loc", ci90_s, ci95_s)

    fig.suptitle("8v8 per-seed paired deltas (n=16) with pooled bootstrap CI",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved -> {args.out}")
    print(f"C_end(w): mean={d_c.mean():+.3f} 90%{tuple(round(x,3) for x in ci90_c)} "
          f"95%{tuple(round(x,3) for x in ci95_c)}")
    print(f"seq_loc : mean={d_s.mean():+.3f} 90%{tuple(round(x,3) for x in ci90_s)} "
          f"95%{tuple(round(x,3) for x in ci95_s)}")


if __name__ == "__main__":
    main()