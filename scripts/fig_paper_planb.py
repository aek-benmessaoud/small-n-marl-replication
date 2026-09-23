"""scripts/fig_paper_planb.py - generate all Plan-B paper figures from persisted JSONs.

Figures:
  fig1_forest.pdf  : per-seed paired seq_loc deltas for C1 (3 treatments 4v8 + 8v8 confirm)
  fig2_lambda.pdf  : lambda sweep (intr C_end / U_end vs ratio)
  fig3_bins.pdf    : bin-sensitivity (dC vs K, greedy reference; + structural probe offsets)
  fig4_sens.pdf    : sensitivity control (seq_loc k8/k6/k4 trained vs ablated vs random)
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load_seq(tag, arm_a, arm_b, suffix="s16_31"):
    fs = glob.glob(os.path.join(RES, f"campaign_{tag}/eval_localization/*seq_ep5_{suffix}.json"))
    if fs:
        fp = fs[0]
    else:
        fp = glob.glob(os.path.join(RES, f"campaign_{tag}/eval_localization/*seq_ep5.json"))[0]
    with open(fp) as f:
        rows = json.load(f)
    d = {}
    for r in rows:
        d.setdefault(r["seed"], {})[r["arm"]] = r["seq_loc"]
    seeds = sorted(d)
    A = np.array([d[s][arm_a] for s in seeds])
    B = np.array([d[s][arm_b] for s in seeds])
    return seeds, A, B


def final_eval(tag, seed, arm):
    fs = glob.glob(os.path.join(RES, f"campaign_{tag}_s{seed:02d}_{arm}/*/results.json"))
    if not fs:
        return None
    with open(fs[0]) as f:
        return json.load(f)["final_eval"]


def agg_arm(tag, arm, seeds):
    rs = [final_eval(tag, s, arm) for s in seeds]
    rs = [r for r in rs if r and r["mean_U_end"] is not None]
    if not rs:
        return None, None, None
    u = np.mean([r["mean_U_end"] for r in rs])
    c = np.mean([r["mean_C_end"] for r in rs])
    return u, c, np.mean([r["mean_return"] for r in rs])


# Shared no-intrinsic control (seeds 0-3): identical for lambda/bins sweeps by
# construction (no intrinsic reward, same config). The l2_0 paired control was
# resumed mid-training after a power cut (step 24500 checkpoint), so it is NOT
# the same policy; we use the clean shared control for the \lambda x2 row.
SHARED_CTRL_C = 0.3237
SHARED_CTRL_U = 1.6667


# ----------------------------------------------------------------------------
# FIG 1 - forest, per-seed paired seq_loc deltas
# ----------------------------------------------------------------------------
def fig1(out="fig1_forest.pdf"):
    cases = [
        ("8v8 confirm  n=32", "chao8v8", "intrinsic", "no_intrinsic",
         [s for s in range(16, 48)]),
        ("8v8 first n=16", "chao8v8", "intrinsic", "no_intrinsic",
         [s for s in range(16)]),
        ("4v8 Chao-U  n=16", "chao4v8c", "intrinsic", "no_intrinsic",
         [s for s in range(16, 32)]),
        ("4v8 loc     n=16", "loc4v8c", "loc", "no_intrinsic",
         [s for s in range(16, 32)]),
        ("4v8 RND     n=16", "rnd4v8c", "rnd", "no_intrinsic",
         [s for s in range(16, 32)]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharey=False)
    for ax, (title, tag, a, b, seeds) in zip(axes, cases):
        js = glob.glob(os.path.join(RES, f"campaign_{tag}/eval_localization/*seq_ep5*.json"))
        d = {}
        for cand in js:
            with open(cand) as f:
                rows = json.load(f)
            for r in rows:
                d.setdefault(r["seed"], {})[r["arm"]] = r["seq_loc"]
        ss = [s for s in seeds if s in d and a in d[s] and b in d[s]]
        diffs = np.array([d[s][a] - d[s][b] for s in ss])
        ax.axvline(0, color="#888", lw=0.8, ls="--", zorder=1)
        y = np.arange(len(ss))
        cols = ["#2ca02c" if v > 0 else "#d62728" for v in diffs]
        ax.hlines(y, np.minimum(diffs, 0), np.maximum(diffs, 0),
                  color="#444", lw=0.7, zorder=2)
        ax.scatter(diffs, y, c=cols, s=26, edgecolors="#333", linewidths=0.4, zorder=3)
        m = diffs.mean()
        ax.axvline(m, color="#7f7f7f", lw=1.1, zorder=2)
        ax.text(m, -1.6, f"{m:+.3f}", fontsize=9, ha="center", va="center",
                color="#222", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([str(s) for s in ss], fontsize=6.5)
        ax.invert_yaxis()
        ax.set_ylim(-2.3, len(ss) - 0.5)
        ax.set_xlabel("Δ seq\\_loc  (treat $-$ ctrl)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Per-seed paired seq\\_loc deltas, treatment vs. shared no-intrinsic control",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(RES, out), dpi=200)
    plt.close(fig)
    print(f"saved -> {RES}/{out}")


# ----------------------------------------------------------------------------
# FIG 2 - lambda sweep
# ----------------------------------------------------------------------------
def fig2(out="fig2_lambda.pdf"):
    ratios = [("chao8v8_l0_5", "λ/2", 0.5), ("chao8v8_l1_0", "λ×1", 1.0),
              ("chao8v8_l2_0", "λ×2", 2.0)]
    xs, cs, us = [], [], []
    for tag, lab, r in ratios:
        ui, ci, _ = agg_arm(tag, "intrinsic", range(4))
        un, cn, _ = SHARED_CTRL_U, SHARED_CTRL_C, None
        xs.append(r); cs.append(ci - cn); us.append(ui - un)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(xs, cs, "o-", color="#1f77b4", lw=1.6, ms=6)
    ax1.axhline(0, color="#888", ls="--", lw=0.8)
    ax1.set_xlabel("reward scaling λ (auto-calibrated = ×1)")
    ax1.set_ylabel("ΔC$_{end}$  (intr $-$ ctrl)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["λ/2", "λ×1", "λ×2"])
    ax1.set_title("Configuration diversity", fontsize=10)
    ax1.grid(alpha=0.3)
    ax2.plot(xs, us, "o-", color="#d62728", lw=1.6, ms=6)
    ax2.axhline(0, color="#888", ls="--", lw=0.8)
    ax2.set_xlabel("reward scaling λ (auto-calibrated = ×1)")
    ax2.set_ylabel("ΔU$_{end}$  (intr $-$ ctrl)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["λ/2", "λ×1", "λ×2"])
    ax2.set_title("Remaining work (Chao-U)", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, out), dpi=200)
    plt.close(fig)
    print(f"saved -> {RES}/{out}")


# ----------------------------------------------------------------------------
# FIG 3 - bins sweep with greedy nominal reference
# ----------------------------------------------------------------------------
def fig3(out="fig3_bins.pdf"):
    ks = ["chao8v8_b08", "chao8v8_b12", "chao8v8_b16", "chao8v8_b24"]
    labels = ["K=8", "K=12", "K=16", "K=24"]
    cs = []
    for tag in ks:
        ui, ci, _ = agg_arm(tag, "intrinsic", range(4))
        un, cn, _ = agg_arm(tag, "no_intrinsic", range(4))
        cs.append(ci - cn if ui is not None else np.nan)
    ui, ci, _ = agg_arm("chao8v8", "intrinsic", range(4))
    un, cn, _ = agg_arm("chao8v8", "no_intrinsic", range(4))
    greedy_dc = ci - cn
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(ks))
    ax.bar(x, cs, width=0.55, color=["#9ecae1"] * len(ks),
           edgecolor="#3182bd", lw=0.8)
    ax.axhline(greedy_dc, color="#d62728", lw=1.5, ls="--",
               label=f"greedy nominal ΔC$_{{end}}$ = {greedy_dc:+.4f}")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("fixed-grid angular bins (8/12/16/24 = 45/30/22.5/15° per bin)")
    ax.set_ylabel("ΔC$_{end}$  (intr $-$ ctrl)")
    ax.set_title("Bin-sensitivity of the configuration counter", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, out), dpi=200)
    plt.close(fig)
    print(f"saved -> {RES}/{out}")


# ----------------------------------------------------------------------------
# FIG 4 - sensitivity control: seq_loc by angular tolerance k8/k6/k4
# ----------------------------------------------------------------------------
def fig4(out="fig4_sens.pdf"):
    ab = {}; ra = {}
    for f in glob.glob(os.path.join(RES, "campaign_chao8v8/sens_ablate_ep5_*.json")):
        for r in json.load(open(f)):
            ab.setdefault(r["tag"], []).append(r)
    for f in glob.glob(os.path.join(RES, "campaign_chao8v8/sens_random_ep5_*.json")):
        for r in json.load(open(f)):
            ra.setdefault(r["tag"], []).append(r)
    klabels = ["k8 (15°)", "k6 (30°)", "k4 (60°)"]
    trained = [np.mean([r[f"seq_loc_k{k}"] for r in ab["intrinsic"]]) for k in (8, 6, 4)]
    random = [np.mean([r[f"seq_loc_k{k}"] for r in ra["random"]]) for k in (8, 6, 4)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(3)
    ax.bar(x - 0.18, trained, 0.34, color="#1f77b4", label="intrinsic-trained",
           edgecolor="#333", lw=0.8)
    ax.bar(x + 0.18, random, 0.34, color="#d62728", label="random policy",
           edgecolor="#333", lw=0.8)
    for xi, v in zip(x - 0.18, trained):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.18, random):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(klabels)
    ax.set_xlabel("angular tolerance (cluster criterion)")
    ax.set_ylabel("seq\\_loc")
    ax.set_title("Sensitivity control: signal is not inert", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, out), dpi=200)
    plt.close(fig)
    print(f"saved -> {RES}/{out}")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()