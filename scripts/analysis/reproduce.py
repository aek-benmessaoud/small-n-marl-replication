"""scripts/analysis/reproduce.py - regenerate every number in the paper from
results_digests/ (the committed, reviewer-accessible data subset).

No training, no rollouts: the digests contain the eval_localization JSONs
(per-seed seq_loc), the sens_* JSONs (trained vs random policies) and the
per-seed results.json (mean_C_end / mean_U_end) used by the paper.

Usage:  .venv\\Scripts\\python.exe scripts/analysis/reproduce.py

Prints, in order, the exact values that appear in paper/src/paper.tex:
  - Table 1 (tab:c1): paired seq_loc deltas, wins, one-sided Wilcoxon p,
    90%% bootstrap CI on the paired mean, for the 8v8 waves and 4v8 arms.
  - Sensitivity control (fig4 / control section): trained vs random seq_loc
    at k8/k6/k4 and the ratios.
  - Reward-scale sweep (fig2 / tab:lambda) and bin sweep (fig3 / tab:bins).
  - Saturation claim: mean C_end (trained vs control) at n=48.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIG = os.path.join(ROOT, "results_digests")

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")

RNG = np.random.default_rng(20260917)


def loc_rows(tag):
    pat = os.path.join(DIG, f"campaign_{tag}", "eval_localization", "*seq_ep5*.json")
    cands = sorted(glob.glob(pat))
    rows = []
    for c in cands:
        with open(c) as fh:
            rows.extend(json.load(fh))
    return {(r["seed"], r["arm"]): r for r in rows}


def paired_loc(tag, seeds, arm_a="intrinsic", arm_b="no_intrinsic"):
    rows = loc_rows(tag)
    ss = sorted(s for s in seeds if (s, arm_a) in rows and (s, arm_b) in rows)
    a = np.array([rows[(s, arm_a)]["seq_loc"] for s in ss])
    b = np.array([rows[(s, arm_b)]["seq_loc"] for s in ss])
    return ss, a, b


def fmt(ss, a, b):
    d = a - b
    n = len(ss)
    wins = int((d > 0).sum())
    try:
        # paper p-values use two-sided / 2 (same convention as quick_wilcoxon)
        p = wilcoxon(a, b).pvalue / 2
    except ValueError:
        p = float("nan")
    boot = np.array([
        np.mean(d[RNG.integers(0, n, n)]) for _ in range(20000)
    ])
    lo, hi = np.percentile(boot, [5, 95])
    sd = d.std(ddof=1)
    return (f"n={n:<3d} d={d.mean():+.4f} sd={sd:.4f} wins={wins}/{n} "
            f"p_one={p:.3f} 90%CI=[{lo:+.4f},{hi:+.4f}]")


def sens_means():
    ab, ra = {}, {}
    for f in sorted(glob.glob(os.path.join(DIG, "campaign_chao8v8", "sens_ablate_ep5_*.json"))):
        for r in json.load(open(f)):
            ab.setdefault(r["tag"], []).append(r)
    for f in sorted(glob.glob(os.path.join(DIG, "campaign_chao8v8", "sens_random_ep5_*.json"))):
        for r in json.load(open(f)):
            ra.setdefault(r["tag"], []).append(r)
    t = [np.mean([r[f"seq_loc_k{k}"] for r in ab["intrinsic"]]) for k in (8, 6, 4)]
    r = [np.mean([r[f"seq_loc_k{k}"] for r in ra["random"]]) for k in (8, 6, 4)]
    return t, r


def final_arm(tag, s, arm):
    fs = glob.glob(os.path.join(DIG, f"campaign_{tag}_s{s:02d}_{arm}", "*", "results.json"))
    if not fs:
        return None
    return json.load(open(fs[0]))["final_eval"]


def agg_final(tag, arm, seeds):
    rs = [final_arm(tag, s, arm) for s in seeds]
    rs = [r for r in rs if r and r["mean_U_end"] is not None]
    if not rs:
        return None
    u = np.mean([r["mean_U_end"] for r in rs])
    c = np.mean([r["mean_C_end"] for r in rs])
    return u, c


def main():
    print("=" * 78)
    print("Reproduction check against paper/src/paper.tex")
    print("Data from results_digests/")
    print("=" * 78)

    print("\n[tab:c1] Paired seq_loc (treat - ctrl), one-sided Wilcoxon, 90% CI")
    for label, tag, arm_a, seeds in [
        ("8v8 chao n=16 (0-15)", "chao8v8", "intrinsic", range(0, 16)),
        ("8v8 chao n=32 (16-47)", "chao8v8", "intrinsic", range(16, 48)),
        ("8v8 chao n=48 (pooled)", "chao8v8", "intrinsic", range(0, 48)),
        ("4v8 chao n=16", "chao4v8c", "intrinsic", range(16, 32)),
        ("4v8 loc  n=16", "loc4v8c", "loc", range(16, 32)),
        ("4v8 rnd  n=16", "rnd4v8c", "rnd", range(16, 32)),
    ]:
        ss, a, b = paired_loc(tag, seeds, arm_a)
        print(f"  {label:26s} {fmt(ss, a, b)}")

    print("\n[Sensitivity / fig4] seq_loc trained vs random, seeds 16-47")
    t, r = sens_means()
    for k, tv, rv in zip((8, 6, 4), t, r):
        print(f"  k{k}: trained={tv:.3f}  random={rv:.3f}  ratio={tv/rv:.2f}x")

    print("\n[fig2 / lambda sweep] dC_end = intr - ctrl (shared ctrl, seeds 0-3)")
    for tag, lab in [("chao8v8_l0_5", "lam x0.5"), ("chao8v8_l1_0", "lam x1.0"),
                     ("chao8v8_l2_0", "lam x2.0")]:
        ui, ci = agg_final(tag, "intrinsic", range(4))
        print(f"  {lab:9s} intr C_end (seeds0-3)={ci:.4f}" )

    print("\n[fig3 / bin sweep] dC_end intr - ctrl (paired, seeds 0-3)")
    for tag in ["chao8v8_b08", "chao8v8_b12", "chao8v8_b16", "chao8v8_b24"]:
        ui, ci = agg_final(tag, "intrinsic", range(4))
        un, cn = agg_final(tag, "no_intrinsic", range(4))
        print(f"  {tag.replace('chao8v8_b','K='):5s} dC_end={ci - cn:+.4f}")
    ui, ci = agg_final("chao8v8", "intrinsic", range(4))
    un, cn = agg_final("chao8v8", "no_intrinsic", range(4))
    print(f"  greedy K={ci - cn:+.4f}  (nominal reference)")

    print("\n[Saturation] mean C_end at n=48")
    i = agg_final("chao8v8", "intrinsic", range(48))
    n = agg_final("chao8v8", "no_intrinsic", range(48))
    print(f"  trained={i[1]:.3f}  ctrl={n[1]:.3f}")


if __name__ == "__main__":
    main()