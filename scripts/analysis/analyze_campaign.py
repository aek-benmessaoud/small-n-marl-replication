"""
scripts/analyze_campaign.py — Phase 6 analysis of the global campaign.

Reads results/campaign_campaign_<tag>/<timestamp>/ for all seeds and writes:
  * analysis.json  (stats: Wilcoxon signed-rank, Holm-Bonferroni, baselines)
  * console summary

Comparisons (pre-registered):
  1. coverage : mean final eval-return (5 episodes)  intrinsic vs no_intrinsic
     (Wilcoxon signed-rank, paired by seed)
  2. localization : mean U_end (flat floor expected -> reported, not tested)
  3. MAPPO (best arm on coverage) vs heuristic baselines (two-sample
     Mann-Whitney, Bonferroni across the 4 baselines)

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_campaign.py --tag campaign --ts 20260808_000125
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RESULTS_DIR  # noqa: E402


def load_arm(tag, ts, arm_pat):
    base_pat = os.path.join(RESULTS_DIR, f"campaign_{tag}_s*")
    out = {}
    for f in glob.glob(base_pat):
        m = re.match(rf"^campaign_{tag}_s(\d+)_{arm_pat}$", os.path.basename(f))
        if not m:
            continue
        jp = os.path.join(f, ts, "results.json")
        if not os.path.exists(jp):
            continue
        with open(jp) as fh:
            j = json.load(fh)
        fe = j.get("final_eval")
        if not fe:
            continue
        out[int(m.group(1))] = {
            "mean_return": fe["mean_return"],
            "mean_U_end": fe.get("mean_U_end"),
            "mean_C_end": fe.get("mean_C_end"),
            "curve": j.get("curve", {}),
        }
    return out


def wilcoxon_signed_rank(x, y):
    """Exact/approx Wilcoxon signed-rank (one-sample on the diffs)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 1.0, 0.0, n
    from scipy.stats import rankdata
    r = rankdata(np.abs(nz))
    w = float(np.sum(r[nz > 0]))
    # normal approx with continuity correction + tie handling
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # tie correction
    r_abs = np.abs(nz)
    unique = np.unique(r_abs)
    tie = 0.0
    for u in unique:
        c = int((r_abs == u).sum())
        tie += c * (c - 1) * (c + 1) / 2.0
    var -= tie / 48.0
    z = (w - mu + (0.5 if w < mu else -0.5)) / np.sqrt(var)
    from scipy.stats import norm
    p_two = 2.0 * (1.0 - norm.cdf(abs(z)))
    return p_two, w, n


def mann_whitney(x, y):
    from scipy.stats import mannwhitneyu
    try:
        u, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def holm_bonferroni(ps):
    """Holm-Bonferroni adjusted p-values (no statsmodels dependency)."""
    ps = np.asarray(ps, dtype=float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, ps[i] * (m - rank))
    for rank in range(m - 2, -1, -1):
        i, j = order[rank], order[rank + 1]
        adj[i] = max(adj[i], adj[j])
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--ts", default=None)
    args = ap.parse_args()

    ts = args.ts
    if ts is None:
        first = glob.glob(os.path.join(RESULTS_DIR, f"campaign_{args.tag}_s00_intrinsic", "*"))[0]
        ts = os.path.basename(first)
    ts_dir = os.path.join(RESULTS_DIR, f"campaign_{args.tag}_s00_intrinsic", ts)

    intr = load_arm(args.tag, ts, "intrinsic")
    nointr = load_arm(args.tag, ts, "no_intrinsic")
    seeds = sorted(set(intr) & set(nointr))

    report = {"tag": args.tag, "timestamp": ts, "n": len(seeds)}

    if not seeds:
        print("No paired data found.")
        return

    # ---- 1. coverage: intrinsic vs no_intrinsic ---------------------------
    xi = np.array([intr[s]["mean_return"] for s in seeds])
    xo = np.array([nointr[s]["mean_return"] for s in seeds])
    p_two, w, nz = wilcoxon_signed_rank(xi, xo)
    d = xi - xo
    from scipy.stats import rankdata
    # signed-rank for "intrinsic > no_intrinsic" (one-sided)
    p_one = p_two / 2 if np.median(d) >= 0 else 1.0 - p_two / 2
    eff = 2 * (d > 0).mean() - 1  # concordance (-1..1)
    report["coverage"] = {
        "intrinsic_mean": round(float(xi.mean()), 3),
        "intrinsic_std": round(float(xi.std()), 3),
        "no_intrinsic_mean": round(float(xo.mean()), 3),
        "no_intrinsic_std": round(float(xo.std()), 3),
        "mean_paired_diff": round(float(d.mean()), 3),
        "median_paired_diff": round(float(np.median(d)), 3),
        "wins_intrinsic": int((d > 0).sum()),
        "wins_no_intrinsic": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "wilcoxon_W": round(w, 1),
        "wilcoxon_n": int(nz),
        "p_two_sided": round(float(p_two), 6),
        "p_one_sided_intr_gt": round(float(p_one), 6),
        "concordance": round(float(eff), 4),
    }
    print("=== COVERAGE (mean final eval-return, paired) ===")
    print(f"  intrinsic   : {xi.mean():.2f} +- {xi.std():.2f}")
    print(f"  no_intrinsic: {xo.mean():.2f} +- {xo.std():.2f}")
    print(f"  paired diff : {d.mean():.2f} (median {np.median(d):.2f}), "
          f"wins {int((d>0).sum())} vs {int((d<0).sum())} (ties {int((d==0).sum())})")
    print(f"  Wilcoxon    : W={w:.1f} n={nz}  p2={p_two:.4f}  p1(intr>nointr)={p_one:.4f}")

    # ---- 2. localization: U_end / C_end ------------------------------------
    ui = np.array([intr[s]["mean_U_end"] for s in seeds], dtype=float)
    uo = np.array([nointr[s]["mean_U_end"] for s in seeds], dtype=float)
    ci = np.array([intr[s]["mean_C_end"] if intr[s]["mean_C_end"] is not None
                   else np.nan for s in seeds], dtype=float)
    co = np.array([nointr[s]["mean_C_end"] if nointr[s]["mean_C_end"] is not None
                   else np.nan for s in seeds], dtype=float)
    uniq_i = np.unique(ui)
    uniq_o = np.unique(uo)
    report["localization"] = {
        "intrinsic_U_end_unique": [float(v) for v in uniq_i],
        "no_intrinsic_U_end_unique": [float(v) for v in uniq_o],
        "intrinsic_C_end": {"mean": round(float(np.nanmean(ci)), 4)
                            if np.isfinite(ci).any() else None,
                            "std": round(float(np.nanstd(ci)), 4)
                            if np.isfinite(ci).any() else None},
        "no_intrinsic_C_end": {"mean": round(float(np.nanmean(co)), 4)
                               if np.isfinite(co).any() else None,
                               "std": round(float(np.nanstd(co)), 4)
                               if np.isfinite(co).any() else None},
        "note": "flat floor (1.0) -> U test skipped, report only; C_end is the "
                "circular-variance diversity, reported as the dense metric",
    }
    print("=== LOCALIZATION (mean U_end / C_end) ===")
    print(f"  intrinsic    unique U_end: {uniq_i}   C_end: {np.nanmean(ci):.4f} +- {np.nanstd(ci):.4f}")
    print(f"  no_intrinsic unique U_end: {uniq_o}   C_end: {np.nanmean(co):.4f} +- {np.nanstd(co):.4f}")

    # ---- 3. MAPPO vs baselines --------------------------------------------
    bl = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, f"campaign_{args.tag}_s*")):
        m = re.match(rf"^campaign_{args.tag}_s(\d+)_baselines$", os.path.basename(f))
        if not m:
            continue
        bjp = os.path.join(f, ts, "baselines.json")
        if not os.path.exists(bjp):
            continue
        with open(bjp) as fh:
            bj = json.load(fh)
        seed = int(m.group(1))
        for name, v in bj.get("results", {}).items():
            bl.setdefault(name, {})[seed] = v["mean_return"]

    # pick the better MAPPO arm (higher mean coverage return) for the
    # baseline comparison
    better_arm = "intrinsic" if xi.mean() >= xo.mean() else "no_intrinsic"
    mappo_vals = xi if better_arm == "intrinsic" else xo
    report["vs_baselines"] = {"mappo_arm": better_arm, "baselines": {}}
    print(f"=== VS BASELINES (Mann-Whitney, {better_arm} MAPPO vs 4 baselines) ===")
    from scipy.stats import mannwhitneyu
    for name, by_seed in sorted(bl.items()):
        bvals = np.array([by_seed[s] for s in seeds if s in by_seed])
        try:
            u, p = mannwhitneyu(mappo_vals, bvals, alternative="two-sided")
        except ValueError:
            u, p = None, 1.0
        report["vs_baselines"]["baselines"][name] = {
            "mean": round(float(bvals.mean()), 3),
            "n": int(len(bvals)),
            "p": round(float(p), 6),
        }
        print(f"  {name:<16} mean={bvals.mean():9.2f} n={len(bvals)}  p={p:.4f}")

    # ---- Holm-Bonferroni over the 4 baseline comparisons -------------------
    ps = [v["p"] for v in report["vs_baselines"]["baselines"].values()]
    names = list(report["vs_baselines"]["baselines"].keys())
    if ps:
        padj = holm_bonferroni(ps)
        report["vs_baselines"]["holm_adjusted_p"] = {
            n: round(float(p), 6) for n, p in zip(names, padj)
        }
    print(f"  Holm-adjusted: {report['vs_baselines'].get('holm_adjusted_p', {})}")

    # ---- learning curves (mean +- sem over seeds) ---------------------------
    curve_pts = sorted({pt for s in seeds for pt in intr[s]["curve"].get("step", [])})
    curves = {}
    for arm, data in [("intrinsic", intr), ("no_intrinsic", nointr)]:
        steps, means, sems, Us = [], [], [], []
        for pt in curve_pts:
            vals = [data[s]["curve"]["eval_return"][
                data[s]["curve"]["step"].index(pt)]
                for s in seeds if pt in data[s]["curve"]["step"]]
            if not vals:
                continue
            steps.append(pt)
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(len(vals))))
        curves[arm] = {"step": steps, "mean": means, "sem": sems}
    report["learning_curves"] = curves
    print("=== LEARNING CURVES (mean eval-return) ===")
    for arm in ["intrinsic", "no_intrinsic"]:
        c = curves[arm]
        print(f"  {arm:<14}: " + "  ".join(
            f"{s}:{m:.0f}" for s, m in zip(c["step"], c["mean"])))

    out_dir = os.path.join(RESULTS_DIR, f"campaign_{args.tag}_s00_intrinsic", ts)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[ANALYSIS-OK] wrote {os.path.join(out_dir, 'analysis.json')}")


if __name__ == "__main__":
    main()
