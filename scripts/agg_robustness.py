"""
scripts/agg_robustness.py -- Bootstrap CI + per-seed table + wave analysis
for campaign_chao8v8 (paired intr vs no_intrinsic).

Reads existing eval_final JSONs and eval_localization JSONs. No rollouts.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")


RNG = np.random.default_rng(20260917)


def paired_from_eval_final(tag, arm_a="intrinsic", arm_b="no_intrinsic"):
    files = sorted(glob.glob(os.path.join(
        "results", f"campaign_{tag}", "eval_final", f"{tag}_s*.json")))
    groups = {}
    for f in files:
        r = json.load(open(f))
        groups.setdefault((r["arm"], r["seed"]), r)
    seeds = sorted({k[1] for k in groups})
    data = {}
    for key, name in [("rets", "true_rew"), ("seens", "mean_seen"),
                      ("uend_w", "U_end_w"), ("cend_w", "C_end_w"),
                      ("cend_c", "C_end_cum")]:
        a = np.array([np.mean(groups[(arm_a, s)][key]) for s in seeds])
        b = np.array([np.mean(groups[(arm_b, s)][key]) for s in seeds])
        data[name] = {"a": a, "b": b, "d": a - b, "seeds": seeds}
    return seeds, data


def paired_from_loc(tag, mode):
    if mode == "simult":
        keys = ["simult_loc", "loc_peak", "loc_final", "gn_err"]
    else:
        keys = ["seq_loc", "seq_peak", "seq_err"]
    path = os.path.join("results", f"campaign_{tag}", "eval_localization",
                        f"MATE-8v8-9-v0_{mode}_ep5.json")
    recs = json.load(open(path))
    groups = {(r["seed"], r["arm"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    out = {}
    for key in keys:
        a = np.array([groups[(s, "intrinsic")][key] for s in seeds])
        b = np.array([groups[(s, "no_intrinsic")][key] for s in seeds])
        out[key] = {"a": a, "b": b, "d": a - b, "seeds": seeds}
    return seeds, out


def scan_key(key):
    """Heuristic: bigger is better for loc metrics, smaller for errors."""
    if key in ("gn_err",):
        return "lower"
    return "higher"


def quick_wilcoxon(a, b):
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        p = 1.0
    return p


def bootstrap_ci(d, n_boot=20000, alpha=0.10):
    """Bootstrap over seeds for the mean paired difference delta."""
    d = np.asarray(d)
    n = len(d)
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    bias = means.mean() - d.mean()
    return lo, hi, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--waves", default="0-3,4-7,8-11,12-15")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="significance level for CI (0.10 = 90%%, 0.05 = 95%%)")
    ap.add_argument("--latex", action="store_true",
                    help="print a LaTeX appendix table of per-seed diffs")
    args = ap.parse_args()

    seeds, base = paired_from_eval_final(args.tag)
    _, loc = paired_from_loc(args.tag, "simult")
    _, seq = paired_from_loc(args.tag, "seq")

    metrics = ["true_rew", "mean_seen", "U_end_w", "C_end_w", "C_end_cum",
               "simult_loc", "loc_peak", "loc_final", "seq_loc", "seq_err"]
    tables = {"seq_loc": seq, "seq_err": seq}
    all_d = {m: (base if m in base else (tables[m] if m in tables else loc))[m]["d"]
             for m in metrics}
    refs = {"true_rew": "ret", "mean_seen": "seen", "U_end_w": "uend",
            "C_end_w": "cend", "C_end_cum": "cend_c"}

    pct = int((1 - args.alpha) * 100)
    print(f"=== BOOTSTRAP {pct}% CI on paired mean delta | tag={args.tag} "
          f"| n={len(seeds)} | seeds={seeds} | B={20000} | alpha={args.alpha} ===")
    for m in metrics:
        d = all_d[m]
        tbl = base if m in base else (tables[m] if m in tables else loc)
        lo, hi, bias = bootstrap_ci(d, n_boot=20000, alpha=args.alpha)
        p = quick_wilcoxon(tbl[m]["a"], tbl[m]["b"])
        wins = int(np.sum(d > 0))
        eq = int(np.sum(d == 0))
        sign = "+" if lo > 0 else ("-" if hi < 0 else "~")
        print(f"{m:<12} meanΔ={d.mean():+.4f}  wins={wins}/{len(d)} eq={eq}"
              f"  wilcoxon p={p:.3f}  {pct}%CI=[{lo:+.4f}, {hi:+.4f}]"
              f"  bias={bias:+.4f}  -> {sign}")

    print()
    print("=== PER-SEED PAIRED DIFFERENCES (intr - no_intr) ===")
    hdr = "seed  " + "".join(f"{m:>11}" for m in metrics)
    print(hdr)
    for i, s in enumerate(seeds):
        row = f"{s:<5}" + "".join(
            f"{all_d[m][i]:+11.4f}" for m in metrics)
        print(row)

    print()
    print("=== WAVE ANALYSIS (blocks of 4 seeds) ===")
    for wave in args.waves.split(","):
        lo_s, hi_s = map(int, wave.split("-"))
        sl = [s for s in seeds if lo_s <= s <= hi_s]
        if not sl:
            continue
        ii = [seeds.index(s) for s in sl]
        line = f"wave {wave} (n={len(sl)}):"
        for m in metrics:
            d = all_d[m][ii]
            wins = int(np.sum(d > 0))
            line += f"  {m}={d.mean():+.4f}({wins}/{len(d)})"
        print(line)

    if args.latex:
        cols = ["C$_{\\text{end}}$ (w)", "C$_{\\text{end}}$ (cum)",
                "seq\\_loc", "simult\\_loc", "true\\_rew"]
        lk = {"C$_{\\text{end}}$ (w)": "C_end_w",
              "C$_{\\text{end}}$ (cum)": "C_end_cum",
              "seq\\_loc": "seq_loc", "simult\\_loc": "simult_loc",
              "true\\_rew": "true_rew"}
        print()
        print("=== LATEX APPENDIX TABLE (per-seed paired diffs) ===")
        print("\\begin{table*}[t]")
        print("\\centering")
        print("\\caption{Per-seed paired differences "
              "(intrinsic $-\\;$ no\\_intrinsic) in 8v8, $n{=}16$. "
              "Each column shows the same seed evaluated under both arms, "
              "5 episodes deterministic.}")
        print("\\label{tab:per-seed}")
        print("\\begin{tabular}{r" + "c" * len(cols) + "}")
        print("\\toprule")
        print("Seed & " + " & ".join(cols) + " \\\\")
        print("\\midrule")
        for i, s in enumerate(seeds):
            cells = []
            for c in cols:
                key = lk[c]
                cells.append(f"{all_d[key][i]:+.4f}")
            print(f"{s} & " + " & ".join(cells) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table*}")


if __name__ == "__main__":
    main()