"""
scripts/agg_p1_slow.py -- Slow-env (quasi-static) 8v8 comparison for chao-win.

Merges the per-range eval_localization split JSONs written by
eval_slow_split.cmd (MATE-8v8-9-slow-v0_{mode}_ep5_{range}.json) and compares
paired intr-no_intrinsic deltas on the slow environment against the fast
control (MATE-8v8-9-v0, same env as training). Prints a LaTeX-ready table.
No rollouts.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.analysis.agg_robustness import bootstrap_ci

sys.stdout.reconfigure(encoding="utf-8")

PREFIX = "MATE-8v8-9-slow-v0"


def merge_slow(tag, mode, ranges):
    recs = []
    for rng in ranges:
        path = os.path.join("results", f"campaign_{tag}", "eval_localization",
                            f"{PREFIX}_{mode}_ep5_{rng}.json")
        if os.path.exists(path):
            recs.extend(json.load(open(path)))
    groups = {(r["seed"], r["arm"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    return groups, seeds


def load_fast(tag, mode):
    path = os.path.join("results", f"campaign_{tag}", "eval_localization",
                        f"MATE-8v8-9-v0_{mode}_ep5.json")
    recs = json.load(open(path))
    groups = {(r["seed"], r["arm"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    return groups, seeds


def paired(groups, seeds, key):
    a = np.array([groups[(s, "intrinsic")][key] for s in seeds])
    b = np.array([groups[(s, "no_intrinsic")][key] for s in seeds])
    return a, b, a - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--ranges", default="0..3,4..7,8..11,12..15")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    ranges = args.ranges.split(",")

    print(f"=== SLOW (quasi-static) vs FAST control | tag={args.tag} "
          f"| {PREFIX} ===")
    for mode, keys, emp in [("simult", ["simult_loc"], ""),
                            ("seq", ["seq_loc", "seq_peak"], "")]:
        groups, seeds = merge_slow(args.tag, mode, ranges)
        if not groups:
            print(f"  [{mode}] no slow data yet for {ranges}")
            continue
        f_groups, _ = load_fast(args.tag, mode)
        print()
        print(f"--- {mode} (n={len(seeds)}) "
              f"seeds={seeds} ---")
        for key in keys:
            if key not in groups[(seeds[0], "intrinsic")]:
                continue
            a, b, d = paired(groups, seeds, key)
            fa, fb, fd = paired(f_groups, seeds, key)
            lo, hi, bias = bootstrap_ci(d, n_boot=20000, alpha=args.alpha)
            flo, fhi, fbias = bootstrap_ci(fd, n_boot=20000, alpha=args.alpha)
            wins = int(np.sum(d > 0))
            sign = "+" if lo > 0 else ("-" if hi < 0 else "~")
            print(f"  {key:<10} slowÎ”={d.mean():+.4f} wins={wins}/{len(d)} "
                  f"{int((1-args.alpha)*100)}%CI=[{lo:+.4f},{hi:+.4f}] {sign}"
                  f"   |  fastÎ”={fd.mean():+.4f} wins={int(np.sum(fd>0))}/"
                  f"{len(fd)} CI=[{flo:+.4f},{fhi:+.4f}]")
            for i, s in enumerate(seeds):
                print(f"      seed {s:<3} slow intr={a[i]:.5f} "
                      f"noi={b[i]:.5f} Î”={d[i]:+.5f}   |  fast Î”={fd[i]:+.5f}")


if __name__ == "__main__":
    main()