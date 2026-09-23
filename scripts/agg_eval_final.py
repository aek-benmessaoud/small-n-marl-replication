"""
scripts/agg_eval_final.py — Paired aggregation of existing eval_final JSONs.

Reads results/campaign_<tag>/eval_final/*.json written by eval_final.py and
prints the same Wilcoxon paired aggregation, without re-running rollouts.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", default="intrinsic,no_intrinsic")
    args = ap.parse_args()

    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]
    files = sorted(glob.glob(
        os.path.join("results", f"campaign_{args.tag}", "eval_final",
                     f"{args.tag}_s*.json")))
    if not files:
        print(f"no eval_final JSONs for campaign_{args.tag}")
        return
    groups = {}
    for f in files:
        r = json.load(open(f))
        groups.setdefault((r["arm"], r["seed"]), r)

    intr = [groups[k] for k in sorted(groups) if k[0] == arm_a]
    noi = [groups[k] for k in sorted(groups) if k[0] == arm_b]
    if not intr or not noi:
        print("incomplete data", [(k[0], k[1]) for k in groups])
        return
    if len(intr) != len(noi):
        print(f"mismatched seeds: {len(intr)} vs {len(noi)}")

    print(f"=== AGGREGATE {args.tag} ({arm_a} vs {arm_b}, paired, "
          f"n={min(len(intr), len(noi))}) ===")
    for name, key in [("vraie récompense", "rets"),
                      ("mean_seen", "seens"),
                      ("U_end fenêtré", "uend_w"),
                      ("C_end fenêtré", "cend_w"),
                      ("C_end cumulatif", "cend_c")]:
        a = np.array([np.mean(r[key]) for r in intr])
        b = np.array([np.mean(r[key]) for r in noi])
        wins = int(np.sum(a > b))
        eq = int(np.sum(a == b))
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        print(f"{name:<16} {arm_a[:4]}={a.mean():+.3f}  "
              f"{arm_b[:4]}={b.mean():+.3f}  delta={a.mean() - b.mean():+.3f}  "
              f"wins={wins}/{len(a)} eq={eq}  p={p:.3f}")


if __name__ == "__main__":
    main()
