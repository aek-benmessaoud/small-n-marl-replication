"""
scripts/agg_localization.py — Paired aggregate of eval_localization JSONs.

Reads results/campaign_<tag>/eval_localization/<regime>_<mode>_ep<n>.json
(merged per-seed rows) and prints the paired treatment-vs-control aggregate.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT_REGIME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--arms", default="intrinsic,no_intrinsic")
    args = ap.parse_args()

    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]
    out_dir = os.path.join("results", f"campaign_{args.tag}",
                           "eval_localization")
    for mode in ("simult", "seq"):
        path = os.path.join(out_dir,
                            f"{args.regime}_{mode}_ep{args.episodes}.json")
        if not os.path.exists(path):
            print(f"[{mode}] no file {path}")
            continue
        rows = json.load(open(path))
        if mode == "seq":
            metrics = ("seq_loc", "seq_peak", "seq_err")
        else:
            metrics = ("simult_loc", "loc_peak", "loc_final", "gn_err")
        intr = [r for r in rows if r["arm"] == arm_a]
        noi = [r for r in rows if r["arm"] == arm_b]
        print(f"=== {mode} ({arm_a} vs {arm_b}, n={len(intr)}) ===")
        for metric in metrics:
            a = [r[metric] for r in intr if np.isfinite(r[metric])]
            b = [r[metric] for r in noi if np.isfinite(r[metric])]
            if not a or not b:
                print(f"{metric:<10} incomplete")
                continue
            wins = sum(x > y for x, y in zip(a, b))
            print(f"{metric:<10} {arm_a[:4]}={np.mean(a):+.4f}  "
                  f"{arm_b[:4]}={np.mean(b):+.4f}  "
                  f"delta={np.mean(a) - np.mean(b):+.4f}  wins={wins}/{len(a)}")


if __name__ == "__main__":
    main()
