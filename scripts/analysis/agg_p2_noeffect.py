"""
scripts/agg_p2_noeffect.py -- Compact no-effect summary for 4v8 families.

Reads existing eval_final + eval_localization (seq) JSONs for the four 4v8
intrinsic families and prints a LaTeX-ready table: paired mean delta,
wins/ratio, and 90% seed bootstrap CI for the headline metrics.
No rollouts.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.analysis.agg_robustness import bootstrap_ci

CAMPAIGNS = [
    ("chao4v8",    "Chao discrete"),
    ("chao4v8win", "Chao windowed"),
    ("loc4v8",     "Loc objective"),
    ("rnd4v8",     "RND"),
]

ARMS = {
    "chao4v8": ("intrinsic", "no_intrinsic", "c8"),
    "chao4v8win": ("intrinsic", "no_intrinsic", "chao4v8win"),
    "loc4v8": ("loc", "no_intrinsic", "loc4v8"),
    "rnd4v8": ("rnd", "no_intrinsic", "rnd4v8"),
}


def load_eval_final(tag, prefix, arm_a, arm_b):
    files = sorted(glob.glob(os.path.join(
        "results", f"campaign_{tag}", "eval_final", f"{prefix}_s*.json")))
    groups = {}
    for f in files:
        r = json.load(open(f))
        groups.setdefault((r["seed"], r["arm"]), r)
    seeds = sorted({s for (s, _) in groups})
    out = {}
    for key in ("rets", "seens", "uend_w", "cend_w", "cend_c"):
        if any(key in r for r in groups.values()):
            a = np.array([np.mean(groups[(s, arm_a)][key]) for s in seeds])
            b = np.array([np.mean(groups[(s, arm_b)][key]) for s in seeds])
            out[key] = {"d": a - b, "seeds": seeds}
    return out


def paired_loc_auto(tag, arm_a, arm_b):
    path = os.path.join("results", f"campaign_{tag}", "eval_localization",
                        "MATE-4v8-9-v0_seq_ep5.json")
    if not os.path.exists(path):
        return None
    recs = json.load(open(path))
    groups = {(r["seed"], r["arm"]): r for r in recs}
    seeds = sorted({r["seed"] for r in recs})
    if not seeds:
        return None
    arms = {r["arm"] for r in recs}
    a_name = arm_a if arm_a in arms else None
    b_name = arm_b if arm_b in arms else None
    if a_name is None or b_name is None:
        return None
    a = np.array([groups[(s, a_name)]["seq_loc"] for s in seeds])
    b = np.array([groups[(s, b_name)]["seq_loc"] for s in seeds])
    return {"d": a - b, "seeds": seeds}


def fmt_cell(d):
    lo, hi, bias = bootstrap_ci(d, n_boot=20000, alpha=0.10)
    wins = int(np.sum(np.asarray(d) > 0))
    return d.mean(), wins, lo, hi


def main():
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Paired intrinsic-vs-baseline no-effect summary in 4v8 "
          "(4 seeds). $\\Delta$ is the paired mean difference; $90\\%$ CI is "
          "the seed bootstrap ($B{=}20{,}000$).}")
    print("\\label{tab:4v8noeffect}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Intrinsic & Metric & $\\Delta$ & Wins & $90\\%$ CI \\\\")
    print("\\midrule")
    first = True
    for tag, label in CAMPAIGNS:
        arm_a, arm_b, prefix = ARMS[tag]
        base = load_eval_final(tag, prefix, arm_a, arm_b)
        loc = paired_loc_auto(tag, arm_a, arm_b)
        rows = []
        if "cend_w" in base:
            d = base["cend_w"]["d"]
            m, w, lo, hi = fmt_cell(d)
            rows.append(("C$_{\\text{end}}$ (w)", m, w, lo, hi))
        if loc is not None:
            d = loc["d"]
            m, w, lo, hi = fmt_cell(d)
            rows.append(("seq\\_loc", m, w, lo, hi))
        for i, (metric, m, w, lo, hi) in enumerate(rows):
            lab = label if i == 0 else ""
            print(f"{lab} & {metric} & ${m:+.4f}$ & {w}/{len(d)} "
                  f"& $[{lo:+.4f},{hi:+.4f}]$ \\\\")
        if tag != CAMPAIGNS[-1][0]:
            print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")


if __name__ == "__main__":
    main()