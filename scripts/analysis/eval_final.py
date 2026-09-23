"""
scripts/eval_final.py — Final per-checkpoint evaluation of a campaign.

For each (seed, arm) checkpoint: n deterministic episodes on a fresh,
UN-clipped MATE env wrapped in UTracker(window=W) so the reported U/C match the
campaign's reward semantics. Per episode it records:
  * rets   : true (un-clipped) env return,
  * seens  : mean number of targets seen by >=1 camera per step,
  * uend_w : windowed U at episode end (work still to do within the window),
  * cend_w : windowed circular-variance confidence at episode end,
  * cend_c : episode-CUMULATIVE circular-variance confidence (cross-campaign
             comparable to the pre-window reports).
Persists one JSON per (seed, arm) under results/campaign_<tag>/eval_final/ and
prints the paired aggregation (Wilcoxon signed-rank over seeds).

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_final.py --tag chao4v8win \\
      --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 --window 20
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import (  # noqa: E402
    BASE_SEED, DEFAULT_REGIME, MAPPO_CLIP, MAPPO_EPOCHS, MAPPO_GAMMA,
    MAPPO_HIDDEN, MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCH, MAX_STEPS,
    SEED_STRIDE,
)
from src.environment.mate_env import make_mate, num_entities, state_arrays  # noqa: E402
from src.environment.reward import UTracker  # noqa: E402
from src.estimators.continuous import TargetTracker  # noqa: E402
from src.rl.mappo import MAPPO  # noqa: E402


def _load(tag, arm, seed, regime):
    env_seed = BASE_SEED + seed * SEED_STRIDE
    fs = sorted(glob.glob(f"results/campaign_{tag}_s{seed:02d}_{arm}/*/latest.pt"))
    if not fs:
        return None
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=env_seed)
    env.seed(env_seed)
    n_cam, _, _ = num_entities(env)
    obs_dim = env.reset().shape[1]
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                  seed=env_seed * 1000 + 1, hidden=MAPPO_HIDDEN, lr=MAPPO_LR,
                  gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA, clip=MAPPO_CLIP,
                  epochs=MAPPO_EPOCHS, minibatch=MAPPO_MINIBATCH,
                  entropy_coef=0.0, frame_skip=1)
    mappo.load(fs[0])
    return env, mappo


def eval_arm(tag, arm, seed, regime, n_ep, cap, window):
    loaded = _load(tag, arm, seed, regime)
    if loaded is None:
        return None
    env, mappo = loaded
    wrapped = UTracker(env, window=window)
    cumulative = TargetTracker(wrapped.num_targets)
    rets, seens, uends, cends_w, cends_c = [], [], [], [], []
    for _ in range(n_ep):
        obs = wrapped.reset()
        cumulative.reset()
        cam, tar, view = state_arrays(wrapped.env)
        cumulative.observe(cam, tar, view)
        done = False
        st = 0
        ret, seen_sum = 0.0, 0.0
        while not done and st < cap:
            a, _ = mappo.act_batch(obs, deterministic=True)
            obs, r, done, _ = wrapped.step(a)
            cam, tar, view = state_arrays(wrapped.env)
            cumulative.observe(cam, tar, view)
            ret += float(r)
            seen_sum += float(view.any(axis=0).sum())
            st += 1
        rets.append(round(ret, 2))
        seens.append(round(seen_sum / max(st, 1), 3))
        uends.append(float(wrapped.current_U()))
        cends_w.append(float(wrapped.current_C()))
        cends_c.append(float(cumulative.variance_confidence_sum()))
    env.close()
    return {"tag": tag, "arm": arm, "seed": seed, "regime": regime,
            "rets": rets, "seens": seens,
            "uend_w": uends, "cend_w": cends_w, "cend_c": cends_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--seeds", default="0..3")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--cap", type=int, default=MAX_STEPS)
    ap.add_argument("--window", type=int, default=20,
                    help="sliding-window width of the UTracker used for U/C "
                         "(match the campaign's --reward-window)")
    ap.add_argument("--arms", default="intrinsic,no_intrinsic",
                    help="comma list of the two arm names, in order "
                         "treatment,control (saved as *_intr.json / *_noi.json)")
    args = ap.parse_args()

    lo, hi = [int(x) for x in args.seeds.split("..")]
    seeds = list(range(lo, hi + 1))
    print(f"=== eval_final {args.tag} | {args.regime} | window={args.window} "
          f"| seeds={seeds} ep={args.episodes} ===", flush=True)

    out_dir = os.path.join("results", f"campaign_{args.tag}", "eval_final")
    os.makedirs(out_dir, exist_ok=True)
    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]
    rows = []
    for s in seeds:
        for arm, suf in ((arm_a, "intr"), (arm_b, "noi")):
            r = eval_arm(args.tag, arm, s, args.regime,
                         args.episodes, args.cap, args.window)
            if r is None:
                print(f"  missing ckpt seed={s} arm={arm}", flush=True)
                continue
            with open(os.path.join(out_dir,
                                   f"{args.tag}_s{s:02d}_"
                                   f"{suf}.json"),
                      "w") as f:
                json.dump(r, f, indent=2)
            print(f"  seed={s} {arm:<12} ret={np.mean(r['rets']):9.1f} "
                  f"seen={np.mean(r['seens']):.3f} "
                  f"U_end_w={np.mean(r['uend_w']):.3f} "
                  f"C_end_w={np.mean(r['cend_w']):.4f} "
                  f"C_end_cum={np.mean(r['cend_c']):.4f}", flush=True)
            rows.append(r)

    intr = [r for r in rows if r["arm"] == arm_a]
    noi = [r for r in rows if r["arm"] == arm_b]
    if not intr or not noi:
        print("incomplete data — aggregation skipped")
        return

    from scipy.stats import wilcoxon
    print(f"\n=== AGGREGATE ({arm_a} vs {arm_b}, paired, n={len(intr)}) ===")
    for name, key in [("vraie récompense", "rets"),
                      ("mean_seen", "seens"),
                      ("U_end fenêtré", "uend_w"),
                      ("C_end fenêtré", "cend_w"),
                      ("C_end cumulatif", "cend_c")]:
        a = np.array([np.mean(r[key]) for r in intr])
        b = np.array([np.mean(r[key]) for r in noi])
        wins = int(np.sum(a > b))
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        print(f"{name:<16} intr={a.mean():+.3f}  no_intr={b.mean():+.3f}  "
              f"delta={a.mean() - b.mean():+.3f}  wins={wins}/{len(a)}  "
              f"p={p:.3f}")


if __name__ == "__main__":
    main()
