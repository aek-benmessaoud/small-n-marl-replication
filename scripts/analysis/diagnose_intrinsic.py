"""
scripts/diagnose_intrinsic.py — why did the intrinsic signal not help?

Replays deterministic episodes with the trained campaign checkpoints (and a
random policy) and measures the intrinsic-signal statistics:

  * how often does U decrease during an episode (r_intrinsic > 0)?
  * how large is r_intrinsic vs the env reward (signal-to-noise)?
  * U trajectory (start / min / end) — the floor at 1.0 masks localization,
    so we report the full trajectory.

Usage:
    .venv\\Scripts\\python.exe scripts\\diagnose_intrinsic.py --tag campaign --ts 20260808_000125 --seeds 0..7
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (  # noqa: E402
    BASE_SEED, DEFAULT_REGIME, MAPPO_CLIP, MAPPO_EPOCHS, MAPPO_GAMMA,
    MAPPO_HIDDEN, MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCH, MAX_STEPS,
    REWARD_LAMBDA, SEED_STRIDE,
)
from src.environment.mate_env import make_mate  # noqa: E402
from src.environment.reward import ChaoUReward, UTracker  # noqa: E402
from src.rl.mappo import MAPPO  # noqa: E402


def load_checkpoint(tag, ts, seed, arm):
    pat = os.path.join("results", f"campaign_{tag}_s{seed:02d}_{arm}", ts, "latest.pt")
    return pat if os.path.exists(pat) else None


def run_episode_stats(env, act_fn, cap=MAX_STEPS):
    """One episode; act_fn(obs)->action. Returns per-step intrinsic/env rewards
    and the U trajectory (U after each step)."""
    obs = env.reset()
    U_traj = [float(env.current_U())]
    intr, envr = [], []
    done = False
    steps = 0
    while not done and steps < cap:
        obs, r, done, info = env.step(act_fn(obs))
        r_intr = info[0].get("intrinsic_reward", 0.0) if info else 0.0
        intr.append(float(r_intr))
        envr.append(float(r) - float(r_intr))
        U_traj.append(float(env.current_U()))
        steps += 1
    return np.asarray(intr), np.asarray(envr), np.asarray(U_traj)


def summarize(name, intr, envr, U):
    n = len(intr)
    if n == 0:
        return None
    dec = int((intr > 1e-9).sum())
    inc = int((intr < -1e-9).sum())
    print(f"  {name:<22} n={n:4d} | U-decrease steps={dec:3d} ({100*dec/max(n,1):4.1f}%) "
          f"U-increase={inc} stable={n-dec-inc}")
    print(f"    r_intr: mean={intr.mean():+.4f} |mean|={np.abs(intr).mean():.4f} "
          f"max={intr.max():+.4f} | r_env: mean={envr.mean():+.3f} |mean|={np.abs(envr).mean():.3f}")
    print(f"    U: start={U[0]:.2f} min={U.min():.2f} end={U[-1]:.2f} "
          f"drops_total={int(np.sum(np.diff(U) < -1e-9))}")
    return {
        "n": n, "u_decrease": dec, "u_increase": inc,
        "r_intr_mean": float(intr.mean()), "r_intr_abs_mean": float(np.abs(intr).mean()),
        "r_env_mean": float(envr.mean()), "r_env_abs_mean": float(np.abs(envr).mean()),
        "U_start": float(U[0]), "U_min": float(U.min()), "U_end": float(U[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--ts", default="20260808_000125")
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--seeds", default="0..7")
    ap.add_argument("--cap", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    lo, hi = [int(x) for x in args.seeds.split("..")]
    seeds = list(range(lo, hi + 1))
    agg = {}

    for seed in seeds:
        env_seed = BASE_SEED + seed * SEED_STRIDE
        print(f"--- seed {seed} (env_seed {env_seed}) ---")

        # random policy on intrinsic env
        env = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                        seed=env_seed)
        env.seed(env_seed)
        env = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
        rng = np.random.default_rng(seed)
        intr, envr, U = run_episode_stats(
            env, lambda obs, rng=rng: rng.uniform(-1, 1, size=(obs.shape[0], 2)),
            cap=args.cap)
        env.close()
        s = summarize("random", intr, envr, U)
        agg.setdefault("random", []).append(s)

        for arm, wrapper in [("intrinsic", "ChaoUReward"),
                             ("no_intrinsic", "UTracker")]:
            ck = load_checkpoint(args.tag, args.ts, seed, arm)
            if not ck:
                print(f"  [{arm}] checkpoint missing: {ck}")
                continue
            env = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                            seed=env_seed)
            env.seed(env_seed)
            env = (ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
                   if wrapper == "ChaoUReward" else UTracker(env))
            obs0 = env.reset()
            obs_dim = obs0.shape[1]
            n_cam = obs0.shape[0]
            mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                          seed=env_seed * 1000 + 1, hidden=MAPPO_HIDDEN,
                          lr=MAPPO_LR, gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA,
                          clip=MAPPO_CLIP, epochs=MAPPO_EPOCHS,
                          minibatch=MAPPO_MINIBATCH, frame_skip=1)
            mappo.load(ck)
            # restore the env to the fresh reset (we consumed one reset above)
            intr, envr, U = run_episode_stats(
                env, lambda obs: mappo.act_batch(obs, deterministic=True)[0],
                cap=args.cap)
            env.close()
            s = summarize(f"trained[{arm}]", intr, envr, U)
            agg.setdefault(f"trained_{arm}", []).append(s)

    print("\n=== AGGREGATE (mean over seeds) ===")
    for name, lst in agg.items():
        lst = [x for x in lst if x]
        if not lst:
            continue
        n = len(lst)
        print(f"{name:<22} seeds={n} | "
              f"U-decrease {np.mean([x['u_decrease'] for x in lst]):.1f} steps/ep "
              f"({100*np.mean([x['u_decrease']/max(x['n'],1) for x in lst]):.2f}%) | "
              f"|r_intr|={np.mean([x['r_intr_abs_mean'] for x in lst]):.4f} "
              f"|r_env|={np.mean([x['r_env_abs_mean'] for x in lst]):.3f} | "
              f"U_start={np.mean([x['U_start'] for x in lst]):.2f} "
              f"U_min={np.mean([x['U_min'] for x in lst]):.2f} "
              f"U_end={np.mean([x['U_end'] for x in lst]):.2f}")


if __name__ == "__main__":
    main()
