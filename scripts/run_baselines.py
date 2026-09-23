"""
scripts/run_baselines.py — campaign heuristic baselines for ONE seed.

Evaluates the four heuristic policies (random / greedy_tracking /
memory_tracking / chao_u_aware) on a fresh plain-MATE env (UTracker for U) with
the same seed protocol as the MAPPO arms. Each baseline is evaluated for
--eval-episodes deterministic episodes (capped at --eval-cap); returns and
U_end are recorded per episode.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_baselines.py --seed 0 --tag campaign
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project09.config import (  # noqa: E402
    BASE_SEED, DEFAULT_REGIME, MAX_STEPS, REWARD_LAMBDA, SEED_STRIDE,
)
from project09.devops.runner import RunContext  # noqa: E402
from project09.environment.mate_env import make_mate, num_entities  # noqa: E402
from project09.environment.reward import UTracker  # noqa: E402
from project09.rl.heuristics import (  # noqa: E402
    ChaoUAwarePolicy, GreedyTrackingPolicy, RandomCameraPolicy,
    TeamMemoryTrackingPolicy,
)

BASELINES = {
    "random": lambda nc, nt, no, s: RandomCameraPolicy(
        num_cameras=nc, num_targets=nt, num_obstacles=no, seed=s),
    "greedy_tracking": lambda nc, nt, no, s: GreedyTrackingPolicy(
        num_cameras=nc, num_targets=nt, num_obstacles=no),
    "memory_tracking": lambda nc, nt, no, s: TeamMemoryTrackingPolicy(
        num_cameras=nc, num_targets=nt, num_obstacles=no),
    "chao_u_aware": lambda nc, nt, no, s: ChaoUAwarePolicy(
        num_cameras=nc, num_targets=nt, num_obstacles=no, seed=s),
}


def run_episode(env, policy, cap):
    obs = env.reset()
    done = False
    steps = 0
    ret = 0.0
    while not done and steps < cap:
        obs, r, done, info = env.step(policy.act(obs, tracker=env.tracker))
        ret += float(r)
        steps += 1
    U_end = env.current_U() if hasattr(env, "current_U") else None
    return float(ret), steps, (None if U_end is None else float(U_end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="run index 0..29")
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-cap", type=int, default=MAX_STEPS)
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    env_seed = BASE_SEED + args.seed * SEED_STRIDE
    probe = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                      seed=env_seed)
    probe.seed(env_seed)
    n_cam, n_tar, n_obs = num_entities(probe)
    probe.close()

    config = {
        "experiment": f"campaign_{args.tag}_s{args.seed:02d}_baselines",
        "phase": 6,
        "regime": args.regime,
        "seed": env_seed,
        "run_index": args.seed,
        "num_runs": 1,
        "reward_lambda": REWARD_LAMBDA,
        "chao_variant": "bias_cap",
        "max_steps": args.eval_cap,
        "eval_episodes": args.eval_episodes,
    }
    with RunContext(experiment=config["experiment"], phase=6, config=config,
                    timestamp=args.timestamp) as ctx:
        results = {}
        t0 = time.perf_counter()
        for name, factory in BASELINES.items():
            policy = factory(n_cam, n_tar, n_obs, args.seed)
            env = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                            seed=env_seed)
            env.seed(env_seed)
            env = UTracker(env)
            rets, uends = [], []
            for _ in range(args.eval_episodes):
                ret, steps, U_end = run_episode(env, policy, args.eval_cap)
                rets.append(round(ret, 3))
                uends.append(None if U_end is None else round(U_end, 3))
            env.close()
            mean_u = (float(np.mean([u for u in uends if u is not None]))
                      if any(u is not None for u in uends) else None)
            results[name] = {
                "returns": rets,
                "U_ends": uends,
                "mean_return": round(float(np.mean(rets)), 3),
                "mean_U_end": mean_u,
                "wall_sec": round(time.perf_counter() - t0, 2),
            }
            print(f"[CAMPAIGN] baseline {name}: mean_ret={results[name]['mean_return']} "
                  f"mean_U_end={mean_u}", flush=True)

        report = {"env_seed": env_seed, "results": results}
        with open(os.path.join(ctx.runner.out_dir, "baselines.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
    print(f"[CAMPAIGN-BASELINES-OK] seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
