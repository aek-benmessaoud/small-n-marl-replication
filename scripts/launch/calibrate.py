"""
scripts/calibrate.py — Phase 5a calibration run.

Answers, before the full campaign:
  1. How fast is the MATE step loop + MAPPO training on this machine?
  2. Does MAPPO actually learn anything on a short budget (eval return
     improving, finite losses)?
  3. Do the heuristic baselines produce sane U / returns?

Runs (all under the RunContext protocol):
  - env creation timing
  - one capped episode per baseline (random / greedy / memory / chao-aware)
  - MAPPO training for --steps with periodic deterministic eval, recording
    the eval-return / U_end / wall-time curve
  - same training WITHOUT the intrinsic reward (ablation), same budget

Usage:
    .venv\\Scripts\\python.exe scripts\\calibrate.py --regime MATE-4v8-9-v0 \
        --steps 15000 --horizon 1000 --eval-every 2000 --tag calib
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")

import mate  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_REGIME, MAX_STEPS, REWARD_LAMBDA,
)
from src.devops.runner import RunContext  # noqa: E402
from src.environment.mate_env import (  # noqa: E402
    make_mate, num_entities,
)
from src.environment.reward import ChaoUReward  # noqa: E402
from src.rl.heuristics import (  # noqa: E402
    ChaoUAwarePolicy, GreedyTrackingPolicy, RandomCameraPolicy,
    TeamMemoryTrackingPolicy,
)
from src.rl.mappo import MAPPO  # noqa: E402


def make_env(regime, seed, intrinsic=True):
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=seed)
    env.seed(seed)
    if intrinsic:
        env = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
    return env


def run_capped_episode(env, policy, cap):
    """One episode of `policy` against `env`, capped at `cap` steps.
    Returns (return, steps, U_end or None)."""
    obs = env.reset()
    done = False
    steps = 0
    ret = 0.0
    while not done and steps < cap:
        tracker = getattr(env, "tracker", None)
        obs, r, done, info = env.step(policy.act(obs, tracker=tracker))
        ret += float(r)
        steps += 1
    U_end = env.current_U() if hasattr(env, "current_U") else None
    return ret, steps, U_end


def train_calibrate(mappo, env, max_steps, episode_cap, horizon, eval_every,
                    eval_cap, logger):
    """Short MAPPO training with periodic deterministic eval. Returns a
    dict with curve lists and timing."""
    curve = {"step": [], "eval_return": [], "U_end": [], "wall_sec": []}
    t0 = time.perf_counter()
    training_steps = 0
    while training_steps < max_steps:
        buffer = mappo.collect_rollout(env, horizon, episode_cap)
        mappo.update(buffer)
        training_steps += len(buffer[0])
        if training_steps % eval_every < horizon:
            eret, elen = mappo.collect_episode(env, eval_cap, deterministic=True)
            U_end = env.current_U() if hasattr(env, "current_U") else None
            curve["step"].append(training_steps)
            curve["eval_return"].append(round(float(eret), 3))
            curve["U_end"].append(None if U_end is None else round(float(U_end), 3))
            curve["wall_sec"].append(round(time.perf_counter() - t0, 2))
            if logger is not None:
                print(f"[CALIB] step={training_steps} eval_return={eret:.3f} "
                      f"U_end={U_end} elapsed={time.perf_counter()-t0:.1f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    return {
        "wall_sec": round(wall, 2),
        "steps_per_sec": round(training_steps / wall, 3),
        "total_steps": int(training_steps),
        "curve": curve,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--episode-cap", type=int, default=MAX_STEPS)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-cap", type=int, default=MAX_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--timestamp", default=None)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    config = {
        "experiment": f"calibrate_{args.tag}",
        "phase": 5,
        "regime": args.regime,
        "seed": args.seed,
        "num_runs": 1,
        "reward_lambda": REWARD_LAMBDA,
        "chao_variant": "bias_cap",
        "max_steps": args.steps,
        "horizon": args.horizon,
        "episode_cap": args.episode_cap,
        "eval_every": args.eval_every,
        "eval_cap": args.eval_cap,
        "tag": args.tag,
    }

    with RunContext(experiment=f"calibrate_{args.tag}", phase=5,
                    config=config, timestamp=args.timestamp) as ctx:
        report = {"args": vars(args)}

        # ---- env creation timing ----
        t0 = time.perf_counter()
        env = make_env(args.regime, args.seed, intrinsic=True)
        n_cam, n_tar, n_obs = num_entities(env)
        report["env_creation_sec"] = round(time.perf_counter() - t0, 3)
        print(f"[CALIB] regime={args.regime} cameras={n_cam} targets={n_tar} "
              f"obstacles={n_obs}")

        # ---- baselines ----
        base_results = {}
        for name, factory in BASELINES.items():
            policy = factory(n_cam, n_tar, n_obs, args.seed)
            t0 = time.perf_counter()
            ret, steps, U_end = run_capped_episode(env, policy, args.eval_cap)
            wall = time.perf_counter() - t0
            base_results[name] = {
                "return": round(float(ret), 3),
                "steps": steps,
                "U_end": None if U_end is None else round(float(U_end), 3),
                "wall_sec": round(wall, 3),
                "steps_per_sec": round(steps / wall, 3),
            }
            print(f"[CALIB] baseline {name}: ret={ret:.1f} steps={steps} "
                  f"U_end={U_end} {steps/wall:.1f} steps/s")
        report["baselines"] = base_results
        env.close()

        # ---- MAPPO with intrinsic ----
        env = make_env(args.regime, args.seed, intrinsic=True)
        obs_dim = env.reset().shape[1]
        mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                      seed=args.seed * 1000 + 1)
        mappo.fit_obs_normalizer(env, seed=args.seed)
        print(f"[CALIB] training MAPPO (with intrinsic) budget={args.steps}")
        report["mappo_with_intrinsic"] = train_calibrate(
            mappo, env, args.steps, args.episode_cap, args.horizon,
            args.eval_every, args.eval_cap, ctx.tee)
        env.close()

        # ---- MAPPO without intrinsic (ablation) ----
        if not args.skip_ablation:
            env = make_env(args.regime, args.seed, intrinsic=False)
            mappo2 = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                           seed=args.seed * 1000 + 1)
            mappo2.fit_obs_normalizer(env, seed=args.seed)
            print(f"[CALIB] training MAPPO (no intrinsic) budget={args.steps}")
            report["mappo_no_intrinsic"] = train_calibrate(
                mappo2, env, args.steps, args.episode_cap, args.horizon,
                args.eval_every, args.eval_cap, ctx.tee)
            env.close()

        # ---- persist + summary ----
        with open(os.path.join(ctx.runner.out_dir, "calibration.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        print("[CALIB-SUMMARY]")
        for name, r in base_results.items():
            print(f"  {name:<16} {r['steps_per_sec']:>7.2f} steps/s  "
                  f"ret={r['return']:>10.2f}  U_end={r['U_end']}")
        for key in ("mappo_with_intrinsic", "mappo_no_intrinsic"):
            if key in report:
                r = report[key]
                c = r["curve"]
                print(f"  {key:<16} {r['steps_per_sec']:>7.2f} steps/s  "
                      f"eval_return first={c['eval_return'][0] if c['eval_return'] else '-'}  "
                      f"last={c['eval_return'][-1] if c['eval_return'] else '-'}")
        print("[CALIB-OK]")


if __name__ == "__main__":
    main()
