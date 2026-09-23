"""
scripts/run_mappo.py — campaign run for ONE seed (paired arms).

For run index r in 0..29 (env_seed = BASE_SEED + r * SEED_STRIDE) trains two
MAPPO agents under identical conditions (same env seed, same policy seed, same
hyper-parameters), each in its own RunContext directory:

  * arm 'intrinsic'    : training reward = MATE + lambda*(U_t - U_{t+1})/U_max
  * arm 'no_intrinsic' : training reward = plain MATE reward (tracker only)

Both arms are evaluated on a FRESH plain-MATE env (same seed, wrapped with
UTracker) with a deterministic policy, so coverage return and U metrics are
directly comparable (the intrinsic arm is NOT rewarded during eval).

Eval schedule: intermediate points use 1 episode (curve), the final point uses
--eval-episodes episodes (test metric: mean return, mean U_end).

Resume-anywhere: latest.pt checkpoint is saved after every update; on start a
checkpoint in the run dir is loaded and training continues. Pass --timestamp to
re-enter a previously created (incomplete) run directory.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_mappo.py --seed 0 --tag campaign
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project09.config import (  # noqa: E402
    BASE_SEED, DEFAULT_REGIME, MAPPO_CLIP, MAPPO_EPOCHS, MAPPO_GAMMA,
    MAPPO_HIDDEN, MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCH, MAX_STEPS,
    REWARD_LAMBDA, SEED_STRIDE,
)
from project09.devops.runner import RunContext  # noqa: E402
from project09.environment.mate_env import make_mate, num_entities  # noqa: E402
from project09.environment.reward import (  # noqa: E402
    ChaoUReward, ClipEnvReward, LocalizationReward, RNDIntrinsic, UTracker,
    default_intr_clip, measure_loc_scale, measure_reward_scale,
    measure_rnd_scale, suggest_lambda, suggest_loc_lambda, suggest_rnd_lambda,
)
from project09.rl.mappo import MAPPO  # noqa: E402


def make_train_env(regime, seed, arm, reward_mode="chao",
                   reward_lambda=REWARD_LAMBDA, dense_window=None,
                   r_intr_clip=None, env_reward_clip=None, reward_window=None,
                   num_bins=None):
    """Training env for one arm. Arm determines the reward wrapper:
      'intrinsic' -> Chao-U shaping; 'loc' -> explicit localization objective
      (LocalizationReward); 'rnd' -> Random Network Distillation novelty over
      camera poses (RNDIntrinsic); 'no_intrinsic' / 'warm_intrinsic' -> plain
      MATE reward with the UTracker diagnostic wrapper."""
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=seed)
    env.seed(seed)
    if env_reward_clip is not None:
        env = ClipEnvReward(env, low=-env_reward_clip, high=env_reward_clip)
    if arm == "intrinsic":
        env = ChaoUReward(env, reward_lambda=reward_lambda,
                          dense=(reward_mode == "dense"),
                          dense_window=dense_window,
                          r_intr_clip=r_intr_clip,
                          window=reward_window, n_bins=num_bins)
    elif arm == "loc":
        env = LocalizationReward(env, reward_lambda=reward_lambda)
    elif arm == "rnd":
        env = RNDIntrinsic(env, reward_lambda=reward_lambda,
                           seed=seed * 1000 + 1)
    else:
        env = UTracker(env, n_bins=num_bins)
    return env


def make_eval_env(regime, seed, reward_window=None, num_bins=None):
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=seed)
    env.seed(seed)
    return UTracker(env, window=reward_window, n_bins=num_bins)


def eval_policy(mappo, env, n_episodes, cap):
    """Deterministic evaluation. Returns (returns, U_ends, C_ends, lengths)."""
    rets, uends, cends, lens = [], [], [], []
    for _ in range(n_episodes):
        ret, length = mappo.collect_episode(env, cap, deterministic=True)
        rets.append(float(ret))
        uends.append(float(env.current_U()) if hasattr(env, "current_U") else None)
        cends.append(float(env.current_C()) if hasattr(env, "current_C") else None)
        lens.append(int(length))
    return rets, uends, cends, lens


def train_arm(args, env_seed, arm, n_cam, obs_dim, ts, reward_lambda):
    """RunContext + MAPPO training for one arm. Returns the report dict."""
    config = {
        "experiment": f"campaign_{args.tag}_s{args.seed:02d}_{arm}",
        "phase": 6,
        "regime": args.regime,
        "seed": env_seed,
        "run_index": args.seed,
        "num_runs": 1,
        "reward_lambda": reward_lambda,
        "reward_mode": args.reward,
        "reward_window": args.reward_window,
        "num_bins": args.num_bins,
        "chao_variant": "bias_cap",
        "max_steps": args.steps,
        "arm": arm,
        "init_checkpoint": args.init_checkpoint,
        "frame_skip": args.frame_skip,
        "eval_episodes": args.eval_episodes,
        "policy_seed": env_seed * 1000 + 1,
    }
    with RunContext(experiment=config["experiment"], phase=6, config=config,
                    timestamp=ts) as ctx:
        env = make_train_env(args.regime, env_seed, arm,
                             reward_mode=args.reward, reward_lambda=reward_lambda,
                             dense_window=args.dense_window,
                             r_intr_clip=args.r_intr_clip,
                             env_reward_clip=args.env_reward_clip,
                             reward_window=args.reward_window,
                             num_bins=args.num_bins)
        eval_env = make_eval_env(args.regime, env_seed,
                                 reward_window=args.reward_window,
                                 num_bins=args.num_bins)
        mappo = MAPPO(
            obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
            seed=env_seed * 1000 + 1,
            hidden=MAPPO_HIDDEN, lr=args.lr, gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA,
            clip=MAPPO_CLIP, epochs=MAPPO_EPOCHS, minibatch=MAPPO_MINIBATCH,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            frame_skip=args.frame_skip,
        )

        ckpt = os.path.join(ctx.runner.out_dir, "latest.pt")
        if os.path.exists(ckpt):
            mappo.load(ckpt)
            print(f"[CAMPAIGN] {arm}: resumed at step {mappo.step_count}")
        elif args.init_checkpoint and os.path.exists(args.init_checkpoint):
            mappo.load(args.init_checkpoint)
            # Warm-start: the init checkpoint belongs to a previous campaign
            # phase; step_count/episode_count are re-based so --steps counts
            # the PHASE-2 steps only (eval schedule restarts at --eval-every).
            mappo.step_count = 0
            mappo.episode_count = 0
            print(f"[CAMPAIGN] {arm}: warm-started from {args.init_checkpoint} "
                  f"(re-based to step 0)")
        elif args.init_checkpoint and not os.path.exists(args.init_checkpoint):
            raise SystemExit(f"[CAMPAIGN] {arm}: --init-checkpoint missing: "
                             f"{args.init_checkpoint}")
        if not mappo.obs_normalizer.fitted:
            mappo.fit_obs_normalizer(env, seed=args.seed)

        curve = {"step": [], "eval_return": [], "eval_return_std": [],
                 "U_end": [], "C_end": [], "wall_sec": []}
        t0 = time.perf_counter()
        next_eval = ((mappo.step_count // args.eval_every) + 1) * args.eval_every
        final_metrics = None
        last_eval_sc = -1
        while mappo.step_count < args.steps:
            buffer = mappo.collect_rollout(env, args.horizon, args.episode_cap)
            mappo.update(buffer)
            mappo.save(ckpt, tag="latest")

            sc = mappo.step_count
            if sc >= next_eval:
                n_ep = args.eval_episodes if sc >= args.steps else 1
                rets, uends, cends, _ = eval_policy(mappo, eval_env, n_ep, args.eval_cap)
                mean_r = float(np.mean(rets))
                std_r = float(np.std(rets)) if len(rets) > 1 else 0.0
                mean_u = (float(np.mean([u for u in uends if u is not None]))
                          if any(u is not None for u in uends) else None)
                mean_c = (float(np.mean([c for c in cends if c is not None]))
                          if any(c is not None for c in cends) else None)
                curve["step"].append(sc)
                curve["eval_return"].append(round(mean_r, 3))
                curve["eval_return_std"].append(round(std_r, 3))
                curve["U_end"].append(None if mean_u is None else round(mean_u, 3))
                curve["C_end"].append(None if mean_c is None else round(mean_c, 3))
                curve["wall_sec"].append(round(time.perf_counter() - t0, 2))
                print(f"[CAMPAIGN] {arm} step={sc} eval_return={mean_r:.3f}+-{std_r:.3f} "
                      f"U_end={mean_u} C_end={mean_c} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
                last_eval_sc = sc
                if sc >= args.steps:
                    final_metrics = {
                        "returns": [round(float(r), 3) for r in rets],
                        "U_ends": [None if u is None else round(float(u), 3) for u in uends],
                        "C_ends": [None if c is None else round(float(c), 3) for c in cends],
                        "mean_return": round(mean_r, 3),
                        "mean_U_end": mean_u,
                        "mean_C_end": mean_c,
                    }
                next_eval += args.eval_every

        if last_eval_sc < args.steps:
            rets, uends, cends, _ = eval_policy(mappo, eval_env, args.eval_episodes,
                                                args.eval_cap)
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets)) if len(rets) > 1 else 0.0
            mean_u = (float(np.mean([u for u in uends if u is not None]))
                      if any(u is not None for u in uends) else None)
            mean_c = (float(np.mean([c for c in cends if c is not None]))
                      if any(c is not None for c in cends) else None)
            curve["step"].append(int(mappo.step_count))
            curve["eval_return"].append(round(mean_r, 3))
            curve["eval_return_std"].append(round(std_r, 3))
            curve["U_end"].append(None if mean_u is None else round(mean_u, 3))
            curve["C_end"].append(None if mean_c is None else round(mean_c, 3))
            curve["wall_sec"].append(round(time.perf_counter() - t0, 2))
            print(f"[CAMPAIGN] {arm} step={mappo.step_count} eval_return={mean_r:.3f}+-{std_r:.3f} "
                  f"U_end={mean_u} C_end={mean_c} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
            final_metrics = {
                "returns": [round(float(r), 3) for r in rets],
                "U_ends": [None if u is None else round(float(u), 3) for u in uends],
                "C_ends": [None if c is None else round(float(c), 3) for c in cends],
                "mean_return": round(mean_r, 3),
                "mean_U_end": mean_u,
                "mean_C_end": mean_c,
            }

        wall = time.perf_counter() - t0
        report = {
            "arm": arm,
            "env_seed": env_seed,
            "total_steps": int(mappo.step_count),
            "wall_sec": round(wall, 2),
            "steps_per_sec": round(mappo.step_count / wall, 3),
            "curve": curve,
            "final_eval": final_metrics,
        }
        with open(os.path.join(ctx.runner.out_dir, "results.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        env.close()
        eval_env.close()
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="run index 0..29")
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--episode-cap", type=int, default=MAX_STEPS)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-cap", type=int, default=MAX_STEPS)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--reward", choices=["chao", "dense", "loc", "rnd"], default="chao",
                    help="reward objective: 'chao' = discrete U shaping, "
                         "'dense' = continuous angular-diversity confidence, "
                         "'loc' = EXPLICIT instantaneous CRLB localization "
                         "term added to the env reward (changes the objective, "
                         "not a potential-based shaping), "
                         "'rnd' = Random Network Distillation novelty over "
                         "camera poses (non-stationary potential, targets "
                         "coverage directly)")
    ap.add_argument("--reward-lambda", default="auto",
                    help="lambda for the intrinsic term, or 'auto' to calibrate "
                         "it so p75|r_intr| ~= target_ratio * env-reward STD")
    ap.add_argument("--lambda-target-ratio", type=float, default=1.0,
                    help="target p75|r_intr| / env-reward-noise (per-step STD) "
                         "when --reward-lambda auto")
    ap.add_argument("--dense-window", type=int, default=None,
                    help="only the last N bearings per target feed the dense "
                         "confidence (None = full buffer)")
    ap.add_argument("--reward-window", type=int, default=None,
                    help="sliding-window richness: the tracker only keeps "
                         "bearings recorded in the last N steps, so U reflects "
                         "SHORT-TIMESCALE angular diversity (compatible with "
                         "moving targets). None = episode-cumulative (default)")
    ap.add_argument("--num-bins", type=int, default=None,
                    help="bin-sensitivity knob (roadmap step 6): discretize the "
                         "full circle in this many equal angular bins and count "
                         "OCCUPIED BINS per target as its independent "
                         "configurations, replacing the greedy 15-deg "
                         "clustering. 8/12/16/24 bins = 45/30/22.5/15 deg/bin. "
                         "None = greedy clustering (default)")
    ap.add_argument("--r-intr-clip", type=float, default=None,
                    help="clip |intrinsic reward| to this value per step")
    ap.add_argument("--env-reward-clip", type=float, default=None,
                    help="clip the env reward used for LEARNING to +/- this "
                         "value (removes exogenous delivery spikes; eval uses "
                         "the un-clipped reward)")
    ap.add_argument("--entropy-coef", type=float, default=0.0,
                    help="PPO entropy bonus coefficient")
    ap.add_argument("--lr", type=float, default=MAPPO_LR)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--timestamp", default=None,
                    help="fixed timestamp to re-enter an existing run dir")
    ap.add_argument("--skip-intrinsic", action="store_true")
    ap.add_argument("--skip-no-intrinsic", action="store_true")
    ap.add_argument("--arm", choices=["intrinsic", "no_intrinsic",
                                      "warm_intrinsic", "loc", "rnd"],
                    default=None,
                    help="run a SINGLE named arm. 'intrinsic' = Chao-U shaped; "
                         "'loc' = explicit localization objective; "
                         "'rnd' = RND novelty; 'warm_intrinsic' and "
                         "'no_intrinsic' train on the pure reward, differing "
                         "only in --init-checkpoint")
    ap.add_argument("--init-checkpoint", default=None,
                    help="start from this latest.pt (weights + optimizer) and "
                         "re-base step_count to 0, then train --steps more. "
                         "Used for warm-start / continuation controls.")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

    env_seed = BASE_SEED + args.seed * SEED_STRIDE
    probe = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                      seed=env_seed)
    probe.seed(env_seed)
    n_cam, n_tar, n_obs = num_entities(probe)
    obs_dim = probe.reset().shape[1]
    probe.close()

    # ---- calibrate reward_lambda ('auto') --------------------------------
    if args.reward_lambda == "auto":
        cal_env = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                            seed=env_seed)
        cal_env.seed(env_seed)
        # Calibrate on the SAME env the learner sees (ClipEnvReward included):
        # env_std must be the noise of the LEARNING signal, not the raw
        # un-clipped reward with its exogenous delivery spikes (which would
        # over-scale lambda by 10-50x).
        if args.env_reward_clip is not None:
            cal_env = ClipEnvReward(cal_env, low=-args.env_reward_clip,
                                    high=args.env_reward_clip)
        if args.reward == "loc":
            scale = measure_loc_scale(cal_env, n_episodes=2, n_steps=500,
                                      seed=args.seed)
            reward_lambda = suggest_loc_lambda(scale,
                                               target_ratio=args.lambda_target_ratio)
            print(f"[CAMPAIGN] seed={args.seed} reward={args.reward} auto-lambda: "
                  f"env_std={scale['env_std']:.3f} loc mean/p75/nz_p75="
                  f"{scale['loc_mean']:.4f}/{scale['loc_p75']:.4f}/"
                  f"{scale['loc_nonzero_p75']:.4f} (fires n="
                  f"{scale['loc_nonzero_n']}) -> lambda={reward_lambda:.3f}",
                  flush=True)
        elif args.reward == "rnd":
            scale = measure_rnd_scale(cal_env, n_episodes=2, n_steps=500,
                                      seed=args.seed)
            reward_lambda = suggest_rnd_lambda(scale,
                                               target_ratio=args.lambda_target_ratio)
            print(f"[CAMPAIGN] seed={args.seed} reward={args.reward} auto-lambda: "
                  f"env_std={scale['env_std']:.3f} rnd mean/p75/nz_p75="
                  f"{scale['rnd_mean']:.4f}/{scale['rnd_p75']:.4f}/"
                  f"{scale['rnd_nonzero_p75']:.4f} -> lambda={reward_lambda:.4f}",
                  flush=True)
        else:
            cal_env = ChaoUReward(cal_env, reward_lambda=1.0,
                                  dense=(args.reward == "dense"),
                                  dense_window=args.dense_window,
                                  window=args.reward_window)
            scale = measure_reward_scale(cal_env, n_episodes=2, n_steps=500,
                                         seed=args.seed)
            cal_env.close()
            reward_lambda = suggest_lambda(scale, target_ratio=args.lambda_target_ratio)
            if args.r_intr_clip is None and args.reward == "dense":
                args.r_intr_clip = default_intr_clip(
                    scale, target_ratio=args.lambda_target_ratio)
            print(f"[CAMPAIGN] seed={args.seed} reward={args.reward} auto-lambda: "
                  f"|r_env|={scale['env_abs_mean']:.3f} unit p50/p75/p95="
                  f"{scale['unit_p50']:.5f}/{scale['unit_p75']:.5f}/{scale['unit_p95']:.5f} "
                  f"-> lambda={reward_lambda:.1f} r_intr_clip="
                  f"{args.r_intr_clip if args.r_intr_clip is not None else 'off'}", flush=True)
    else:
        reward_lambda = float(args.reward_lambda)

    if args.reward == "loc":
        treat, ctrl = "loc", "no_intrinsic"
    elif args.reward == "rnd":
        treat, ctrl = "rnd", "no_intrinsic"
    else:
        treat, ctrl = "intrinsic", "no_intrinsic"
    arms = []
    if args.arm is not None:
        arms = [args.arm]
    else:
        if not args.skip_intrinsic:
            arms.append(treat)
        if not args.skip_no_intrinsic:
            arms.append(ctrl)

    print(f"[CAMPAIGN] seed={args.seed} env_seed={env_seed} regime={args.regime} "
          f"cameras={n_cam} targets={n_tar} arms={arms}", flush=True)
    for arm in arms:
        train_arm(args, env_seed, arm, n_cam, obs_dim, args.timestamp,
                  reward_lambda)
    print(f"[CAMPAIGN-OK] seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
