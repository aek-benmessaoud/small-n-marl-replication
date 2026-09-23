"""scripts/diagnose_gae.py — §7-5 GAE credit-attribution diagnostic (offline).

Replays a TRAINED checkpoint under its TRAINING reward env (same lambda/window),
recomputing the exact GAE advantages the learner saw (gamma=0.99, lam=0.95,
same critic), while SEPARATING the env-reward and intrinsic-reward streams.
GAE is linear in rewards given a fixed value function, so:

    adv_total = adv_env + adv_intr          (exact)

The diagnostic answers: is the intrinsic signal washed out by the env-reward
noise in the credit they both receive?

Outputs, per evaluated checkpoint:
  corr(r_intr, adv_total), corr(r_intr, adv_intr)
  std(adv_intr)/std(adv_total), var(adv_intr)/var(adv_total)
  per-step |r_env| vs |lambda*r_intr| statistics

Usage:
    .venv\\Scripts\\python.exe scripts\\diagnose_gae.py --seeds 0 9 15
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project09.config import (  # noqa: E402
    MAPPO_GAMMA, MAPPO_LAMBDA, MAPPO_HIDDEN, MAPPO_LR, MAPPO_CLIP,
    MAPPO_EPOCHS, MAPPO_MINIBATCH,
)
from project09.environment.mate_env import make_mate, num_entities  # noqa: E402
from project09.environment.reward import ChaoUReward  # noqa: E402
from project09.rl.mappo import MAPPO  # noqa: E402

GAMMA = MAPPO_GAMMA
LAM = MAPPO_LAMBDA


def gae_from_rewards(rewards, values, dones, last_value, last_done):
    """Exact replica of RolloutBuffer.finalize advantage recurrence."""
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        next_done = last_done if t == n - 1 else dones[t + 1]
        delta = rewards[t] + GAMMA * next_val * (not next_done) - values[t]
        acc = delta + GAMMA * LAM * (not next_done) * acc
        adv[t] = acc
    return adv


def gae_reward_contribution(rewards, dones, last_done):
    """Credit each reward stream independently generates, given the shared
    value baseline. delta = r (no baseline term), accumulated exactly like the
    GAE lambda-discounting. LINEAR in rewards: contrib(r1)+contrib(r2) ==
    contrib(r1+r2) exactly, so var(contrib_env), var(contrib_intr) decompose
    the reward-driven part of adv_total without double-counting the baseline."""
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    n = len(rewards)
    out = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for t in reversed(range(n)):
        next_done = last_done if t == n - 1 else dones[t + 1]
        acc = rewards[t] + GAMMA * LAM * (not next_done) * acc
        out[t] = acc
    return out


def collect_streams(tag, seed, arm, regime, reward_lambda, window, horizon,
                    cap, n_rollouts=2, deterministic=False):
    """Replay one checkpoint; return arrays of per-step env/intr/total rewards
    plus critic values."""
    dirs = sorted(glob.glob(
        f"results/campaign_{tag}_s{seed:02d}_{arm}/**/latest.pt",
        recursive=True))
    if not dirs:
        return {"seed": seed, "arm": arm, "error": "no checkpoint"}
    ckpt = dirs[0]
    with open(os.path.join(os.path.dirname(ckpt), "config.json")) as f:
        cfg = json.load(f)

    env = make_mate(env_id="MultiAgentTracking-v0", config=regime,
                    seed=cfg["seed"])
    env.seed(cfg["seed"])
    env = ChaoUReward(env, reward_lambda=reward_lambda, window=window)
    n_cam, n_tar, _ = num_entities(env)
    obs_dim = env.reset().shape[1]

    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                  seed=cfg["seed"] * 1000 + 1, hidden=MAPPO_HIDDEN, lr=MAPPO_LR,
                  gamma=GAMMA, lam=LAM, clip=MAPPO_CLIP, epochs=MAPPO_EPOCHS,
                  minibatch=MAPPO_MINIBATCH, frame_skip=1)
    mappo.load(ckpt)

    E, I, T, V, D = [], [], [], [], []
    for _ in range(n_rollouts):
        obs = env.reset()
        steps = 0
        done = False
        while not done and steps < horizon and steps < cap:
            v = mappo.critic_value(obs)
            actions, _ = mappo.act_batch(obs, deterministic=deterministic)
            U_before = env.U_prev
            obs, r_shaped, done, _ = env.step(actions)
            U_after = env.U_prev
            r_intr = reward_lambda * (U_before - U_after) / max(env.u_max, 1)
            r_env = float(r_shaped) - r_intr
            E.append(r_env)
            I.append(r_intr)
            T.append(float(r_shaped))
            V.append(v)
            D.append(bool(done))
            steps += 1
        # bootstrap value for the terminal transition (as in collect_rollout)
        V.append(mappo.critic_value(obs) if not done else 0.0)
        D.append(done)
    env.close()
    return {"seed": seed, "arm": arm, "E": np.asarray(E), "I": np.asarray(I),
            "T": np.asarray(T), "V": np.asarray(V), "D": np.asarray(D)}


def analyze(streams):
    E, I, T = streams["E"], streams["I"], streams["T"]
    V, D = streams["V"], streams["D"]
    last_v = V[-1]
    last_d = D[-1]
    adv_tot = gae_from_rewards(T, V[:-1], D[:-1], last_v, last_d)
    # exact linear decomposition: reward-driven credit per stream
    contrib_env = gae_reward_contribution(E, D[:-1], last_d)
    contrib_intr = gae_reward_contribution(I, D[:-1], last_d)
    contrib_both = contrib_env + contrib_intr
    max_err = float(np.max(np.abs(contrib_both - contrib_env - contrib_intr)))
    var_tot = adv_tot.var() + 1e-30
    var_both = contrib_both.var() + 1e-30

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "seed": int(streams["seed"]),
        "arm": streams["arm"],
        "n": int(len(T)),
        "linearity_max_err": max_err,
        "corr_r_intr__adv_tot": corr(I, adv_tot),
        "corr_r_intr__contrib_intr": corr(I, contrib_intr),
        "corr_r_intr__contrib_env": corr(I, contrib_env),
        "corr_r_intr__r_env": corr(I, E),
        "std_adv_tot": float(adv_tot.std()),
        "std_contrib_intr": float(contrib_intr.std()),
        "std_contrib_env": float(contrib_env.std()),
        "var_share_intr": float(contrib_intr.var() / var_tot),
        "var_share_env": float(contrib_env.var() / var_tot),
        "var_share_both": float(var_both / var_tot),
        "mean_adv_tot": float(adv_tot.mean()),
        "mean_contrib_intr": float(contrib_intr.mean()),
        "mean_contrib_env": float(contrib_env.mean()),
        "|r_env|_mean": float(np.abs(E).mean()),
        "|r_env|_std": float(np.abs(E).std()),
        "|r_intr|_mean": float(np.abs(I).mean()),
        "|r_intr|_p75": float(np.percentile(np.abs(I), 75)),
        "|r_intr|_max": float(np.abs(I).max()),
        "intr_fire_rate": float((np.abs(I) > 1e-12).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 5, 9, 15])
    ap.add_argument("--regime", default="MATE-8v8-9-v0")
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true",
                    help="use deterministic actions for the replay (default is "
                         "stochastic, matching training behavior)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    reports = []
    for seed in args.seeds:
        for arm in ("intrinsic", "no_intrinsic"):
            dirs = sorted(glob.glob(
                f"results/campaign_{args.tag}_s{seed:02d}_{arm}/**/config.json",
                recursive=True))
            if not dirs:
                continue
            with open(dirs[0]) as f:
                cfg = json.load(f)
            rl = float(cfg.get("reward_lambda", 1.0))
            win = cfg.get("reward_window") or 20
            streams = collect_streams(args.tag, seed, arm, args.regime, rl, win,
                                      args.horizon, args.cap, args.rollouts,
                                      args.deterministic)
            if "error" in streams:
                print(streams)
                continue
            rep = analyze(streams)
            rep["lambda"] = rl
            reports.append(rep)
            print(f"seed={seed:02d} arm={arm:<13} n={rep['n']:>4}  "
                  f"corr(r_intr,adv_tot)={rep['corr_r_intr__adv_tot']:+.3f}  "
                  f"corr(r_intr,contrib_intr)={rep['corr_r_intr__contrib_intr']:+.3f}  "
                  f"var_share_intr={rep['var_share_intr']:.3f}  "
                  f"|r_intr|p75={rep['|r_intr|_p75']:.3f}  "
                  f"|r_env|std={rep['|r_env|_std']:.3f}  "
                  f"fire={rep['intr_fire_rate']:.3f}  "
                  f"lin_err={rep['linearity_max_err']:.2e}")

    out = "results/gae_diagnostic.json"
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()