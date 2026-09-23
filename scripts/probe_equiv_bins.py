"""
Equivalence probe: same trajectories, greedy vs n_bins counting.

Replays the SAME random rollout for several seeds and compares the per-target
configuration-count sequences (visit) produced by:
   - greedy clustering (tol=15deg, cap=8)    [the manuscript mechanism]
   - n_bins discretization K = 8/12/16/24    [the step-6 knob]

Because the rollout (env seed + actions) is identical inside each seed, the
difference between counters is STRUCTURAL (zero trajectory variance). This
decides whether bins=K is a re-parameterization of greedy (counts match) or a
different mechanism (counts diverge), answering the user's §1-§3 before we
throw more campaign compute at b16/b24.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project09.config import BASE_SEED, DEFAULT_REGIME, SEED_STRIDE  # noqa: E402
from project09.environment.mate_env import make_mate, num_entities, state_arrays  # noqa: E402
from project09.estimators.continuous import TargetTracker  # noqa: E402


def rollout_counts(regime, seed, n_bins, steps, episodes=2):
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=seed)
    env.seed(seed)
    tracker = TargetTracker(int(env.unwrapped.num_targets), n_bins=n_bins)
    rng = np.random.default_rng(seed + 999)      # SAME rng for every counter → same actions
    n_cam = num_entities(env)[0]
    seq = []
    for _ in range(episodes):
        env.reset()
        tracker.reset()
        cam, tar, view = state_arrays(env)
        tracker.observe(cam, tar, view)
        done = False
        n = 0
        while not done and n < steps:
            act = rng.uniform(-1.0, 1.0, size=(n_cam, 2))
            _, r, done, _ = env.step(act)
            cam, tar, view = state_arrays(env)
            tracker.observe(cam, tar, view)
            seq.append(tracker.visit().copy())
            n += 1
    env.close()
    return np.asarray(seq)


def main():
    regime = "MATE-8v8-9-v0"
    steps = 150
    seeds = [BASE_SEED + s * SEED_STRIDE for s in (0, 1, 2, 3)]
    ks = [None, 7, 8, 12, 16, 24]
    labels = {None: "greedy(15deg)", 7: "bins=7", 8: "bins=8",
              12: "bins=12", 16: "bins=16", 24: "bins=24"}

    print(f"regime={regime} seeds={len(seeds)} steps={steps} "
          f"| comparing config-count sequences on IDENTICAL rollouts")
    print(f"{'counter':<15}{'mean_visit':>11}{'mean_F1':>9}{'mean_F2':>9}"
          f"{'frac_cap':>10}{'vs_greedy(max_|d|)':>20}", flush=True)

    data = {}
    for k in ks:
        seqs = []
        for sd in seeds:
            seqs.append(rollout_counts(regime, sd, k, steps))
            print(f"  [{labels[k]}] seed done", flush=True)
        S = np.concatenate(seqs, axis=0)
        visit = S
        mean_visit = visit.mean()
        mean_F1 = np.mean((visit == 1).sum(axis=-1))
        mean_F2 = np.mean((visit == 2).sum(axis=-1))
        frac_cap = np.mean((visit >= 8).sum(axis=-1) / visit.shape[-1])
        data[k] = visit
        if k is None:
            print(f"{labels[k]:<15}{mean_visit:>11.3f}{mean_F1:>9.3f}"
                  f"{mean_F2:>9.3f}{frac_cap:>10.4f}{'-':>20}")
        else:
            g = data[None]
            # elementwise max |delta| per target step, over the whole trace
            dmax = float(np.abs(visit - g).max())
            print(f"{labels[k]:<15}{mean_visit:>11.3f}{mean_F1:>9.3f}"
                  f"{mean_F2:>9.3f}{frac_cap:>10.4f}{dmax:>20.3f}")


if __name__ == "__main__":
    main()