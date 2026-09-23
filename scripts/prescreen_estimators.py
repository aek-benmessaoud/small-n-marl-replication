"""
scripts/prescreen_estimators.py — pré-crible des estimateurs de richesse.

Mesure, sur des rollouts aléatoires, les propriétés de SIGNAL de chaque
estimateur de richesse avant de lancer une campagne:

  * densité    : fraction de pas où le delta intrinsèque unitaire ≠ 0
                 (un signal à ~0.5% est inapprenable en 30k étapes);
  * amplitude  : percentiles p50/p75/p95/p99 de |delta| unitaire
                 (sert à calibrer lambda: lambda = ratio*bruit/p75);
  * redondance : corr(delta, #cibles visibles par >=1 caméra) et
                 corr(delta, r_env clippé) — le proxy de l'hypothèse
                 "l'intrinsèque ne fait que recompenser le tracking".

Estimateurs comparés (tous sur les comptes de configurations par cible):
  chao_bias_cap / chao_original / jackknife (U=F1) / ace (ACE-U),
  dense_cvar (variance circulaire, mode dense de final4) / dense_spread
  (étalement angulaire max, window).

Usage:
    .venv\\Scripts\\python.exe scripts\\prescreen_estimators.py --regime MATE-4v2-9-v0 \
        --seeds 0..3 --episodes 2 --steps 400 --dense-window 25
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project09.config import BASE_SEED, DEFAULT_REGIME, SEED_STRIDE  # noqa: E402
from project09.environment.mate_env import make_mate, num_entities, state_arrays  # noqa: E402
from project09.estimators.continuous import TargetTracker  # noqa: E402
from project09.estimators.richness import ace_u, chao_u, jackknife_u  # noqa: E402

WINDOW = 25


def collect_estimator_state(tracker, window):
    """Dictionnaire {label: valeur du signal} après l'observation courante."""
    visit = tracker.visit()
    known = tracker.known()
    obs = tracker.obs()
    tu = int(np.sum(tracker.underdetermined()))
    return {
        "chao_bias_cap": chao_u(visit, known, obs, total_unknown=tu,
                                variant="bias_cap"),
        "chao_original": chao_u(visit, known, obs, total_unknown=tu,
                                variant="original"),
        "jackknife": jackknife_u(visit, known, obs),
        "ace": ace_u(visit, known, obs, total_unknown=tu),
        "dense_cvar": tracker.variance_confidence_sum(window=window),
        "dense_spread": tracker.dense_confidence_sum(window=window),
    }


def run(regime, seed, episodes, steps, window, env_clip):
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=seed)
    env.seed(seed)
    tracker = TargetTracker(int(env.unwrapped.num_targets))
    rng = np.random.default_rng(seed + 999)
    n_cam = num_entities(env)[0]

    cols = {label: [] for label in collect_estimator_state(tracker, window)}
    vis, renv = [], []

    for _ in range(episodes):
        env.reset()
        tracker.reset()
        cam, tar, view = state_arrays(env)
        tracker.observe(cam, tar, view)
        prev = collect_estimator_state(tracker, window)
        done = False
        n = 0
        while not done and n < steps:
            act = rng.uniform(-1.0, 1.0, size=(n_cam, 2))
            _, r, done, _ = env.step(act)
            cam, tar, view = state_arrays(env)
            tracker.observe(cam, tar, view)
            after = collect_estimator_state(tracker, window)
            u_max = tracker.u_max()
            for label in cols:
                d = (prev[label] - after[label]) / u_max
                cols[label].append(float(d))
            vis.append(float(np.sum(view.any(axis=0))))
            renv.append(float(np.clip(r, -env_clip, env_clip)))
            prev = after
            n += 1
    env.close()
    return cols, vis, renv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--seeds", default="0..3")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dense-window", type=int, default=WINDOW)
    ap.add_argument("--env-clip", type=float, default=2.0)
    args = ap.parse_args()

    lo, hi = [int(x) for x in args.seeds.split("..")]
    seeds = list(range(lo, hi + 1))
    print(f"=== prescreen {args.regime} | seeds={seeds} ep={args.episodes} "
          f"steps={args.steps} dense_window={args.dense_window} "
          f"env_clip={args.env_clip} ===")

    all_cols = {label: [] for label in [
        "chao_bias_cap", "chao_original", "jackknife", "ace",
        "dense_cvar", "dense_spread"]}
    all_vis, all_renv = [], []
    for seed in seeds:
        cols, vis, renv = run(args.regime, BASE_SEED + seed * SEED_STRIDE,
                              args.episodes, args.steps, args.dense_window,
                              args.env_clip)
        for label in all_cols:
            all_cols[label].extend(cols[label])
        all_vis.extend(vis)
        all_renv.extend(renv)

    vis = np.asarray(all_vis)
    renv = np.asarray(all_renv)
    n = len(vis)
    print(f"total steps: {n} | mean #visible={vis.mean():.3f} | "
          f"r_env: mean={renv.mean():+.3f} std={renv.std():.3f}")

    print(f"\n{'estimator':<16}{'density%':>9}{'p50':>9}{'p75':>9}"
          f"{'p95':>9}{'p99':>9}{'corr_vis':>10}{'corr_env':>10}")
    for label in all_cols:
        d = np.asarray(all_cols[label])
        nonzero = d[d != 0]
        dens = 100.0 * np.mean(d != 0)
        if len(nonzero) == 0:
            print(f"{label:<16}{dens:>9.2f}{'-':>9}{'-':>9}{'-':>9}{'-':>9}"
                  f"{'-':>10}{'-':>10}")
            continue
        ab = np.abs(nonzero)
        corr_v = float(np.corrcoef(d, vis)[0, 1])
        corr_e = float(np.corrcoef(d, renv)[0, 1])
        print(f"{label:<16}{dens:>9.2f}{np.percentile(ab, 50):>9.5f}"
              f"{np.percentile(ab, 75):>9.5f}{np.percentile(ab, 95):>9.5f}"
              f"{np.percentile(ab, 99):>9.5f}{corr_v:>10.3f}{corr_e:>10.3f}")


if __name__ == "__main__":
    main()
