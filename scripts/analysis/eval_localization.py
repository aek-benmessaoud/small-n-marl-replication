"""
scripts/eval_localization.py — re-évalue des checkpoints MAPPO avec des
métriques de LOCALISATION (oracle CRLB, éval-only), pas de couverture.

Motivation: l'intrinsèque Chao-U/richesse est conçue pour la localisation
(>= 2 configurations angulaires bien séparées par cible), et les campagnes
final4 / chao4v8 ont montré C_end (diversité) plus élevé SANS gain de
couverture. Cette éval teste si la diversité accumulée se traduit en
localisation réellement exploitable:

  simult_loc : fraction (sur les pas) de cibles bien-localisables
               (>= 2 caméras qui la voient simultanément, CRLB <= seuil)
               -> métrique dense moyenne sur l'épisode
  loc_peak   : fraction de cibles qui ont été bien-localisées au moins une
               fois dans l'épisode
  loc_final  : fraction de cibles bien-localisées à l'état final
  gn_err     : erreur Gauss-Newton (triangulation bearing-only) moyenne sur
               les pas où une cible est bien-localisable (≥ 2 caméras)

--mode seq (oracle séquentiel, compatible cibles mobiles) :
  seq_loc    : fraction des pas où une cible est localisable depuis sa FENÊTRE
               glissante d'observations (CRLB de l'historique récent <= seuil)
  seq_peak   : fraction de cibles localisables au moins une fois (fenêtre)
  seq_err    : erreur Gauss-Newton depuis les relèvements de la fenêtre, vs
               position courante de la cible (unités de terrain)

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_localization.py --tag chao4v8 \
        --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 3 [--mode seq]
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (  # noqa: E402
    BASE_SEED, DEFAULT_REGIME, MAPPO_CLIP, MAPPO_EPOCHS, MAPPO_GAMMA,
    MAPPO_HIDDEN, MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCH, MAX_STEPS,
    QUALITY_SIGMA_BEARING_DEG, QUALITY_THRESHOLD, SEED_STRIDE,
)
from src.environment.mate_env import make_mate, num_entities, state_arrays  # noqa: E402
from src.estimators.validation import (  # noqa: E402
    crlb_bound, gauss_newton_localize, per_target_bounds, well_localized,
)
from src.rl.mappo import MAPPO  # noqa: E402


def _pairwise_intersection(oc, theta, rng):
    """Robust GN init: median of pairwise ray intersections of bearings with
    separation > ~30 degrees (scale/translation invariant). Falls back to the
    observer centroid when fewer than 2 well-separated pairs exist."""
    oc = np.asarray(oc, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    n = len(oc)
    pts = []
    idx = rng.permutation(n)
    for _ in range(min(200, n * n)):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        if abs(np.sin(th[i] - th[j])) < 0.5:
            continue
        d1 = np.array([np.cos(th[i]), np.sin(th[i])])
        d2 = np.array([np.cos(th[j]), np.sin(th[j])])
        A = np.stack([d1, -d2], axis=1)
        b = oc[j] - oc[i]
        try:
            s, t = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        pt = oc[i] + s * d1
        if np.all(np.isfinite(pt)) and np.all(np.abs(pt) < 1e4):
            pts.append(pt)
        if len(pts) >= 60:
            break
    if len(pts) < 2:
        return oc.mean(axis=0)
    return np.median(np.asarray(pts), axis=0)


def _gn_scaled(observers, theta, rng=None):
    """Gauss-Newton on scale-normalized coordinates (bearings are
    translation/scale-invariant); rescale the estimate back to terrain units.
    Uses a ray-intersection median as init (a centroid init bails out because
    the normal matrix looks degenerate far from the target)."""
    rng = rng or np.random.default_rng(0)
    oc = np.asarray(observers, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    c = oc.mean(axis=0)
    s = float(np.median(np.linalg.norm(oc - c, axis=1))) or 1.0
    init = (_pairwise_intersection(oc, th, rng) - c) / s
    est = gauss_newton_localize((oc - c) / s, theta, init=init)
    if not np.isfinite(est[0]):
        return est
    return float(est[0] * s + c[0]), float(est[1] * s + c[1])


def eval_ckpt(tag, arm, seed, regime, n_ep, cap):
    env_seed = BASE_SEED + seed * SEED_STRIDE
    fs = glob.glob(f"results/campaign_{tag}_s{seed:02d}_{arm}/*/latest.pt")
    if not fs:
        return None
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=env_seed)
    env.seed(env_seed)
    n_cam, n_tar, _ = num_entities(env)
    obs_dim = env.reset().shape[1]
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                  seed=env_seed * 1000 + 1, hidden=MAPPO_HIDDEN, lr=MAPPO_LR,
                  gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA, clip=MAPPO_CLIP,
                  epochs=MAPPO_EPOCHS, minibatch=MAPPO_MINIBATCH,
                  entropy_coef=0.0, frame_skip=1)
    mappo.load(fs[0])

    sums = {"simult": 0.0, "steps": 0, "gn_n": 0.0, "gn_sum": 0.0}
    peaks = np.zeros(n_tar, dtype=bool)
    finals = []
    rng = np.random.default_rng(env_seed)
    for _ in range(n_ep):
        obs = env.reset()
        done = False
        st = 0
        ep_peak = np.zeros(n_tar, dtype=bool)
        while not done and st < cap:
            a, _ = mappo.act_batch(obs, deterministic=True)
            obs, _, done, _ = env.step(a)
            cam, tar, view = state_arrays(env)
            bounds = per_target_bounds(cam, tar, view,
                                       sigma_deg=QUALITY_SIGMA_BEARING_DEG)
            good = np.isfinite(bounds) & (bounds <= QUALITY_THRESHOLD)
            sums["simult"] += float(np.mean(good)) if n_tar else 0.0
            sums["steps"] += 1
            ep_peak |= good
            # Gauss-Newton error on every well-localizable target right now
            for t in np.flatnonzero(good):
                obs_cams = np.flatnonzero(view[:, t])
                if obs_cams.size < 2:
                    continue
                observers = cam[obs_cams]
                true = tar[t]
                theta = np.arctan2(true[1] - observers[:, 1],
                                   true[0] - observers[:, 0])
                theta += rng.normal(0.0, np.deg2rad(QUALITY_SIGMA_BEARING_DEG),
                                    size=theta.size)
                est = _gn_scaled(observers, theta)
                if np.isfinite(est[0]):
                    sums["gn_n"] += 1
                    sums["gn_sum"] += float(np.hypot(est[0] - true[0],
                                                     est[1] - true[1]))
            st += 1
        peaks += ep_peak
        # final state
        cam, tar, view = state_arrays(env)
        bounds = per_target_bounds(cam, tar, view,
                                   sigma_deg=QUALITY_SIGMA_BEARING_DEG)
        finals.append(well_localized(bounds, QUALITY_THRESHOLD))
    env.close()

    simult = sums["simult"] / max(sums["steps"], 1)
    return {
        "seed": seed, "arm": arm, "simult_loc": simult,
        "loc_peak": float(np.mean(peaks)) / n_ep,
        "loc_final": float(np.mean(finals)),
        "gn_err": (sums["gn_sum"] / sums["gn_n"]) if sums["gn_n"] else float("nan"),
        "gn_n": int(sums["gn_n"]),
    }


def _load_checkpoint(tag, arm, seed, regime):
    env_seed = BASE_SEED + seed * SEED_STRIDE
    fs = glob.glob(f"results/campaign_{tag}_s{seed:02d}_{arm}/*/latest.pt")
    if not fs:
        return None
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=env_seed)
    env.seed(env_seed)
    n_cam, n_tar, _ = num_entities(env)
    obs_dim = env.reset().shape[1]
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                  seed=env_seed * 1000 + 1, hidden=MAPPO_HIDDEN, lr=MAPPO_LR,
                  gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA, clip=MAPPO_CLIP,
                  epochs=MAPPO_EPOCHS, minibatch=MAPPO_MINIBATCH,
                  entropy_coef=0.0, frame_skip=1)
    mappo.load(fs[0])
    return env, mappo


def eval_ckpt_seq(tag, arm, seed, regime, n_ep, cap, window=60, max_obs=120):
    """Windowed (sequential) localization oracle: a target is localizable when
    its recent observation history (last `window` steps) determines it
    (CRLB <= threshold). This is the mobile-compatible version of the
    accumulated-bearings model the intrinsic reward optimizes."""
    env_seed = BASE_SEED + seed * SEED_STRIDE
    loaded = _load_checkpoint(tag, arm, seed, regime)
    if loaded is None:
        return None
    env, mappo = loaded
    n_cam, n_tar, _ = num_entities(env)

    sums = {"loc": 0.0, "steps": 0.0, "gn_n": 0.0, "gn_sum": 0.0}
    peaks = np.zeros(n_tar, dtype=bool)
    rng = np.random.default_rng(env_seed)
    for _ in range(n_ep):
        obs = env.reset()
        done = False
        st = 0
        wc = [[] for _ in range(n_tar)]  # camera positions per observation
        wt = [[] for _ in range(n_tar)]  # target position at observation time
        ws = [[] for _ in range(n_tar)]  # step index per observation
        wci = [[] for _ in range(n_tar)]  # camera index per observation
        ep_peak = np.zeros(n_tar, dtype=bool)
        while not done and st < cap:
            a, _ = mappo.act_batch(obs, deterministic=True)
            obs, _, done, _ = env.step(a)
            cam, tar, view = state_arrays(env)
            for t in range(n_tar):
                cams_t = np.flatnonzero(view[:, t])
                if cams_t.size:
                    wc[t].append(cam[cams_t])
                    wt[t].append(np.tile(tar[t], (cams_t.size, 1)))
                    ws[t].append(np.full(cams_t.size, st))
                    wci[t].append(cams_t)
                while ws[t] and ws[t][0][0] < st - window:
                    wc[t].pop(0)
                    wt[t].pop(0)
                    ws[t].pop(0)
                    wci[t].pop(0)
                while ws[t] and sum(len(x) for x in ws[t]) > max_obs:
                    wc[t].pop(0)
                    wt[t].pop(0)
                    ws[t].pop(0)
                    wci[t].pop(0)
            good = np.zeros(n_tar, dtype=bool)
            for t in range(n_tar):
                if not wc[t]:
                    continue
                oc = np.concatenate(wc[t])
                bound = crlb_bound(oc, tar[t], sigma_deg=QUALITY_SIGMA_BEARING_DEG)
                good[t] = np.isfinite(bound) and bound <= QUALITY_THRESHOLD
            sums["loc"] += float(np.mean(good)) if n_tar else 0.0
            sums["steps"] += 1
            ep_peak |= good
            for t in np.flatnonzero(good):
                # Only genuinely multi-view geometry (>= 2 distinct cameras in
                # the window) yields a well-conditioned GN solve; single-camera
                # parallax history is near-collinear.
                if len(np.unique(np.concatenate(wci[t]))) < 2:
                    continue
                oc = np.concatenate(wc[t])
                ot = np.concatenate(wt[t])
                theta = np.arctan2(ot[:, 1] - oc[:, 1], ot[:, 0] - oc[:, 0])
                theta += rng.normal(0.0, np.deg2rad(QUALITY_SIGMA_BEARING_DEG),
                                    size=theta.size)
                est = _gn_scaled(oc, theta)
                if np.isfinite(est[0]):
                    sums["gn_n"] += 1
                    sums["gn_sum"] += float(np.hypot(est[0] - tar[t][0],
                                                     est[1] - tar[t][1]))
            st += 1
        peaks += ep_peak
    env.close()

    return {
        "seed": seed, "arm": arm,
        "seq_loc": sums["loc"] / max(sums["steps"], 1.0),
        "seq_peak": float(np.mean(peaks)) / n_ep,
        "seq_err": (sums["gn_sum"] / sums["gn_n"]) if sums["gn_n"] else float("nan"),
        "seq_n": int(sums["gn_n"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--seeds", default="0..3")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--cap", type=int, default=MAX_STEPS)
    ap.add_argument("--mode", choices=["simult", "seq"], default="simult",
                    help="simult: instantaneous CRLB; seq: windowed oracle "
                         "(mobile-compatible accumulated model)")
    ap.add_argument("--arms", default="intrinsic,no_intrinsic",
                    help="comma list of the two arm names, in order "
                         "treatment,control")
    ap.add_argument("--out-suffix", default="",
                    help="optional suffix appended to the output JSON name "
                         "(allows parallel seed splits to write distinct "
                         "files that are merged later)")
    args = ap.parse_args()

    lo, hi = [int(x) for x in args.seeds.split("..")]
    seeds = list(range(lo, hi + 1))
    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]
    print(f"=== eval_localization {args.tag} | {args.regime} | mode={args.mode} "
          f"| seeds={seeds} ep={args.episodes} | arms={arm_a},{arm_b} ===")

    if args.mode == "seq":
        metrics = ("seq_loc", "seq_peak", "seq_err")
    else:
        metrics = ("simult_loc", "loc_peak", "loc_final", "gn_err")

    rows = []
    for s in seeds:
        if args.mode == "seq":
            intr = eval_ckpt_seq(args.tag, arm_a, s, args.regime,
                                 args.episodes, args.cap)
            noi = eval_ckpt_seq(args.tag, arm_b, s, args.regime,
                                args.episodes, args.cap)
        else:
            intr = eval_ckpt(args.tag, arm_a, s, args.regime,
                             args.episodes, args.cap)
            noi = eval_ckpt(args.tag, arm_b, s, args.regime,
                            args.episodes, args.cap)
        if intr is None or noi is None:
            print(f"  missing ckpt for seed {s}")
            continue
        rows.append((intr, noi))

    if args.mode == "seq":
        print(f"\n{'seed':<5}{'arm':<13}{'seq_loc':>10}{'seq_peak':>10}"
              f"{'seq_err':>9}{'seq_n':>7}")
        for intr, noi in rows:
            for r in (intr, noi):
                print(f"{r['seed']:<5}{r['arm']:<13}{r['seq_loc']:>10.3f}"
                      f"{r['seq_peak']:>10.3f}{r['seq_err']:>9.1f}{r['seq_n']:>7d}")
    else:
        print(f"\n{'seed':<5}{'arm':<13}{'simult_loc':>11}{'loc_peak':>10}"
              f"{'loc_final':>10}{'gn_err':>9}{'gn_n':>6}")
        for intr, noi in rows:
            for r in (intr, noi):
                print(f"{r['seed']:<5}{r['arm']:<13}{r['simult_loc']:>11.3f}"
                      f"{r['loc_peak']:>10.3f}{r['loc_final']:>10.3f}"
                      f"{r['gn_err']:>9.1f}{r['gn_n']:>6d}")

    # persist (merge with an existing aggregate file so partial seed ranges
    # can be re-run without recomputing the rest)
    import json
    out_dir = os.path.join("results", f"campaign_{args.tag}", "eval_localization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
                            f"{args.regime}_{args.mode}_ep{args.episodes}"
                            f"{args.out_suffix}.json")
    merged = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path)):
            merged[(r["seed"], r["arm"])] = r
    for r in (dict(x) for intr, noi in rows for x in (intr, noi)):
        merged[(r["seed"], r["arm"])] = r
    flat = [merged[k] for k in sorted(merged)]
    with open(out_path, "w") as f:
        json.dump(flat, f, indent=2)

    print("\n=== AGGREGATE ===")
    for metric in metrics:
        a = [r[metric] for r, _ in rows if np.isfinite(r[metric])]
        b = [r[metric] for _, r in rows if np.isfinite(r[metric])]
        if not a or not b:
            print(f"{metric:<10} incomplete")
            continue
        wins = sum(x > y for x, y in zip(a, b))
        print(f"{metric:<10} intr={np.mean(a):+.4f}  no_intr={np.mean(b):+.4f}  "
              f"delta={np.mean(a) - np.mean(b):+.4f}  wins={wins}/{len(a)}")


if __name__ == "__main__":
    main()
