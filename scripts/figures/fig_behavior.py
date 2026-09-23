"""scripts/fig_behavior.py — behavior-snapshot figure (Fig. B).

Plays the two arms (intrinsic vs no_intrinsic) of a campaign *in lock-step* on
the same deterministic episode (same env seed), and renders a side-by-side
top-down snapshot of the cameras' field-of-view cones over the targets.

Because MATE cameras are anchored (action = rotation + zoom), the behavioral
contrast lives in the orientation / sight-range of the FOV cones, not in camera
positions. The snapshot step is chosen as the step where the two arms most
differ in a simple coverage score (mean targets-in-cone, incl. partial overlap).

Usage:
    .venv\\Scripts\\python.exe scripts\\fig_behavior.py --tag chao8v8 \\
        --regime MATE-8v8-9-v0 --seeds 6 --window 20 \\
        --cap 600 --out paper/figures/fig_behavior.pdf --dpi 200
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from src.config import (  # noqa: E402
    BASE_SEED, MAPPO_CLIP, MAPPO_EPOCHS, MAPPO_GAMMA, MAPPO_HIDDEN,
    MAPPO_LAMBDA, MAPPO_LR, MAPPO_MINIBATCH, SEED_STRIDE,
)
from src.environment.mate_env import make_mate, state_arrays  # noqa: E402
from src.rl.mappo import MAPPO  # noqa: E402

XMIN, XMAX, YMIN, YMAX = -1000.0, 1000.0, -1000.0, 1000.0


def load(tag, arm, seed, regime):
    env_seed = BASE_SEED + seed * SEED_STRIDE
    fs = sorted(glob.glob(f"results/campaign_{tag}_s{seed:02d}_{arm}/*/latest.pt"))
    env = make_mate(env_id="MultiAgentTracking-v0", config=regime, seed=env_seed)
    env.seed(env_seed)
    n_cam = env.unwrapped.num_cameras
    obs_dim = env.reset().shape[1]
    mappo = MAPPO(obs_dim=obs_dim, act_dim=2, n_cam=n_cam,
                  seed=env_seed * 1000 + 1, hidden=MAPPO_HIDDEN, lr=MAPPO_LR,
                  gamma=MAPPO_GAMMA, lam=MAPPO_LAMBDA, clip=MAPPO_CLIP,
                  epochs=MAPPO_EPOCHS, minibatch=MAPPO_MINIBATCH,
                  entropy_coef=0.0, frame_skip=1)
    mappo.load(fs[0])
    return env, mappo


def cone_in_target(cam_xy, ang_deg, fov_deg, sight, tar_xy):
    """Boolean: target centre inside the FOV cone (orientation + viewing angle,
    range = sight). Uses target centre only — a cheap approximation of the
    extent check the engine performs."""
    to_cam = cam_xy - tar_xy
    d = np.linalg.norm(to_cam, axis=1)
    out = np.zeros(len(tar_xy), dtype=bool)
    ok = d < sight
    if ok.any():
        ang = np.degrees(np.arctan2(-(tar_xy[ok, 1] - cam_xy[1]),
                                    tar_xy[ok, 0] - cam_xy[0]))
        rel = (ang - ang_deg + 180.0) % 360.0 - 180.0
        out[ok] = np.abs(rel) <= fov_deg / 2.0
    return out


def step_state(env, mappo, obs):
    cam_xy, tar_xy, _ = state_arrays(env)
    cams = env.unwrapped.cameras
    ang = np.array([c.orientation for c in cams])
    fov = np.array([c.viewing_angle for c in cams])
    sight = np.array([c.sight_range for c in cams])
    n_cam = len(cams)
    n_tar = len(tar_xy)
    coverage = np.zeros(n_tar)
    for i in range(n_cam):
        coverage += cone_in_target(cam_xy[i], ang[i], fov[i], sight[i], tar_xy)
    a, _ = mappo.act_batch(obs, deterministic=True)
    obs2, r, done, _ = env.step(a)
    return obs2, done, dict(cam=cam_xy, tar=tar_xy, ang=ang, fov=fov,
                            sight=sight, cover=coverage,
                            multi=(coverage > 1.5).sum())


def run_arms(envs, mappos, cap):
    """Lock-step both arms; records state and coverage at every step."""
    states = [[], []]
    scores = [[], []]
    obs0 = envs[0].reset()
    obs1 = envs[1].reset()
    obs = [obs0, obs1]
    for k in range(2):
        s0 = step_state(envs[k], mappos[k], obs[k])
        obs[k] = s0[0]
        states[k].append(s0[2])
        scores[k].append(float(s0[2]["multi"]))
    done = [False, False]
    st = 1
    while (not all(done)) and st < cap:
        for k in range(2):
            if not done[k]:
                r = step_state(envs[k], mappos[k], obs[k])
                obs[k], done[k] = r[0], r[1]
                states[k].append(r[2])
                scores[k].append(float(r[2]["multi"]))
            else:
                states[k].append(states[k][-1])
                scores[k].append(scores[k][-1])
        st += 1
    return states, scores


def render_panel(ax, state, title, seed, step):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    for c in ax.spines.values():
        c.set_linewidth(0.5)
    cam = state["cam"]
    tar = state["tar"]
    ang = state["ang"]
    fov = state["fov"]
    sight = state["sight"]
    n_cam = len(cam)
    multi = state["cover"] > 1.5
    single = (state["cover"] > 0.5) & (~multi)
    for t in range(len(tar)):
        if multi[t]:
            ec = mfc = "#7b1fa2"
            ms = 4.2
        elif single[t]:
            ec = "#1f77b4"
            mfc = "none"
            ms = 3.5
        else:
            ec = "#9e9e9e"
            mfc = "none"
            ms = 3.0
        ax.plot(tar[t, 0], tar[t, 1], marker="s", ms=ms,
                color=ec, mfc=mfc, mec=ec, mew=0.8, zorder=3)
    for i in range(n_cam):
        r = min(sight[i], 900.0)
        wedge = Wedge((cam[i, 0], cam[i, 1]), r,
                      ang[i] - fov[i] / 2.0, ang[i] + fov[i] / 2.0,
                      width=r * 0.30, alpha=0.18, facecolor="#d62728",
                      edgecolor="none", zorder=2)
        ax.add_patch(wedge)
        ax.plot(cam[i, 0], cam[i, 1], marker="o", ms=4, color="#2a2a2a",
                zorder=4)
        ax.plot(cam[i, 0], cam[i, 1], marker="o", ms=2.2, color="#d62728",
                zorder=5)
    ax.text(0.02, 0.02, f"seed {seed} \\ step {step}",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
            color="#555555", family="monospace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chao8v8")
    ap.add_argument("--regime", default="MATE-8v8-9-v0")
    ap.add_argument("--seeds", default="6")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--fixed-step", type=int, default=None,
                    help="If set, use this step for all seeds instead of auto-selecting.")
    ap.add_argument("--out", default="paper/figures/fig_behavior.pdf")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
        axes = axes[np.newaxis, :]
    else:
        fig, axes = plt.subplots(len(seeds), 2, figsize=(8.6, 4.2 * len(seeds)))

    for r, seed in enumerate(seeds):
        envs, mappos = [], []
        for arm in ("intrinsic", "no_intrinsic"):
            e, m = load(args.tag, arm, seed, args.regime)
            envs.append(e)
            mappos.append(m)
        states, scores = run_arms(envs, mappos, args.cap)
        for k in range(2):
            envs[k].close()
        sc0, sc1 = np.array(scores[0]), np.array(scores[1])
        if args.fixed_step is not None:
            step = int(args.fixed_step)
        else:
            well = np.where((sc0 >= 1) & (sc1 == 0))[0]
            if len(well):
                step = int(np.median(well))
            else:
                diff = np.abs(sc0 - sc1)
                step = int(np.argmax(diff)) if len(diff) else 0
            step = max(step, 1)
        for k, name in enumerate(("intrinsic", "no_intrinsic")):
            ax = axes[r, k]
            render_panel(ax, states[k][step], name, seed, step)

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()