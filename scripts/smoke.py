"""
scripts/smoke.py — Phase-0 pipeline smoke under the RunContext protocol.

Runs one episode with random camera actions on the wrapped Chao-U environment
and prints U / F1 / F2 diagnostics. Validates end-to-end:
    make_mate -> ChaoUReward -> step loop -> RunContext golden trio.

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke.py --regime MATE-4v8-9-v0 --steps 300
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

import mate  # noqa: E402

from project09.config import DEFAULT_REGIME, REWARD_LAMBDA  # noqa: E402
from project09.devops.runner import RunContext  # noqa: E402
from project09.environment.mate_env import (  # noqa: E402
    make_mate, num_cameras_targets,
)
from project09.environment.reward import ChaoUReward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timestamp", default=None)
    args = ap.parse_args()

    config = {
        "experiment": "smoke",
        "phase": 0,
        "regime": args.regime,
        "seed": args.seed,
        "num_runs": 1,
        "reward_lambda": REWARD_LAMBDA,
        "chao_variant": "bias_cap",
        "max_steps": args.steps,
        "steps": args.steps,
    }

    with RunContext(experiment="smoke", phase=0, config=config,
                    timestamp=args.timestamp) as ctx:
        env = make_mate(env_id="MultiAgentTracking-v0", config=args.regime,
                        seed=args.seed)
        env = ChaoUReward(env, reward_lambda=REWARD_LAMBDA)
        n_cam, n_tar = num_cameras_targets(env)
        obs = env.reset()
        U0 = env.current_U()
        done = False
        total = 0.0
        steps = 0
        U_min, U_end = U0, U0
        F1_max = 0
        while not done and steps < args.steps:
            obs, r, done, info = env.step(env.action_space.sample())
            total += float(r)
            steps += 1
            U = env.current_U()
            U_min = min(U_min, U)
            U_end = U
            f1, f2 = env.current_components()
            F1_max = max(F1_max, int(f1))
        env.close()
        print(f"[SMOKE] regime={args.regime} cameras={n_cam} targets={n_tar} "
              f"steps={steps} episode_reward={total:.3f}")
        print(f"[SMOKE] U: start={U0:.2f} end={U_end:.2f} min={U_min:.2f} "
              f"F1_max={F1_max}")
        assert steps > 0, "smoke episode produced no steps"
        assert np.isfinite(total), "episode reward is not finite"
        assert np.isfinite(U_end), "U is not finite"
        print(f"[SMOKE-OK] reward={total:.3f}")


if __name__ == "__main__":
    main()
