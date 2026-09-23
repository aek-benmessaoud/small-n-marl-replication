"""
scripts/launch_campaign.py — parallel campaign launcher.

Spawns --workers subprocesses that each handle a contiguous chunk of seeds,
running `run_mappo.py` (and optionally `run_baselines.py`) for every seed in
its chunk. Each worker sets TORCH_THREADS=1 (CPU-bound env is the bottleneck)
and writes its stdout/stderr to results/campaign_<tag>/worker_<i>.log.

Estimates (machine-dependent) are printed before launching, then every worker
is started. Ctrl-C kills all workers.

Usage:
    .venv\\Scripts\\python.exe scripts\\launch_campaign.py --workers 6 --tag campaign
    .venv\\Scripts\\python.exe scripts\\launch_campaign.py --workers 6 --tag campaign --no-baselines
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEFAULT_REGIME, NUM_RUNS, RESULTS_DIR  # noqa: E402


def chunk_seeds(seeds, workers):
    out = []
    n = len(seeds)
    for w in range(workers):
        out.append(seeds[w::workers])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", default="0..29")
    ap.add_argument("--tag", default="campaign")
    ap.add_argument("--regime", default=DEFAULT_REGIME)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--reward", choices=["chao", "dense", "loc", "rnd"], default="chao",
                    help="reward objective (see run_mappo): 'loc' adds an "
                         "explicit CRLB localization term to the env reward, "
                         "'rnd' adds a Random Network Distillation novelty "
                         "bonus over camera poses")
    ap.add_argument("--reward-lambda", default="auto")
    ap.add_argument("--lambda-target-ratio", type=float, default=0.25)
    ap.add_argument("--dense-window", type=int, default=None)
    ap.add_argument("--reward-window", type=int, default=None)
    ap.add_argument("--num-bins", type=int, default=None,
                    help="bin-sensitivity (roadmap step 6): discretize the "
                         "circle in this many angular bins; None = greedy "
                         "clustering (default)")
    ap.add_argument("--r-intr-clip", type=float, default=None)
    ap.add_argument("--env-reward-clip", type=float, default=None)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--baselines", action="store_true", dest="baselines")
    ap.add_argument("--no-baselines", action="store_false", dest="baselines")
    ap.set_defaults(baselines=True)
    ap.add_argument("--skip-no-intrinsic", action="store_true",
                    help="train only the treatment arm (control checkpoints "
                         "reused from a previous campaign)")
    ap.add_argument("--warm-start", action="store_true",
                    help="warm-start mode: for each seed, 'warm_intrinsic' "
                         "initializes from the source campaign's intrinsic "
                         "checkpoint and 'no_intrinsic' from its no_intrinsic "
                         "checkpoint; BOTH then train --steps on the pure "
                         "reward (equal-compute continuation control). "
                         "Forces --no-baselines.")
    ap.add_argument("--source-tag", default="chao4v8win",
                    help="tag of the source campaign (used with --warm-start)")
    ap.add_argument("--source-ts", default=None,
                    help="timestamp subdir of the source runs; if None, the "
                         "unique results/campaign_<source-tag>_s*_<arm>/ dir")
    args = ap.parse_args()

    lo, hi = [int(x) for x in args.seeds.split("..")]
    seeds = list(range(lo, hi + 1))

    if args.warm_start:
        args.baselines = False
        args.reward = args.reward or "chao"

    project_root = Path(__file__).resolve().parent.parent
    py = sys.executable
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(RESULTS_DIR, f"campaign_{args.tag}")
    os.makedirs(log_dir, exist_ok=True)

    per_seed_min = 102.0  # measured on MATE-4v8-9-v0, frame_skip=1 (est.)
    serial_min = per_seed_min * len(seeds)
    eff = args.workers / 1.45  # HT contention on 4P+4T cores
    est_h = serial_min / 60 / max(eff, 1.0)

    print("=" * 64)
    print("CAMPAIGN LAUNCH")
    print(f"  workers            : {args.workers}")
    print(f"  seeds              : {seeds}")
    print(f"  regime             : {args.regime}")
    print(f"  steps/arm          : {args.steps}")
    print(f"  eval episodes      : {args.eval_episodes}")
    print(f"  reward mode        : {args.reward} (lambda={args.reward_lambda})")
    print(f"  reward window      : {args.reward_window}")
    print(f"  num bins           : {args.num_bins}")
    print(f"  baselines          : {args.baselines}")
    if args.warm_start:
        print(f"  warm-start         : YES (from campaign_{args.source_tag}"
              f", source-ts={args.source_ts or 'auto'})")
    print(f"  serial estimate    : {serial_min:.0f} min")
    print(f"  parallel estimate  : ~{est_h:.1f} h (wall)")
    print("=" * 64)

    cmds = []
    for w, chunk in enumerate(chunk_seeds(seeds, args.workers)):
        if not chunk:
            continue
        log_path = os.path.join(log_dir, f"worker_{w}.log")
        invocations = []
        base = [py, "scripts/run_mappo.py", "--tag", args.tag,
                "--regime", args.regime, "--steps", str(args.steps),
                "--eval-episodes", str(args.eval_episodes),
                "--eval-every", str(args.eval_every),
                "--reward", args.reward,
                "--reward-lambda", args.reward_lambda,
                "--lambda-target-ratio", str(args.lambda_target_ratio),
                "--entropy-coef", str(args.entropy_coef),
                "--lr", str(args.lr),
                "--timestamp", ts]
        if args.dense_window is not None:
            base += ["--dense-window", str(args.dense_window)]
        if args.reward_window is not None:
            base += ["--reward-window", str(args.reward_window)]
        if args.num_bins is not None:
            base += ["--num-bins", str(args.num_bins)]
        if args.r_intr_clip is not None:
            base += ["--r-intr-clip", str(args.r_intr_clip)]
        if args.env_reward_clip is not None:
            base += ["--env-reward-clip", str(args.env_reward_clip)]
        if args.skip_no_intrinsic:
            base += ["--skip-no-intrinsic"]
        if args.warm_start:
            import glob as _glob
            for s in chunk:
                for arm, src_arm in (("warm_intrinsic", "intrinsic"),
                                     ("no_intrinsic", "no_intrinsic")):
                    pat = os.path.join(
                        RESULTS_DIR,
                        f"campaign_{args.source_tag}_s{s:02d}_{src_arm}",
                        args.source_ts if args.source_ts else "*", "latest.pt")
                    fs = _glob.glob(pat)
                    if not fs:
                        raise SystemExit(
                            f"[WARM-START] no source checkpoint for seed {s} "
                            f"arm {arm}: {pat}")
                    invocations.append(base + ["--seed", str(s),
                                               "--arm", arm,
                                               "--init-checkpoint", fs[0]])
        else:
            for s in chunk:
                invocations.append(base + ["--seed", str(s)])
            if args.baselines:
                base_bl = [py, "scripts/run_baselines.py", "--tag", args.tag,
                           "--regime", args.regime,
                           "--eval-episodes", str(args.eval_episodes), "--timestamp", ts]
                for s in chunk:
                    invocations.append(base_bl + ["--seed", str(s)])
        cmds.append((invocations, log_path))

    env = dict(os.environ)
    env["TORCH_THREADS"] = "1"
    procs = []
    for invocations, log_path in cmds:
        lf = open(log_path, "w", encoding="utf-8", buffering=1)
        p = subprocess.Popen(invocations.pop(0), cwd=project_root, env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
        procs.append({"p": p, "lf": lf, "log": log_path,
                      "pending": invocations})
        print(f"  started worker: {os.path.basename(log_path)} (pid {p.pid})",
              flush=True)

    try:
        while procs:
            for slot in list(procs):
                rc = slot["p"].poll()
                if rc is None:
                    continue
                if rc != 0:
                    print(f"  {os.path.basename(slot['log'])} invocation rc={rc}",
                          flush=True)
                if slot["pending"]:
                    nxt = slot["pending"].pop(0)
                    slot["p"] = subprocess.Popen(nxt, cwd=project_root, env=env,
                                                 stdout=slot["lf"],
                                                 stderr=subprocess.STDOUT)
                else:
                    slot["lf"].close()
                    print(f"  finished {os.path.basename(slot['log'])}",
                          flush=True)
                    procs.remove(slot)
            time.sleep(5)
    except KeyboardInterrupt:
        print("  Ctrl-C: killing workers...", flush=True)
        for slot in procs:
            if slot["p"].poll() is None:
                slot["p"].kill()
            slot["lf"].close()
        sys.exit(1)

    # summary
    import glob
    done = glob.glob(os.path.join(RESULTS_DIR, f"campaign_{args.tag}_s*", ts, "DONE"))
    print(f"[LAUNCH-OK] {len(done)} DONE markers written under results/", flush=True)


if __name__ == "__main__":
    main()
