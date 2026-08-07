# Project09 — MARL Localization-Aware Coverage on MATE (MAPPO + Chao-U)

Successor to **Project08** (grid-world, angular richness). Project09 transposes
the Chao-U richness signal from per-cell visit counts to **continuous moving
targets**: the localization unit is the *target*, and every camera→target
observation contributes a bearing configuration.

**Short thesis framing:** cameras must both cover (track) targets and
*bearing-localize* them (≥ 2 independent configurations). The raw MATE reward
covers tracking; the Chao-U intrinsic term shapes the remaining-work signal
`U(t)` so the team learns to resolve under-determined targets.

## Reward (locked convention)

    r = coverage_reward + LAMBDA * (U_t - U_{t+1}) / U_max

- `U` = Chao-U remaining-work estimate over targets (`F1²/(2(F2+1))`,
  `bias_cap` variant, capped at #under-determined known targets, floored at 1).
- `U_max = num_targets`. Positive intrinsic when `U` **decreases**.
- Targets never observed are excluded from the cap — the signal starts flat
  exactly like Project08's `get_total_undetermined`.

## Environment stack (locked, reproducible)

| Component | Version / pin |
|---|---|
| Python | 3.10 (`.venv`) |
| numpy | 1.26.4 |
| gym | 0.21.0 (patched sdist — see `docs/gym021.md`) |
| torch | 2.13.0+cpu |
| MATE | 0.1.0 (git submodule, pin `3e631c0`) |

Known quirks (verified):
- gym 0.21 `Tuple.seed(seed)` **hangs** on numpy 1.26
  (`np.random.choice(np.iinfo(int).max, replace=False)` allocates ~9.2e18
  array). Do not call `action_space.seed()`. Use explicit seeded policies.
- numpy 2.x / gym 0.26+ break MATE (`np.bool8`, `Generator.randint`). Do not
  upgrade.

## Regimes

| Regime | Config | Cameras | Targets | Use |
|---|---|---|---|---|
| `MATE-4v2-9-v0` | sparse | 4 | 2 | fast gates |
| `MATE-4v8-9-v0` | imbalanced (MATE default) | 4 | 8 | main |
| `MATE-8v8-9-v0` | dense | 8 | 8 | dense |

## Repository layout

    project09/
        config.py            single source of truth (all constants)
        estimators/
            angular.py       ported greedy angular clustering (byte-identical)
            richness.py      ported Chao-U / ACE-U estimators
            continuous.py    TargetTracker (target-centric bundle)
            validation.py    CRLB oracle (evaluation-only)
        environment/
            mate_env.py      make_mate / state_arrays / num_entities
            reward.py        ChaoUReward wrapper (locked step order)
        rl/
            models.py        shared MLPPolicy + CentralizedCritic
            mappo.py         GAE + clipped PPO + checkpoint/resume
            decode.py        obs-only decoder (fair baselines)
            heuristics.py    Random / GreedyTracking / ChaoUAware baselines
        devops/
            runner.py        ExperimentRunner / Tee / RunContext (Resume-Anywhere)
        analysis/
            stats.py         Wilcoxon, Holm-Bonferroni, gain, Fisher
    scripts/
        smoke.py             Phase-0 pipeline smoke (RunContext protocol)
    tests/
        test_mate_repro.py        Phase 0  determinism + state channel
        test_chaou_continuous.py  Phase 1  continuous == grid at lattice snap
        test_rewards.py           Phase 2  intrinsic sign/shape + wrapper
        test_devops.py            Phase 3  golden trio / DONE / resume
        test_mappo.py             Phase 3  MAPPO save-resume + train smoke
        test_heuristics.py        Phase 4  baselines run + obs decoder

## Resume-Anywhere devops protocol

Every experiment lives in `results/<experiment>/{timestamp}/`:

    config.json    full config + git commit hash + git status (BEFORE the run)
    run.log        tee'd stdout/stderr
    DONE           marker written ONLY on full success
    <artifacts>    CSVs, .pt checkpoints, figures

Rerunning the same experiment+timestamp detects `DONE` and prints `[SKIP]`
(exit 0). Never delete result dirs — rename to `*_DISABLED` to retire.

## Running the gates

    .venv\Scripts\python.exe -m pytest tests -q

## Smoke

    .venv\Scripts\python.exe scripts\smoke.py --regime MATE-4v8-9-v0 --steps 300
