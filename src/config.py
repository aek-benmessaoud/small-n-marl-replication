"""
config.py — Single Source of Truth for Project09 (MARL localization-aware
coverage on MATE).

Domain: continuous moving targets. Each target accumulates independent angular
configurations (bearings > ANG_TOL_DEG apart, greedy clustering per target)
from the cameras that observe it. The Chao-U richness signal is transposed
from Project08's per-cell visit counts to per-target configuration counts
(F1/F2 = targets with exactly 1/2 configurations).

Reward convention (locked): r_intrinsic = lambda * (U_t - U_{t+1}) / U_max.
Positive when the remaining-work estimate U DECREASES (more targets become
localizable). NOTE: the charter formula U_{t+1} - U_t is inverted; this config
holds the corrected sign.

All run_*.py, analysis and tests must import constants from here.
"""

import os

import numpy as np

# ==================== PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CONFIGS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "configs")

# ==================== EXPERIMENT RIGOR ====================
MAX_STEPS = 1000          # MATE default episode length (target horizon)
NUM_RUNS = 30
BASE_SEED = 0
SEED_STRIDE = 1000        # run r -> env_seed = BASE_SEED + r * SEED_STRIDE

# ==================== MATE REGIMES ====================
# Presets matching the charter regimes:
#   sparse     : MATE-4v2-9-v0  (4 cameras, 2 targets)
#   imbalanced : MATE-4v8-9-v0  (4 cameras, 8 targets) — MATE default
#   dense      : MATE-8v8-9-v0  (8 cameras, 8 targets)
REGIME_SPARSE = "MATE-4v2-9-v0"
REGIME_IMBALANCED = "MATE-4v8-9-v0"
REGIME_DENSE = "MATE-8v8-9-v0"
DEFAULT_REGIME = REGIME_IMBALANCED

# ==================== ANGULAR LOCALIZATION MODEL (transposed) ==============
# Two observations of a target are "independent configurations" when their
# bearing directions (camera -> target) are separated by more than ANG_TOL_DEG
# (circular). Greedy clustering per target, bounded by CLUSTER_CAP.
ANG_TOL_DEG = 15.0
CLUSTER_CAP = 8

# ==================== CHAO-U LOCKED CONFIG ====================
CHAO_DEFAULT_VARIANT = "bias_cap"
ADAPTIVE_K = 0.5
# U_max used to normalize the intrinsic reward: max possible remaining-work
# estimate = number of targets (every target under-determined).
U_MAX_TARGETS = None   # None -> derived from env.num_targets at runtime

# ==================== REWARD ====================
# r = coverage_reward + REWARD_LAMBDA * (U_t - U_{t+1}) / U_max
REWARD_LAMBDA = 0.5
REWARD_COVERAGE_COEFF = 1.0   # MATE raw camera reward (negative, sparse/dense)
REWARD_SPARSE = False         # 'dense' vs 'sparse' reward_type config override

# ==================== CRLB ORACLE (evaluation only) ========================
# Bearing-only CRLB per target: J = sum_k u_k u_k^T / (sigma^2 d_k^2),
# bound = sqrt(trace(J^-1)) in terrain units. sigma_ref is the nominal bearing
# precision used to scale the bound (fixed reference).
QUALITY_SIGMA_BEARING_DEG = 1.0
QUALITY_THRESHOLD = 100.0     # well-localized when bound <= threshold (units)
QUALITY_SAMPLE_K = 25
QUALITY_TARGET = 0.9

# ==================== MAPPO ====================
MAPPO_HIDDEN = [512, 256]
MAPPO_LR = 5e-4
MAPPO_GAMMA = 0.99
MAPPO_LAMBDA = 0.95
MAPPO_CLIP = 0.2
MAPPO_EPOCHS = 10
MAPPO_MINIBATCH = 128
MAPPO_FRAME_SKIP = 5          # act every N env steps (MATE example default)
MAPPO_CHECKPOINT_INTERVAL = 1000  # steps between interval checkpoints
MAPPO_MAX_TOTAL_STEPS = 2_000_000
MAPPO_HORIZON = 500           # rollout horizon (MATE example 'horizon')

# ==================== DEVOPS ====================
DONE_MARKER = "DONE"
# Config keys that MUST be present in every experiment config.json.
REQUIRED_CONFIG_KEYS = [
    "experiment", "phase", "regime", "seed", "num_runs",
    "reward_lambda", "chao_variant", "max_steps",
]
