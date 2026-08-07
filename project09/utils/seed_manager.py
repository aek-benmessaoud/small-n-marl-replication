"""
utils/seed_manager.py — Deterministic, paired seed handling (Project08 port).

RULES (locked decisions, transposed to MATE):
  - Env seed for run index r:  env_seed = BASE_SEED + r * SEED_STRIDE.
    All methods share the SAME env seed per run index (paired at env level).
  - Per-agent policy seed: derived deterministically from env_seed + agent_id,
    so each camera has its own independent RNG stream.
  - In MATE the environment is seeded via env.seed(env_seed); cameras/targets
    inherit the env RNG stream through the seeding chain.
"""

from project09.config import BASE_SEED, SEED_STRIDE


def env_seed_for_run(run_index):
    return BASE_SEED + run_index * SEED_STRIDE


def policy_seed_for(env_seed, agent_id):
    """Deterministic per-agent policy RNG seed."""
    return env_seed * 1000 + agent_id + 1
