"""
Project09 — MARL (MAPPO on MATE) with localization-aware coverage.

Transposes Project08's Chao-U richness signal to a continuous, moving-target
world (MATE). The localization unit is the target itself: each target
accumulates independent angular configurations from the cameras that observe
it. Reward = coverage + lambda * (U_t - U_{t+1}) / U_max (positive when U
decreases).
"""
