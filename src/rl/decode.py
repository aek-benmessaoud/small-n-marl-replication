"""decode.py — Observation-based decoding of MATE camera observations.

Camera observation (single agent) layout, per mate.constants:
    preserved_data | self_state (9) | targets (5 each, mask last) |
    obstacles (4 each) | other cameras (7 each)

Each target block: [x, y, sight_range, is_loaded, mask]; mask == 1.0 means
"observed by this camera this step". This decoder uses ONLY the observation,
so it is fair for baselines vs MAPPO.
"""

import numpy as np

from mate import constants


def decode_camera_obs(obs_i, num_cameras, num_targets, num_obstacles):
    """Decode one camera's observation vector -> dict of numpy arrays."""
    slices = constants.camera_observation_slices_of(
        num_cameras, num_targets, num_obstacles
    )
    self_state = obs_i[slices["self_state"]]
    target_block = obs_i[slices["opponent_states_with_mask"]].reshape(
        num_targets, constants.TARGET_STATE_DIM_PUBLIC + 1
    )
    return {
        "self_location": np.asarray(self_state[:2], dtype=np.float64),
        "targets": target_block[:, :2],                       # (n_tar, 2)
        "target_masks": target_block[:, -1] > 0.5,            # (n_tar,) bool visible
    }


def decode_team_obs(obs, num_cameras, num_targets, num_obstacles):
    """Decode the full team observation (n_cam, obs_dim) -> list of dicts."""
    return [decode_camera_obs(obs[i], num_cameras, num_targets, num_obstacles)
            for i in range(num_cameras)]
