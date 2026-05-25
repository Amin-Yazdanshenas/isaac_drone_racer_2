# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def crash_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("collision_sensor"),
) -> torch.Tensor:
    """Stub kept for backward compatibility. Sim 5.1 ContactSensor reports a
    static phantom force on the drone body that pollutes both net_forces_w
    and force_matrix_w (see ground_crash for the working approach).

    Always returns False so this termination is effectively disabled.
    """
    del threshold, sensor_cfg  # unused
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def ground_crash(
    env: ManagerBasedRLEnv,
    z_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the drone body sinks below z_threshold (ground crash).

    Sim 5.1's ContactSensor reports a constant phantom force on the drone body
    in both net_forces_w and force_matrix_w, making contact-based collision
    detection unusable. Falling back to a simple altitude check: any crash
    eventually drives the body to ground level. Combined with flyaway (lateral)
    and time_out (temporal), this covers all real failure modes.

    z_threshold (m): below this the drone is considered crashed. 0.1 sits well
    below the default reset spawn band (z ∈ [0.5, 1.5]) and any normal hover.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < z_threshold


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's is too far away from the target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance
