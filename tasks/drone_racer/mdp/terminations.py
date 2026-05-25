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
    """Crash detector using ContactSensor.data.force_matrix_w.

    Requires the ContactSensor to be configured with filter_prim_paths_expr.
    force_matrix_w contains only the forces against those filtered prims
    (ground, gates) so it excludes the internal articulation phantoms
    (gravity/applied-wrench/gyroscopic) that pollute net_forces_w.

    threshold (N): force magnitude on a filtered contact pair above which
    we declare a crash. 1 N is plenty since the signal is clean.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fm = cs.data.force_matrix_w  # (N, B, F, 3)
    if fm is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # max force magnitude across (bodies, filters) per env
    f_mag = torch.norm(fm, dim=-1)  # (N, B, F)
    return f_mag.amax(dim=(1, 2)) > threshold


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
