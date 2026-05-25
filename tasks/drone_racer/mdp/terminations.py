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


# Per-body static phantom forces (N) reported by Sim 5.1 ContactSensor at rest.
# Indexed by body name. Anything not listed defaults to 0 (no subtraction).
_PHANTOM_FORCE_BY_BODY: dict[str, float] = {
    "body": 76.71774,
}


def crash_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("collision_sensor"),
) -> torch.Tensor:
    """Crash detector that subtracts a hardcoded per-body phantom force baseline.

    Isaac Sim 5.1's ContactSensor reports a constant ~76.7 N on the drone `body`
    even at rest (gravity / applied-wrench accounting quirk). Stock
    `mdp.illegal_contact` either fires every step (small threshold) or misses
    real impacts (huge threshold). Subtract the known phantom per body and fire
    on (current - phantom) > threshold.

    threshold (N): real contact force above phantom. 5 N reliably catches light
    bumps; raise to 20 N if low-speed ground contact during cruise should be
    ignored.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w  # (N, B, 3)
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    f_mag = torch.norm(forces, dim=-1)  # (N, B)

    # Build per-body phantom vector once on the right device.
    phantom = torch.tensor(
        [_PHANTOM_FORCE_BY_BODY.get(name, 0.0) for name in cs.body_names],
        device=f_mag.device,
        dtype=f_mag.dtype,
    )  # (B,)
    delta = f_mag - phantom  # broadcasts over envs

    body_ids = sensor_cfg.body_ids
    if body_ids is not None:
        delta = delta[:, body_ids]
    return (delta > threshold).any(dim=1)


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
