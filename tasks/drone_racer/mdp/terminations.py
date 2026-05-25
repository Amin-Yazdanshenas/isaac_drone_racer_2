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
# Body phantom: gravity/applied-wrench accounting on the drone body.
# Prop phantoms: gyroscopic forces from the spinning motors (200 rad/s init).
_PHANTOM_FORCE_BY_BODY: dict[str, float] = {
    "body": 76.71774,
    "prop1": 21.52437,
    "prop2": 21.52437,
    "prop3": 21.52437,
    "prop4": 21.52437,
}


def crash_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
    min_speed: float = 0.3,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("collision_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Crash detector: contact force delta > threshold AND drone moving > min_speed.

    Isaac Sim 5.1's ContactSensor reports phantom forces on the drone bodies
    that drift after each crash (body baseline 76.7 -> 106 -> ... never clears).
    Velocity gating sidesteps the unreliable absolute force: real impacts only
    happen when the drone is moving. Phantom-only resets have vel=0 -> no fire.

    threshold (N): contact force above phantom baseline. Combined with the speed
    gate, can be small (5 N) without false-positive at rest.
    min_speed (m/s): below this the drone is considered at rest; collision is
    ignored. Hover noise is typically < 0.1 m/s.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w  # (N, B, 3)
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    f_mag = torch.norm(forces, dim=-1)  # (N, B)

    phantom = torch.tensor(
        [_PHANTOM_FORCE_BY_BODY.get(name, 0.0) for name in cs.body_names],
        device=f_mag.device,
        dtype=f_mag.dtype,
    )  # (B,)
    delta = f_mag - phantom

    body_ids = sensor_cfg.body_ids
    if body_ids is not None:
        delta = delta[:, body_ids]
    contact_hit = (delta > threshold).any(dim=1)

    asset: RigidObject = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    moving = speed > min_speed

    return contact_hit & moving


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
