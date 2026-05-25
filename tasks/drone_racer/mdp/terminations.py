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


# Per-body baseline cache: contact-sensor name -> tensor (num_envs, num_bodies) of resting forces.
_CONTACT_BASELINE: dict[str, torch.Tensor] = {}


def crash_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("collision_sensor"),
) -> torch.Tensor:
    """Crash detector that subtracts a per-body static phantom force baseline.

    Isaac Sim 5.1 reports a constant ~76 N phantom force on the drone's `body` even
    at rest (likely a gravity/applied-wrench accounting quirk in the contact pipeline).
    Stock `mdp.illegal_contact` either fires every step (small threshold) or misses
    real impacts (huge threshold). This function caches the steady-state force per
    body once after init and triggers on (current - baseline) > threshold.

    threshold (N): additional contact force above baseline. 5 N reliably catches
    light bumps; raise to 20 N if low-speed ground contact during cruise should
    be ignored.
    """
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w  # (N, B, 3)
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    f_mag = torch.norm(forces, dim=-1)  # (N, B)

    key = sensor_cfg.name
    baseline = _CONTACT_BASELINE.get(key)
    if baseline is None or baseline.shape != f_mag.shape:
        # Snapshot first frame as baseline. Drone is at rest at episode start so
        # whatever is reported here is the steady-state phantom.
        _CONTACT_BASELINE[key] = f_mag.clone().detach()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    delta = f_mag - baseline
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
