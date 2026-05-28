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


def gate_collision(
    env: ManagerBasedRLEnv,
    command_name: str = "target",
) -> torch.Tensor:
    """Terminate when the drone crosses the next gate plane outside its bbox
    (i.e. hits the gate frame).

    Replaces ContactSensor-based gate-hit detection. ContactSensor reads
    Sim 5.1 phantom forces from prop gyro that swamp real gate-impact forces,
    so collision thresholds must be set so high (>500 N) that gate frame hits
    no longer register. Geometric "missed-the-hole" detection in
    GateTargetingCommand is unaffected by physics-side noise.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.gate_missed


def drone_drone_collision(
    env: ManagerBasedRLEnv,
    num_drones: int,
    safety_distance: float = 0.25,
    non_terminal_prob: float = 0.10,
) -> torch.Tensor:
    """Episode-ending pairwise drone-drone proximity check. Returns (num_envs,) bool.
    Tighter than the reward-side safety_distance so the penalty has a band before
    termination fires.

    non_terminal_prob (paper, Geles et al 2024): probability that an otherwise
    terminating drone-drone contact is forgiven, letting the agent learn recovery
    from minor contact instead of being reset every time.
    """
    positions = torch.stack(
        [env.scene[f"drone_{i}"].data.root_pos_w for i in range(num_drones)], dim=1
    )  # (E, N, 3)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    dist = diff.norm(dim=-1)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    hit = ((dist < safety_distance) & mask).any(dim=(1, 2))
    if non_terminal_prob > 0.0:
        forgive = torch.rand(env.num_envs, device=hit.device) < non_terminal_prob
        hit = hit & ~forgive
    return hit
