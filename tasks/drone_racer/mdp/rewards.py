# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute sum of squared errors
    return torch.sum(torch.square(asset.data.root_pos_w - target_pos_tensor), dim=1)


def pos_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset pos from its target pos using L2 squared kernel."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    distance = torch.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return 1 - torch.tanh(distance / std)


def progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asymmetric: bool = False,
) -> torch.Tensor:
    """Progress toward target gate (prev_distance - current_distance).

    asymmetric=False (default): signed — PPO needs negative gradient on retreat so the
    policy commits to moving toward the gate. Matches upstream PPO behavior.
    asymmetric=True: clamp to >=0 (forward-only). Useful for under-trained policies
    that need a one-sided signal to escape hover.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    target_pos = env.command_manager.get_term(command_name).command[:, :3]
    previous_pos = env.command_manager.get_term(command_name).previous_pos
    current_pos = asset.data.root_pos_w

    prev_distance = torch.norm(previous_pos - target_pos, dim=1)
    current_distance = torch.norm(current_pos - target_pos, dim=1)

    progress = prev_distance - current_distance
    if asymmetric:
        progress = progress.clamp(min=0.0)
    return progress


def gate_passed(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    penalize_miss: bool = True,
) -> torch.Tensor:
    """Reward for passing the current target gate, computed INLINE.

    penalize_miss=True (default): +1 pass / -1 plane-crossing-while-off-center. Upstream
    PPO behavior — discourages clipping the gate frame.
    penalize_miss=False: +1 pass only, no penalty for misses. Useful when the policy is
    still learning to even reach the gate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    cur_pos = asset.data.root_pos_w
    prev_pos = cmd.prev_robot_pos_w
    gate_pose = cmd.next_gate_w
    gate_pos = gate_pose[:, :3]
    gate_quat = gate_pose[:, 3:7]
    half_size = cmd.gate_size / 2.0

    x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=cur_pos.device).expand_as(cur_pos)
    gate_normal = math_utils.quat_apply(gate_quat, x_axis)
    rel_old = prev_pos - gate_pos
    rel_new = cur_pos - gate_pos
    proj_old = (rel_old * gate_normal).sum(dim=-1)
    proj_new = (rel_new * gate_normal).sum(dim=-1)
    crossing = (proj_old < 0) & (proj_new > 0)

    abs_diff = torch.abs(cur_pos - gate_pos)
    in_bbox = torch.all(abs_diff < half_size, dim=1)
    out_bbox = torch.any(abs_diff > half_size, dim=1)

    passed = crossing & in_bbox
    if penalize_miss:
        missed = crossing & out_bbox & ~passed
        return passed.float() - missed.float()
    return passed.float()


def lookat_next_gate(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for looking at the next gate."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    drone_pos = asset.data.root_pos_w
    drone_att = asset.data.root_quat_w
    next_gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    vec_to_gate = next_gate_pos - drone_pos
    vec_to_gate = math_utils.normalize(vec_to_gate)

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    drone_x_axis = math_utils.quat_apply(drone_att, x_axis)
    drone_x_axis = math_utils.normalize(drone_x_axis)

    dot = (drone_x_axis * vec_to_gate).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    return torch.exp(-angle / std)


def gate_visibility(
    env: ManagerBasedRLEnv,
    fov_deg: float = 90.0,
    tilt_deg: float = 0.0,
    cam_offset: tuple[float, float, float] = (0.14, 0.0, 0.05),
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """FOV-gated gate-visibility reward in [0, 1] (perception-aware shaping).

    Port of Hs293Go/isaac-drone-racing perception.visibility_reward, adapted from
    NED/FRD to this repo's ENU/FLU torch state. 1 when the target gate sits on the FPV
    optical axis, ramps linearly to 0 at the FOV edge, 0 once the gate leaves the FOV:

        r = relu(cos_angle - cos(fov/2)) / (1 - cos(fov/2))

    cos_angle is between the camera optical axis and the line-of-sight from the camera
    to the gate. The optical axis is the body-forward x-axis pitched up by tilt_deg
    (FLU: [cos t, 0, sin t]); cam_offset is the camera mount in the body frame (defaults
    to the tiled_camera offset). Use a small positive weight (see perception_aware tuning:
    ~0.03 relative to a ~0.05-0.1 per-step progress reward).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    drone_pos = asset.data.root_pos_w
    drone_quat = asset.data.root_quat_w
    gate_pos = env.command_manager.get_term(command_name).command[:, :3]

    # camera world position = drone pos + body-frame mount offset rotated to world
    offset_b = torch.tensor(cam_offset, device=asset.device).expand(env.num_envs, 3)
    cam_pos = drone_pos + math_utils.quat_apply(drone_quat, offset_b)

    los = math_utils.normalize(gate_pos - cam_pos)

    t = torch.deg2rad(torch.tensor(tilt_deg, device=asset.device))
    axis_b = torch.stack([torch.cos(t), torch.zeros_like(t), torch.sin(t)]).expand(env.num_envs, 3)
    optical_axis_w = math_utils.quat_apply(drone_quat, axis_b)

    cos_angle = (optical_axis_w * los).sum(dim=1).clamp(-1.0, 1.0)
    cos_half = torch.cos(torch.deg2rad(torch.tensor(fov_deg, device=asset.device)) / 2.0)
    return torch.clamp(cos_angle - cos_half, min=0.0) / (1.0 - cos_half).clamp(min=1e-9)


def backpedal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Tail-first penalty in [0, 1] (perception-aware shaping). USE NEGATIVE WEIGHT.

    Port of Hs293Go/isaac-drone-racing perception.backpedal. 0 when the body nose leads
    the velocity (flying forward), up to 1 when fully tail-first:

        penalty = relu(-cos(velocity, nose))

    nose = body x-axis in world (FLU forward), velocity = world linear velocity. Returns
    a positive penalty; set a negative cfg weight (their tuning: ~0.01; 0.02+ crashes
    tight crossings).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_w
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).expand(env.num_envs, 3)
    nose = math_utils.normalize(math_utils.quat_apply(asset.data.root_quat_w, x_axis))

    speed = torch.norm(vel, dim=1)
    cos = (vel * nose).sum(dim=1) / speed.clamp(min=1e-9)
    return torch.clamp(-cos, min=0.0)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


def _stack_drone_positions(env: ManagerBasedRLEnv, num_drones: int) -> torch.Tensor:
    """Helper: (num_envs, num_drones, 3) stack of root_pos_w across drone_0..drone_{N-1}."""
    return torch.stack(
        [env.scene[f"drone_{i}"].data.root_pos_w for i in range(num_drones)], dim=1
    )


def drone_drone_collision_penalty(
    env: ManagerBasedRLEnv,
    num_drones: int,
    safety_distance: float = 0.3,
) -> torch.Tensor:
    """+1 per env when ANY pair of drones is within safety_distance (else 0). Use a
    negative weight in the cfg (e.g., -10) to penalize swarm proximity. Pairwise
    upper-triangle check; returns (num_envs,) float."""
    positions = _stack_drone_positions(env, num_drones)  # (E, N, 3)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (E, N, N, 3)
    dist = diff.norm(dim=-1)  # (E, N, N)
    mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    return ((dist < safety_distance) & mask).any(dim=(1, 2)).float()


def ang_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base angular velocity using L1 norm (paper Eq. body-rate term)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)


def opponent_proximity_penalty(
    env: ManagerBasedRLEnv,
    drone_idx: int,
    num_drones: int,
    d_col: float = 0.10,
    lambda5: float = 2.0,
) -> torch.Tensor:
    """Velocity-weighted exponential proximity penalty (paper, Eq. 3 r_prox).
    Active only when nearest opponent distance < 2*d_col. Use negative weight.

        r_prox = ||v|| * (1 + exp(-lambda5 * d_tilde))
        d_tilde = (d_opp - d_col) / d_col
    """
    positions = _stack_drone_positions(env, num_drones)  # (E, N, 3)
    ego_pos = positions[:, drone_idx]  # (E, 3)
    diff = positions - ego_pos.unsqueeze(1)  # (E, N, 3)
    dist = diff.norm(dim=-1)  # (E, N)
    dist[:, drone_idx] = float("inf")  # exclude self
    d_opp, _ = dist.min(dim=1)  # (E,)
    active = d_opp < 2.0 * d_col
    d_tilde = (d_opp - d_col) / d_col
    v_norm = env.scene[f"drone_{drone_idx}"].data.root_lin_vel_w.norm(dim=-1)
    return torch.where(active, v_norm * (1.0 + torch.exp(-lambda5 * d_tilde)), torch.zeros_like(v_norm))


def race_ranking(
    env: ManagerBasedRLEnv,
    drone_idx: int,
    num_drones: int,
    command_prefix: str = "target_",
) -> torch.Tensor:
    """Per-drone ranking reward (paper, Eq. 4 r_rank).

        r_rank = (N - (rank - 1)) / N
        rank=1 (leader) → 1.0, rank=N → 1/N.

    Rank is by primary key = gates passed (next_gate_idx), tiebreak = -dist-to-next-gate.
    """
    next_gate = torch.stack(
        [env.command_manager.get_term(f"{command_prefix}{i}").next_gate_idx for i in range(num_drones)],
        dim=1,
    )  # (E, N) int
    # tiebreak score: -distance to own next gate (closer = higher score)
    dists = []
    for i in range(num_drones):
        cmd = env.command_manager.get_term(f"{command_prefix}{i}")
        ego_pos = env.scene[f"drone_{i}"].data.root_pos_w
        dists.append((cmd.command[:, :3] - ego_pos).norm(dim=-1))
    dist_to_next = torch.stack(dists, dim=1)  # (E, N)
    # Composite score: gates_passed * 100 - dist  (large enough to dominate)
    score = next_gate.float() * 100.0 - dist_to_next
    # Rank of drone_idx = number of drones with strictly higher score + 1
    my_score = score[:, drone_idx:drone_idx + 1]  # (E, 1)
    rank = (score > my_score).sum(dim=1).float() + 1.0  # (E,)
    return (num_drones - (rank - 1.0)) / num_drones
