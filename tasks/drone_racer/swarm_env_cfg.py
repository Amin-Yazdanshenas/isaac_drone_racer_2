# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swarm racing env cfg: N drones share one env. V1 uses a single PPO over
the concatenated state of all drones — not a true shared policy (deferred
to V2). Reward sums across drones; episode terminates if ANY drone crashes
or any two drones collide.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .swarm_utils import (
    make_collision_sensor,
    make_drone_articulation,
    make_imu_sensor,
    make_tiled_camera,
)
from .track_generator import generate_track


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


@configclass
class DroneRacerSwarmSceneCfg(InteractiveSceneCfg):
    """Scene with N drones + per-drone sensors. Drones / sensors are NOT declared
    as fields here — they're injected at runtime by populate_swarm_scene() so
    InteractiveScene's asset discovery doesn't trip on non-asset attrs."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    track: RigidObjectCollectionCfg = generate_track(
        track_config={
            "1": {"pos": (0.0, 0.0, 1.0), "yaw": 0.0},
            "2": {"pos": (10.0, 5.0, 0.0), "yaw": 0.0},
            "3": {"pos": (10.0, -5.0, 0.0), "yaw": (5 / 4) * torch.pi},
            "4": {"pos": (-5.0, -5.0, 2.5), "yaw": torch.pi},
            "5": {"pos": (-5.0, -5.0, 0.0), "yaw": 0.0},
            "6": {"pos": (5.0, 0.0, 0.0), "yaw": (1 / 2) * torch.pi},
            "7": {"pos": (0.0, 5.0, 0.0), "yaw": torch.pi},
        }
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def populate_swarm_scene(scene_cfg: DroneRacerSwarmSceneCfg, num_drones: int, include_camera: bool) -> None:
    """Inject N drone articulations + sensors onto an already-built scene cfg."""
    for i in range(num_drones):
        setattr(scene_cfg, f"drone_{i}", make_drone_articulation(i))
        setattr(scene_cfg, f"collision_sensor_{i}", make_collision_sensor(i))
        setattr(scene_cfg, f"imu_{i}", make_imu_sensor(i))
        if include_camera:
            setattr(scene_cfg, f"camera_{i}", make_tiled_camera(i))


# --------------------------------------------------------------------------
# Module-level configclass shells (must be top-level so they pickle).
# --------------------------------------------------------------------------


@configclass
class _SwarmActionsCfg:
    pass


@configclass
class _SwarmCommandsCfg:
    pass


@configclass
class _SwarmEventCfg:
    pass


@configclass
class _SwarmRewardsCfg:
    pass


@configclass
class _SwarmTerminationsCfg:
    pass


@configclass
class _SwarmPolicyCfg(ObsGroup):
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class _SwarmCriticCfg(ObsGroup):
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class _SwarmObsCfg:
    policy: _SwarmPolicyCfg = None
    critic: _SwarmCriticCfg | None = None


# --------------------------------------------------------------------------
# Action / command / event factories
# --------------------------------------------------------------------------


def _build_actions(num_drones: int, use_ctbr: bool) -> _SwarmActionsCfg:
    """Returns a configclass instance with N action terms named control_action_i."""

    cfg = _SwarmActionsCfg()
    cls = mdp.CTBRActionCfg if use_ctbr else mdp.ControlActionCfg
    for i in range(num_drones):
        if use_ctbr:
            setattr(cfg, f"control_action_{i}", cls(asset_name=f"drone_{i}"))
        else:
            setattr(cfg, f"control_action_{i}", cls(asset_name=f"drone_{i}", use_motor_model=False))
    return cfg


def _build_commands(num_drones: int) -> _SwarmCommandsCfg:
    cfg = _SwarmCommandsCfg()
    for i in range(num_drones):
        setattr(
            cfg,
            f"target_{i}",
            mdp.GateTargetingCommandCfg(
                asset_name=f"drone_{i}",
                track_name="track",
                randomise_start=None,
                record_fpv=False,
                resampling_time_range=(1e9, 1e9),
                debug_vis=False,
                spawn_lerp_alpha=0.0,
                spawn_forward_offset=1.0,
                spawn_forward_velocity=0.0,
            ),
        )
    return cfg


def _build_events(num_drones: int) -> _SwarmEventCfg:
    """Per-drone reset_base events spaced laterally so drones don't overlap."""

    cfg = _SwarmEventCfg()
    # Lateral lane per drone. Pose range remains tight so the spawn box covers all drones.
    for i in range(num_drones):
        y_offset = (i - (num_drones - 1) / 2.0) * 0.6  # spread along Y, centered
        setattr(
            cfg,
            f"reset_base_{i}",
            EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-3.5, -1.5),
                        "y": (y_offset - 0.15, y_offset + 0.15),
                        "z": (0.5, 1.5),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                    "asset_cfg": SceneEntityCfg(f"drone_{i}"),
                },
            ),
        )
    # Per-drone interval push (apply_external_force_torque defaults to asset "robot",
    # which doesn't exist in swarm scenes — supply each drone explicitly).
    for i in range(num_drones):
        setattr(
            cfg,
            f"push_robot_{i}",
            EventTerm(
                func=mdp.apply_external_force_torque,
                mode="interval",
                interval_range_s=(0.0, 0.2),
                params={
                    "force_range": (-0.1, 0.1),
                    "torque_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg(f"drone_{i}"),
                },
            ),
        )
    return cfg


# --------------------------------------------------------------------------
# Observations / rewards / terminations
# --------------------------------------------------------------------------


def _build_observations(num_drones: int, include_camera: bool) -> _SwarmObsCfg:
    """Policy = full-state of all drones (centralized — V1). Critic = same
    plus the env-level state. NoCam variant skips the FPV term; with-camera
    variant flattens each drone's FPV + IMU into the policy obs."""

    policy = _SwarmPolicyCfg()
    critic = _SwarmCriticCfg()

    for i in range(num_drones):
        if include_camera:
            setattr(policy, f"image_{i}", ObsTerm(
                func=mdp.gate_mask,
                params={"sensor_cfg": SceneEntityCfg(f"camera_{i}"), "command_name": f"target_{i}"},
            ))
            setattr(policy, f"imu_ang_{i}", ObsTerm(
                func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg(f"imu_{i}")}
            ))
            setattr(policy, f"imu_att_{i}", ObsTerm(
                func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg(f"imu_{i}")}
            ))
        else:
            setattr(policy, f"pos_{i}", ObsTerm(
                func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
            ))
            setattr(policy, f"att_{i}", ObsTerm(
                func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
            ))
            setattr(policy, f"lin_vel_{i}", ObsTerm(
                func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
            ))
            setattr(policy, f"ang_vel_{i}", ObsTerm(
                func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
            ))
            setattr(policy, f"target_pos_b_{i}", ObsTerm(
                func=mdp.target_pos_b,
                params={"command_name": f"target_{i}", "asset_cfg": SceneEntityCfg(f"drone_{i}")},
            ))
            setattr(policy, f"action_{i}", ObsTerm(
                func=mdp.last_action, params={"action_name": f"control_action_{i}"}
            ))

        # Critic always gets full GT state for every drone.
        setattr(critic, f"pos_{i}", ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
        ))
        setattr(critic, f"att_{i}", ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
        ))
        setattr(critic, f"lin_vel_{i}", ObsTerm(
            func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
        ))
        setattr(critic, f"ang_vel_{i}", ObsTerm(
            func=mdp.root_ang_vel_b, params={"asset_cfg": SceneEntityCfg(f"drone_{i}")}
        ))
        setattr(critic, f"target_pos_b_{i}", ObsTerm(
            func=mdp.target_pos_b,
            params={"command_name": f"target_{i}", "asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        setattr(critic, f"action_{i}", ObsTerm(
            func=mdp.last_action, params={"action_name": f"control_action_{i}"}
        ))

    obs = _SwarmObsCfg()
    obs.policy = policy
    if include_camera:
        obs.critic = critic
    else:
        obs.critic = None
    return obs


def _build_rewards(num_drones: int) -> _SwarmRewardsCfg:
    """Paper-style reward (Geles et al 2024): progress + ranking
    - body_rate(L1) - proximity(velocity-weighted exp). Plus kept project-specific:
    gate_passed (sparse +400 on traversal), lookat_next_gate (small heading aim),
    terminating (small crash penalty)."""
    cfg = _SwarmRewardsCfg()
    for i in range(num_drones):
        setattr(cfg, f"terminating_{i}", RewTerm(func=mdp.is_terminated, weight=-100.0 / num_drones))
        # Body-rate smoothness — paper Eq. 2 (L1).
        setattr(cfg, f"ang_vel_{i}", RewTerm(
            func=mdp.ang_vel_l1, weight=-0.01,
            params={"asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        setattr(cfg, f"progress_{i}", RewTerm(
            func=mdp.progress, weight=20.0,
            params={"command_name": f"target_{i}", "asymmetric": False,
                    "asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        setattr(cfg, f"gate_passed_{i}", RewTerm(
            func=mdp.gate_passed, weight=1000.0,
            params={"command_name": f"target_{i}", "penalize_miss": False,
                    "asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        setattr(cfg, f"lookat_{i}", RewTerm(
            func=mdp.lookat_next_gate, weight=0.1,
            params={"command_name": f"target_{i}", "std": 0.5,
                    "asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        # Paper Eq. 4 ranking reward: leader=1, last=1/N. Weight dropped
        # 2.0 -> 0.5 — rank was dominating signal, policy farmed survival
        # without racing (gate_passed dropped 0.024 -> 0.006 between runs).
        setattr(cfg, f"rank_{i}", RewTerm(
            func=mdp.race_ranking, weight=0.5,
            params={"drone_idx": i, "num_drones": num_drones},
        ))
        # Paper Eq. 3 velocity-weighted exponential proximity penalty.
        # Weight starts at 0 (curriculum bumps it after step_threshold).
        setattr(cfg, f"proximity_{i}", RewTerm(
            func=mdp.opponent_proximity_penalty, weight=0.0,
            params={"drone_idx": i, "num_drones": num_drones,
                    "d_col": 0.10, "lambda5": 2.0},
        ))
    return cfg


def _build_terminations(num_drones: int) -> _SwarmTerminationsCfg:
    cfg = _SwarmTerminationsCfg()
    cfg.time_out = DoneTerm(func=mdp.time_out, time_out=True)
    for i in range(num_drones):
        setattr(cfg, f"flyaway_{i}", DoneTerm(
            func=mdp.flyaway,
            params={"command_name": f"target_{i}", "distance": 20.0,
                    "asset_cfg": SceneEntityCfg(f"drone_{i}")},
        ))
        setattr(cfg, f"collision_{i}", DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg(f"collision_sensor_{i}"), "threshold": 10.0},
        ))
        setattr(cfg, f"gate_collision_{i}", DoneTerm(
            func=mdp.gate_collision, params={"command_name": f"target_{i}"},
        ))
    # Drone-drone collision DoneTerm removed — paper-style training treats
    # inter-agent contact as a soft (proximity) penalty, with curriculum
    # ramping its weight from 0 to penalty value. Lets agent learn to recover
    # from minor contact instead of dying every time.
    return cfg


@configclass
class _SwarmCurriculumCfg:
    pass


def _build_curriculum(num_drones: int) -> _SwarmCurriculumCfg:
    """Paper curriculum: opponents start non-interactive (proximity penalty=0),
    turn on after the policy can pass a few gates. Step threshold tuned for
    ~500k env-steps at 4096 envs (≈120 PPO updates)."""
    cfg = _SwarmCurriculumCfg()
    step_on = 500_000
    for i in range(num_drones):
        setattr(cfg, f"proximity_on_{i}", CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": f"proximity_{i}", "weight": -1.0, "num_steps": step_on},
        ))
    return cfg


# --------------------------------------------------------------------------
# Env cfgs
# --------------------------------------------------------------------------


@configclass
class _SwarmEnvCfgBase(ManagerBasedRLEnvCfg):
    """Base swarm env. Subclasses set num_drones, include_camera, use_ctbr."""

    # Tunables (override in subclass).
    num_drones: int = 4
    include_camera: bool = True
    use_ctbr: bool = False

    scene: DroneRacerSwarmSceneCfg = DroneRacerSwarmSceneCfg(num_envs=512, env_spacing=0.0)
    # MDP slots filled in __post_init__ via builder fns.
    observations: object = None
    actions: object = None
    commands: object = None
    events: object = None
    rewards: object = None
    terminations: object = None
    curriculum: object = None

    def __post_init__(self) -> None:
        populate_swarm_scene(self.scene, self.num_drones, self.include_camera)

        # PhysX buffer bump: N drones × M envs blows past the default contact-patch
        # buffer at 4096 envs × 4 drones (saw "Patch buffer overflow ... at least 401408").
        # 2**20 ~ 1.05M patches gives headroom up to ~8 drones × 4096 envs.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_max_rigid_contact_count = 2**24

        self.actions = _build_actions(self.num_drones, self.use_ctbr)
        self.commands = _build_commands(self.num_drones)
        self.events = _build_events(self.num_drones)
        self.rewards = _build_rewards(self.num_drones)
        self.terminations = _build_terminations(self.num_drones)
        self.observations = _build_observations(self.num_drones, self.include_camera)
        self.curriculum = _build_curriculum(self.num_drones)

        # Training behavior: respawn at random gate (matches single-drone train cfgs).
        for i in range(self.num_drones):
            setattr(self.events, f"reset_base_{i}", None)
            getattr(self.commands, f"target_{i}").randomise_start = True

        self.decimation = 4
        self.episode_length_s = 20
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerSwarmEnvCfg(_SwarmEnvCfgBase):
    """Camera + IMU swarm, motor-omega action."""

    num_drones: int = 4
    include_camera: bool = True
    use_ctbr: bool = False
    scene: DroneRacerSwarmSceneCfg = DroneRacerSwarmSceneCfg(num_envs=64, env_spacing=0.0)


@configclass
class DroneRacerSwarmEnvCfg_PLAY(DroneRacerSwarmEnvCfg):
    scene: DroneRacerSwarmSceneCfg = DroneRacerSwarmSceneCfg(num_envs=1, env_spacing=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # PLAY: corner spawn (reset_base events left active), no push, no random gate.
        self.events = _build_events(self.num_drones)
        for i in range(self.num_drones):
            getattr(self.commands, f"target_{i}").randomise_start = None
            setattr(self.events, f"push_robot_{i}", None)


@configclass
class DroneRacerSwarmEnvCfg_NoCam(_SwarmEnvCfgBase):
    """GT-state swarm, motor-omega action, camera disabled."""

    num_drones: int = 4
    include_camera: bool = False
    use_ctbr: bool = False
    scene: DroneRacerSwarmSceneCfg = DroneRacerSwarmSceneCfg(num_envs=512, env_spacing=0.0)


@configclass
class DroneRacerSwarmEnvCfg_NoCam_PLAY(DroneRacerSwarmEnvCfg_NoCam):
    scene: DroneRacerSwarmSceneCfg = DroneRacerSwarmSceneCfg(num_envs=1, env_spacing=0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events = _build_events(self.num_drones)
        for i in range(self.num_drones):
            getattr(self.commands, f"target_{i}").randomise_start = None
            setattr(self.events, f"push_robot_{i}", None)


@configclass
class DroneRacerSwarmEnvCfg_CTBR(DroneRacerSwarmEnvCfg):
    use_ctbr: bool = True


@configclass
class DroneRacerSwarmEnvCfg_CTBR_PLAY(DroneRacerSwarmEnvCfg_PLAY):
    use_ctbr: bool = True


@configclass
class DroneRacerSwarmEnvCfg_NoCam_CTBR(DroneRacerSwarmEnvCfg_NoCam):
    use_ctbr: bool = True


@configclass
class DroneRacerSwarmEnvCfg_NoCam_CTBR_PLAY(DroneRacerSwarmEnvCfg_NoCam_PLAY):
    use_ctbr: bool = True
