# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Drone-Racer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

# Ground-truth-only variants (no camera, faster training)
gym.register(
    id="Isaac-Drone-Racer-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_NoCam",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg_nocam.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_NoCam_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg_nocam.yaml",
    },
)

# CTBR action variants (collective thrust + body rates instead of motor-omega)
gym.register(
    id="Isaac-Drone-Racer-CTBR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_CTBR",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-CTBR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_CTBR_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-NoCam-CTBR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_NoCam_CTBR",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg_nocam_ctbr.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Racer-NoCam-CTBR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_racer_env_cfg:DroneRacerEnvCfg_NoCam_CTBR_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg_nocam_ctbr.yaml",
    },
)

# Swarm variants — N drones in one env, single PPO over concatenated state.
# Default N=4 (set in cfg class); override via train.py / play.py --num_drones.
for _swarm_id, _cfg_name, _yaml in [
    ("Isaac-Drone-Racer-Swarm-v0", "DroneRacerSwarmEnvCfg", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-Play-v0", "DroneRacerSwarmEnvCfg_PLAY", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-v0", "DroneRacerSwarmEnvCfg_NoCam", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-Play-v0", "DroneRacerSwarmEnvCfg_NoCam_PLAY", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-CTBR-v0", "DroneRacerSwarmEnvCfg_CTBR", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-CTBR-Play-v0", "DroneRacerSwarmEnvCfg_CTBR_PLAY", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-CTBR-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR", "skrl_cfg_swarm.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-CTBR-Play-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR_PLAY", "skrl_cfg_swarm.yaml"),
    # Lite variants — RTX 3060 / 6 GB-friendly preset (smaller MLP, smaller rollout).
    # Pair with --num_envs 32-128 and --num_drones 2-3.
    ("Isaac-Drone-Racer-Swarm-NoCam-CTBR-Lite-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR", "skrl_cfg_swarm_lite.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-CTBR-Lite-Play-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR_PLAY", "skrl_cfg_swarm_lite.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-Lite-v0", "DroneRacerSwarmEnvCfg_NoCam", "skrl_cfg_swarm_lite.yaml"),
    ("Isaac-Drone-Racer-Swarm-NoCam-Lite-Play-v0", "DroneRacerSwarmEnvCfg_NoCam_PLAY", "skrl_cfg_swarm_lite.yaml"),
    # Shared-policy variant — single MLP across all drones via SharedSwarmEnvWrapper
    # (per-drone 20-dim obs / 4-dim action, equivalent to IPPO with shared parameters).
    # Use with `--shared_policy` flag on train.py / play.py.
    ("Isaac-Drone-Racer-Swarm-Shared-NoCam-CTBR-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR", "skrl_cfg_swarm_shared.yaml"),
    ("Isaac-Drone-Racer-Swarm-Shared-NoCam-CTBR-Play-v0", "DroneRacerSwarmEnvCfg_NoCam_CTBR_PLAY", "skrl_cfg_swarm_shared.yaml"),
]:
    gym.register(
        id=_swarm_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.swarm_env_cfg:{_cfg_name}",
            "skrl_cfg_entry_point": f"{agents.__name__}:{_yaml}",
        },
    )
