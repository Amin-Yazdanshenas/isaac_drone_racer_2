# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared-policy wrapper for the swarm racing env.

Restructures a (num_envs, num_drones * 20) concat-state swarm env into a
(num_envs * num_drones, 20) per-drone view so a single MLP can be trained
across all drones with PPO. Each drone processes its own ego state through
the same network weights — true shared policy, equivalent to IPPO with
share_parameters=True.

Reward attribution (first iteration): team reward is divided equally across
drones — simple but introduces a credit assignment bias (free-rider). A
later iteration will compute per-drone reward directly from the reward
manager term breakdown.

Done: broadcast scalar env-level done to all drone slots. An episode end on
any drone's termination cascades to every drone's rollout.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch


class SharedSwarmEnvWrapper(gym.Wrapper):
    """Wraps a swarm env so each drone is exposed as one of (num_envs * num_drones)
    independent transitions."""

    def __init__(self, env: gym.Env, num_drones: int, per_drone_obs_dim: int = 20,
                 per_drone_act_dim: int = 4):
        super().__init__(env)
        self.num_drones = int(num_drones)
        self.per_drone_obs_dim = int(per_drone_obs_dim)
        self.per_drone_act_dim = int(per_drone_act_dim)

        underlying_envs = getattr(env, "num_envs", None) or getattr(env.unwrapped, "num_envs")
        self._underlying_num_envs = int(underlying_envs)
        self.num_envs = self._underlying_num_envs * self.num_drones

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.per_drone_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.per_drone_act_dim,), dtype=np.float32,
        )
        self.state_space = None
        # skrl reads .single_observation_space / .single_action_space on vectorised envs.
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def _split_obs(self, obs):
        """(num_envs, num_drones * per_drone_obs_dim) -> (num_envs * num_drones, per_drone_obs_dim)."""
        if isinstance(obs, dict):
            policy = obs["policy"]
        else:
            policy = obs
        return policy.view(self._underlying_num_envs, self.num_drones, self.per_drone_obs_dim) \
            .reshape(self.num_envs, self.per_drone_obs_dim)

    def _join_action(self, action):
        """(num_envs * num_drones, per_drone_act_dim) -> (num_envs, num_drones * per_drone_act_dim)."""
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        return action.view(self._underlying_num_envs, self.num_drones, self.per_drone_act_dim) \
            .reshape(self._underlying_num_envs, self.num_drones * self.per_drone_act_dim)

    def _broadcast_scalar(self, x):
        """(num_envs,) -> (num_envs * num_drones,) via repeat_interleave."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        return x.unsqueeze(1).expand(-1, self.num_drones).reshape(-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._split_obs(obs), info

    def step(self, action):
        joined = self._join_action(action)
        obs, rew, term, trunc, info = self.env.step(joined)
        rew_per_drone = self._broadcast_scalar(rew) / float(self.num_drones)
        term_per_drone = self._broadcast_scalar(term)
        trunc_per_drone = self._broadcast_scalar(trunc)
        return self._split_obs(obs), rew_per_drone, term_per_drone, trunc_per_drone, info
