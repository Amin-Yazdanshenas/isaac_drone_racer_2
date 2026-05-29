# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swarm spawn diagnostic.

Spawns one env with 4 drones, runs N resets, prints:
  - per-drone spawn position (env 0)
  - per-drone target gate index
  - distance to target gate
  - distance to other drones

Use to verify whether one drone has systematic spawn advantage
(council R2 / R4 hypothesis from training plateau).

Run:
    conda activate isaacsim
    python3 scripts/utils/spawn_diag.py --num_drones 4 --num_resets 20
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_drones", type=int, default=4)
parser.add_argument("--num_resets", type=int, default=20)
parser.add_argument("--task", type=str, default="Isaac-Drone-Racer-Swarm-NoCam-CTBR-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tasks  # noqa: F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.num_drones = args.num_drones
env_cfg.__post_init__()
env = gym.make(args.task, cfg=env_cfg)

print(f"\n[spawn-diag] num_drones={args.num_drones}  num_resets={args.num_resets}")
print(f"[spawn-diag] resetting {args.num_resets} times, recording spawn state per drone\n")

records = {i: {"pos": [], "tgt_idx": [], "dist_to_tgt": []} for i in range(args.num_drones)}
pair_dists = []

for r in range(args.num_resets):
    obs, _ = env.reset()
    # Drone roots + targets
    pos_per_drone = []
    for i in range(args.num_drones):
        drone = env.unwrapped.scene[f"drone_{i}"]
        cmd = env.unwrapped.command_manager.get_term(f"target_{i}")
        p = drone.data.root_pos_w[0].cpu().numpy()
        tgt_idx = int(cmd.next_gate_idx[0].item())
        tgt_pos = cmd.command[0, :3].cpu().numpy()
        d = float(np.linalg.norm(p - tgt_pos))
        records[i]["pos"].append(p)
        records[i]["tgt_idx"].append(tgt_idx)
        records[i]["dist_to_tgt"].append(d)
        pos_per_drone.append(p)
    # pairwise
    for i in range(args.num_drones):
        for j in range(i + 1, args.num_drones):
            pair_dists.append((i, j, float(np.linalg.norm(pos_per_drone[i] - pos_per_drone[j]))))

print(f"{'drone':<7} {'spawn_x (mean±std)':<22} {'spawn_y (mean±std)':<22} {'spawn_z (mean±std)':<22} {'tgt_gates (unique)':<20} {'dist_to_tgt mean':<18}")
print("-" * 130)
for i in range(args.num_drones):
    pos = np.array(records[i]["pos"])
    tgts = records[i]["tgt_idx"]
    dists = records[i]["dist_to_tgt"]
    unique_tgts = sorted(set(tgts))
    print(f"{i:<7} "
          f"{pos[:, 0].mean():+8.3f} ± {pos[:, 0].std():4.3f}      "
          f"{pos[:, 1].mean():+8.3f} ± {pos[:, 1].std():4.3f}      "
          f"{pos[:, 2].mean():+8.3f} ± {pos[:, 2].std():4.3f}      "
          f"{str(unique_tgts):<20} "
          f"{np.mean(dists):+5.3f}")

print()
print(f"Pairwise spawn distance (across {args.num_resets} resets):")
import collections
pair_acc = collections.defaultdict(list)
for i, j, d in pair_dists:
    pair_acc[(i, j)].append(d)
for (i, j), ds in sorted(pair_acc.items()):
    arr = np.array(ds)
    print(f"  drone {i} <-> drone {j}: mean={arr.mean():.3f} m  min={arr.min():.3f}  max={arr.max():.3f}")

env.close()
sim_app.close()
