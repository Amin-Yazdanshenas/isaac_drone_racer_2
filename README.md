![Isaac Drone Racer](media/motion_trace1.jpg)

---

# Isaac Drone Racer 2: RL-Based Autonomous Drone Racing in Isaac Sim 5.1

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Isaac Drone Racer 2** is my modernized fork of the original Isaac Drone Racer project, an open-source reinforcement learning framework for autonomous drone racing initially developed by Kousheek Chakraborty.

In this updated version, I ported and reorganized the framework for newer NVIDIA robotics simulation tools, including Isaac Sim 5.1, Isaac Lab 2.3.2, and Python 3.11. The goal of this fork is to provide a cleaner and more practical research platform for high-speed autonomous UAV racing, reinforcement learning, camera-based policy learning, and sim-to-real development.

My contributions include:

Porting the framework to Isaac Sim 5.1, Isaac Lab 2.3.2, and Python 3.11
Cleaning and reorganizing the forked repository for easier setup and experimentation
Focusing the implementation on practical skrl PPO-based reinforcement learning baselines
Supporting no-camera and camera-based drone racing training modes
Working with FPV camera and IMU-based onboard observations
Supporting asymmetric actor-critic learning with privileged simulation state for the critic
Exploring multiple control/action interfaces, including motor angular-velocity commands and CTBR control
Improving documentation for setup, training, playback, TensorBoard monitoring, and troubleshooting

The project is intended for researchers and developers interested in reinforcement learning, autonomous drone racing, UAV control, robotics simulation, Isaac Sim, Isaac Lab, camera-based policy learning, and sim-to-real transfer.

Demo video: [https://www.youtube.com/watch?v=PsDE_80xkKw](https://www.youtube.com/watch?v=PsDE_80xkKw)

---

## Features

1. **Accurate physics modelling** — rotor first-order dynamics, aerodynamic drag, motor allocation matrix.
2. **Two action interfaces** — direct motor ω (legacy) and CTBR (collective thrust + body rates) with a 400 Hz PD rate controller.
3. **Manager-based env** — Isaac Lab [manager-based architecture](https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html#manager-based).
4. **Onboard sensor suite** — pinhole FPV camera (64×64 semantic seg + RGB), IMU, contact sensor.
5. **Track generator** — gate poses defined inline; semantic labels per-gate for perception tasks.
6. **Asymmetric actor-critic** — actor sees only onboard sensors at deployment; critic sees privileged state during training.
7. **Logger + plotter** — per-episode CSV logs, training metrics in TensorBoard.

---

## Requirements

- **Isaac Sim 5.1**
- **Isaac Lab 2.3.2**
- **Python 3.11** (conda env `isaacsim`)
- **Ubuntu 22.04 (x64)**
- NVIDIA GPU. RTX 4090 (24 GB) reference; ≥ 8 GB VRAM for GUI, ≥ 6 GB headless.

---

## Setup

### Option A — Restore the exact conda environment (recommended)

```bash
git clone https://github.com/Amin-Yazdanshenas/isaac_drone_racer.git
cd isaac_drone_racer

# Create env from lockfile (10–20 min, ~10 GB)
conda env create -f environment.yml
conda activate isaacsim

# Register local packages (tasks/, utils/, dynamics/)
pip3 install -e .
```

> [!IMPORTANT]
> Re-run `pip3 install -e .` after every fresh clone. Skipping it causes `ModuleNotFoundError: No module named 'tasks'`.

### Option B — Manual installation

1. Follow [Isaac Lab pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html), pinning Isaac Lab to `v2.3.2`:
   ```bash
   git clone git@github.com:isaac-sim/IsaacLab.git
   cd IsaacLab && git checkout v2.3.2
   ```
2. Clone + install this repo:
   ```bash
   git clone https://github.com/Amin-Yazdanshenas/isaac_drone_racer.git
   cd isaac_drone_racer && pip3 install -e .
   ```

---

## Tasks

| Task ID | Action | Observation | Use case |
|---------|--------|-------------|----------|
| `Isaac-Drone-Racer-v0` / `-Play-v0` | Motor ω | FPV grayscale + IMU (actor) / GT state (critic) | Deployable vision policy |
| `Isaac-Drone-Racer-NoCam-v0` / `-NoCam-Play-v0` | Motor ω | Full GT state | Fast reward/dynamics iteration |
| `Isaac-Drone-Racer-CTBR-v0` / `-CTBR-Play-v0` | CTBR | FPV grayscale + IMU (actor) / GT state (critic) | Vision policy with rate controller |
| `Isaac-Drone-Racer-NoCam-CTBR-v0` / `-NoCam-CTBR-Play-v0` | CTBR | Full GT state | Fastest sanity test |

**Motor ω** outputs 4 per-motor angular velocity setpoints; **CTBR** outputs `[c, ω_x, ω_y, ω_z]` (collective thrust + body-rate setpoints) and is converted to motor torques by a PD rate controller at physics frequency. CTBR gains: [tasks/drone_racer/configs/ctbr_gains.yaml](tasks/drone_racer/configs/ctbr_gains.yaml).

---

## Usage

### Quick start — NoCam CTBR (recommended baseline)

Fastest convergence, no camera required. ~15 min on RTX 4090.

```bash
# Train
python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-CTBR-v0 --headless --num_envs 4096

# Play
python3 scripts/rl/play.py --task Isaac-Drone-Racer-NoCam-CTBR-Play-v0 --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer_nocam_ctbr/`

### NoCam motor-ω (legacy baseline)

```bash
python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-v0 --headless --num_envs 4096
python3 scripts/rl/play.py --task Isaac-Drone-Racer-NoCam-Play-v0 --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer_nocam/`

### Vision tasks (FPV camera + IMU, asymmetric AC)

Camera-based variants require `--enable_cameras`. Trains a deployable policy using only onboard sensors; the critic sees ground-truth during training only.

```bash
# Motor ω
python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --enable_cameras --num_envs 512
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --enable_cameras --headless --num_envs 1

# CTBR
python3 scripts/rl/train.py --task Isaac-Drone-Racer-CTBR-v0 --headless --enable_cameras --num_envs 512
python3 scripts/rl/play.py --task Isaac-Drone-Racer-CTBR-Play-v0 --enable_cameras --headless --num_envs 1
```

Checkpoints: `logs/skrl/drone_racer/` and `logs/skrl/drone_racer_ctbr/`.

> [!NOTE]
> Extra CLI args from [AppLauncher](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/launch_app.html) work. Hydra overrides also work — e.g. disable the motor first-order lag:
> ```bash
> python3 scripts/rl/train.py --task Isaac-Drone-Racer-NoCam-v0 --headless --num_envs 4096 env.actions.control_action.use_motor_model=False
> ```

---

## Monitoring training

```bash
conda activate isaacsim
tensorboard --logdir logs/skrl --port 6006
# open http://localhost:6006
```

---

## Reward configuration

Default `RewardsCfg` (`tasks/drone_racer/drone_racer_env_cfg.py`) reproduces the upstream PPO convergence behavior:

| Term | Weight | Notes |
|------|--------|-------|
| `terminating` | -500 | Crash penalty |
| `ang_vel_l2` | -0.0001 | Mild rate regularizer |
| `progress` | 20 | Signed `prev_dist − cur_dist` (PPO needs the negative gradient on retreat) |
| `gate_passed` | 400 | +1 pass / -1 miss (`penalize_miss=True`) |
| `lookat_next` | 0.1 | Heading prior |

Spawn (`CommandsCfg.target`): drone respawns at the previous gate +1 m forward (`spawn_lerp_alpha=0`, `spawn_forward_offset=1.0`, `spawn_forward_velocity=0`). `EventCfg.push_robot` is enabled (±0.1 N, ±0.05 N·m every 0–0.2 s) for domain randomization. `flyaway` terminates at 20 m from the target gate.

---

## Troubleshooting

- Always activate the conda env before running anything: `conda activate isaacsim`.
- First Isaac Sim launch may take up to 10 min to compile shaders. Subsequent runs are fast.
- Camera tasks require `--enable_cameras`. NoCam tasks do **not** — running them with `--enable_cameras` just wastes GPU.
- "Failed to add labels … using Replicator API" warnings on NoCam tasks are harmless — the camera is disabled in `__post_init__` so the labels are never read.

### GPU VRAM (GUI mode)

Isaac Sim 5.1 pre-allocates ~4–5 GB VRAM for the RTX renderer when not headless. On ≤ 6 GB GPUs this exhausts VRAM at scene creation → `CUDA error: CUBLAS_STATUS_ALLOC_FAILED`. Always pass `--headless` on small GPUs. GUI mode needs ≥ 8 GB.

---

## Next steps

- [x] CTBR action interface + tuned PD rate controller
- [x] skrl PPO baseline (motor ω and CTBR variants, with and without camera)
- [ ] Data-driven aerodynamic model — system identification + learned residuals
- [ ] Power consumption model — battery discharge tied to motor current
- [ ] Curriculum learning — staged gate spacing and speed targets
- [ ] Sim-to-real transfer — domain randomization of motor dynamics, drag, camera noise

---

## References

- **kousheekc/isaac_drone_racer** — upstream project this repo is forked from. Targets an older Isaac Sim / Isaac Lab stack (Isaac Sim 4.5, Isaac Lab v2.1). [https://github.com/kousheekc/isaac_drone_racer](https://github.com/kousheekc/isaac_drone_racer)
- **Kaufmann, E., Bauersfeld, L., Loquercio, A., Müller, M., Koltun, V., & Scaramuzza, D.** (2023). *Champion-level drone racing using deep reinforcement learning*. [doi.org/10.1038/s41586-023-06419-4](https://doi.org/10.1038/s41586-023-06419-4)
- **Rudin, N., Hoeller, D., Reist, P., & Hutter, M.** (2022). *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*. [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)
- **Ferede, R., De Wagter, C., Izzo, D., & de Croon, G. C. H. E.** (2024). *End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight*. [doi.org/10.1109/ICRA57147.2024.10611665](https://doi.org/10.1109/ICRA57147.2024.10611665)

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Contact

Amin Yazdanshenas — yazdanshenas.amin@gmail.com

Project link: [https://github.com/Amin-Yazdanshenas/isaac_drone_racer](https://github.com/Amin-Yazdanshenas/isaac_drone_racer)

Upstream author (kousheekc/isaac_drone_racer): Kousheek Chakraborty — kousheekc@gmail.com

Issues, bug reports, and PRs welcome.
