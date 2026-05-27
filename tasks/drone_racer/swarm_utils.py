# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for swarm-racing tasks: drone cfg factory + post-spawn recolor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg

from assets.five_in_drone import FIVE_IN_DRONE

if TYPE_CHECKING:
    pass


# 8-color palette for up to 8 drones. RGB in [0, 1].
DRONE_PALETTE: list[tuple[float, float, float]] = [
    (1.0, 0.2, 0.2),   # red
    (0.2, 0.8, 0.2),   # green
    (0.2, 0.4, 1.0),   # blue
    (1.0, 0.85, 0.1),  # yellow
    (1.0, 0.3, 1.0),   # magenta
    (0.2, 0.9, 0.9),   # cyan
    (1.0, 0.55, 0.0),  # orange
    (0.6, 0.2, 1.0),   # purple
]


def make_drone_articulation(drone_idx: int) -> ArticulationCfg:
    """Return an ArticulationCfg for drone i with prim_path Drone_{i}."""
    return FIVE_IN_DRONE.replace(prim_path=f"{{ENV_REGEX_NS}}/Drone_{drone_idx}")


def make_collision_sensor(drone_idx: int) -> ContactSensorCfg:
    """Body-only contact sensor with the doc-recommended buffer settings."""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Drone_{drone_idx}/body",
        history_length=3,
        update_period=0.0,
        force_threshold=10.0,
        debug_vis=False,
    )


def make_imu_sensor(drone_idx: int) -> ImuCfg:
    return ImuCfg(prim_path=f"{{ENV_REGEX_NS}}/Drone_{drone_idx}/body", debug_vis=False)


def make_tiled_camera(drone_idx: int) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Drone_{drone_idx}/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.14, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(),
        width=64,
        height=64,
    )


def recolor_drones(num_envs: int, num_drones: int) -> None:
    """Apply a distinct preview-surface color to each drone's `body` prim.

    Call AFTER the stage is built (e.g., from the env class's `_setup_scene`
    post-hook or right after `gym.make` returns). Uses USD Preview Surface
    so colors show in both ray-traced and rasterized viewports without
    needing MDL.
    """
    try:
        import omni.usd
        from pxr import Sdf, UsdShade, Gf
    except ImportError:
        # Not inside Isaac Sim — skip silently.
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for env_idx in range(num_envs):
        for i in range(num_drones):
            color = DRONE_PALETTE[i % len(DRONE_PALETTE)]
            drone_root = f"/World/envs/env_{env_idx}/Drone_{i}"
            mat_root = f"{drone_root}/Looks"
            mat_path = f"{mat_root}/DroneColor"
            shader_path = f"{mat_path}/Shader"

            # Build or reuse material.
            mat_prim = stage.GetPrimAtPath(mat_path)
            if not mat_prim.IsValid():
                stage.DefinePrim(Sdf.Path(mat_root), "Scope")
                material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
                shader = UsdShade.Shader.Define(stage, Sdf.Path(shader_path))
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            else:
                material = UsdShade.Material(mat_prim)

            # Bind to body link's mesh subtree.
            for prim in stage.Traverse():
                p = prim.GetPath().pathString
                if p.startswith(f"{drone_root}/body") and prim.IsA(UsdShade.ConnectableAPI) is False:
                    if prim.GetTypeName() in ("Mesh", "Xform"):
                        UsdShade.MaterialBindingAPI(prim).Bind(material)
