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
    """Apply a distinct color to each drone via UsdPreviewSurface material override
    with strongerThanDescendants binding. Call AFTER the stage is built (after
    gym.make returns) so the per-env drone prims exist.
    """
    try:
        import omni.usd
        from pxr import Sdf, UsdShade, Gf, UsdGeom, Usd
    except ImportError:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[recolor_drones] no stage — skipping")
        return

    applied = 0
    skipped = 0
    for env_idx in range(num_envs):
        for i in range(num_drones):
            color = DRONE_PALETTE[i % len(DRONE_PALETTE)]
            drone_root = f"/World/envs/env_{env_idx}/Drone_{i}"
            drone_prim = stage.GetPrimAtPath(drone_root)
            if not drone_prim.IsValid():
                skipped += 1
                continue

            mat_path = f"{drone_root}/Looks/DroneColor"
            material = UsdShade.Material.Define(stage, Sdf.Path(mat_path))
            shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{mat_path}/Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(color[0] * 0.4, color[1] * 0.4, color[2] * 0.4)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Target ONLY the known visual Xform paths — body/visuals and the four
            # prop*/visuals. Walking every Imageable descendant (or calling
            # UnbindAllBindings on the body Xform itself) invalidates the PhysX
            # tensor view because Isaac Sim has already cached shape views on those
            # rigid-body Xforms.
            visual_paths = [f"{drone_root}/body/visuals"]
            for p in range(1, 5):  # prop1..prop4
                visual_paths.append(f"{drone_root}/prop{p}/visuals")

            for vp in visual_paths:
                vp_prim = stage.GetPrimAtPath(vp)
                if not vp_prim.IsValid():
                    continue
                UsdShade.MaterialBindingAPI.Apply(vp_prim).Bind(
                    material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
            applied += 1

    print(f"[recolor_drones] applied={applied} skipped={skipped} drones across {num_envs} envs")
