# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for swarm-racing tasks: drone cfg factory + pre-baked tinted USDs.

Runtime USD authoring on any descendant of the drone's rigid-body Xforms
invalidates Isaac Sim 5.1's PhysX tensor view. Workaround: bake N small
wrapper USDs (one per color slot) that reference the source 5_in_drone.usd
and override material at body/visuals + prop*/visuals BEFORE the sim
starts. Each drone i then spawns from its own pre-tinted wrapper.
"""

from __future__ import annotations

import copy
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg

from assets.five_in_drone import FIVE_IN_DRONE


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


_ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "5_in_drone"))
_TINTED_DIR = os.path.join(_ASSET_DIR, "_tinted")
_SOURCE_USD = os.path.join(_ASSET_DIR, "5_in_drone.usd")


def _bake_tinted_usd(drone_idx: int, color: tuple[float, float, float]) -> str:
    """Write a wrapper USD that references 5_in_drone.usd and overrides material
    at body/visuals + prop*/visuals. Returns the wrapper USD path. Cached on disk
    so subsequent runs reuse the file.
    """
    try:
        from pxr import Usd, UsdShade, Sdf, Gf
    except ImportError:
        return _SOURCE_USD  # fallback — no Isaac Sim, return original

    os.makedirs(_TINTED_DIR, exist_ok=True)
    out_path = os.path.join(_TINTED_DIR, f"drone_{drone_idx}.usd")
    if os.path.exists(out_path):
        return out_path

    stage = Usd.Stage.CreateNew(out_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    # Define root prim referencing the original.
    root_path = Sdf.Path("/a_5_in_drone")
    root = stage.DefinePrim(root_path, "Xform")
    # Reference path must be relative or absolute. Use absolute path to the source.
    root.GetReferences().AddReference(_SOURCE_USD)
    stage.SetDefaultPrim(root)

    # Define a UsdPreviewSurface material under the root.
    mat = UsdShade.Material.Define(stage, root_path.AppendChild("Looks").AppendChild("DroneColor"))
    shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(color[0] * 0.4, color[1] * 0.4, color[2] * 0.4)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Override material binding on body/visuals and prop1..prop4/visuals.
    target_paths = [root_path.AppendPath("body").AppendChild("visuals")]
    for p in range(1, 5):
        target_paths.append(root_path.AppendPath(f"prop{p}").AppendChild("visuals"))

    for tp in target_paths:
        over = stage.OverridePrim(tp)
        UsdShade.MaterialBindingAPI.Apply(over).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    stage.GetRootLayer().Save()
    print(f"[swarm] baked tinted USD: {out_path}  color={color}")
    return out_path


def make_drone_articulation(drone_idx: int) -> ArticulationCfg:
    """Return an ArticulationCfg for drone i with prim_path Drone_{i} and a
    pre-baked tinted USD."""
    color = DRONE_PALETTE[drone_idx % len(DRONE_PALETTE)]
    try:
        tinted_path = _bake_tinted_usd(drone_idx, color)
    except Exception as exc:
        print(f"[swarm] tinted-USD bake failed for drone {drone_idx}: {exc!r} — falling back to source")
        tinted_path = _SOURCE_USD

    spawn = copy.deepcopy(FIVE_IN_DRONE.spawn)
    spawn.usd_path = tinted_path
    return FIVE_IN_DRONE.replace(
        prim_path=f"{{ENV_REGEX_NS}}/Drone_{drone_idx}",
        spawn=spawn,
    )


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
    """No-op stub. Coloring is now baked into per-drone wrapper USDs at spawn
    time (see _bake_tinted_usd). Kept as a callable so older callers don't error."""
    print(f"[recolor_drones] no-op: colors are baked into per-drone tinted USDs at spawn time")
