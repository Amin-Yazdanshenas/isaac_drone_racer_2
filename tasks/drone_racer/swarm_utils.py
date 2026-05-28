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
_BASE_DAE = os.path.join(_ASSET_DIR, "meshes", "base_link.dae")


def _collect_arm_material_names() -> set[str]:
    """Return the source material name(s) bound to the drone's 4 arms.

    Verified by bbox inspection of /visuals/body/base_link/mesh subsets:
    Material_001..004 are the four arms of the X-frame quad (each at a ±X,±Z
    corner, ~0.11 m from origin, thin Y plate). All four GeomSubsets share a
    single source material "Material_004", so binding the tint there recolors
    all arms.
    """
    return {"Material_004"}


def _bake_tinted_usd(drone_idx: int, color: tuple[float, float, float]) -> str:
    """Write a wrapper USD that references 5_in_drone.usd, deactivates the
    visuals' instanceable flag, then re-binds every Mesh + GeomSubset to our
    tinted material. Returns the wrapper USD path. Cached on disk so
    subsequent runs reuse the file.

    Writes to a temp file and atomically renames on success so a partial file
    can never poison the cache.
    """
    try:
        from pxr import Usd, UsdShade, Sdf, Gf
    except ImportError:
        return _SOURCE_USD  # fallback — no Isaac Sim, return original

    os.makedirs(_TINTED_DIR, exist_ok=True)
    out_path = os.path.join(_TINTED_DIR, f"drone_{drone_idx}.usd")
    if os.path.exists(out_path):
        return out_path

    # USD CreateNew picks file format from extension — must be .usd/.usda/.usdc.
    # ".part" was silently rejected. Use a sibling .usd temp name instead.
    tmp_path = os.path.join(_TINTED_DIR, f"_partial_drone_{drone_idx}.usd")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    stage = Usd.Stage.CreateNew(tmp_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")

    # Wrapper root is named "Drone" (NOT "a_5_in_drone") to avoid a name
    # collision when Isaac references this wrapper INTO Drone_i — with the
    # same name on both sides composition can produce an unintended extra
    # nesting layer (/Drone_i/a_5_in_drone/...).
    root_path = Sdf.Path("/Drone")
    root = stage.DefinePrim(root_path, "Xform")
    root.GetReferences().AddReference(_SOURCE_USD)
    stage.SetDefaultPrim(root)

    # Material under the wrapper root.
    mat = UsdShade.Material.Define(stage, root_path.AppendChild("Looks").AppendChild("DroneColor"))
    shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(color[0] * 0.15, color[1] * 0.15, color[2] * 0.15)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Walk the source USD to discover every Mesh + GeomSubset prim. The mesh
    # tree lives at /visuals/body and /visuals/prop{1..4}/prop and is brought
    # into the articulation namespace (/a_5_in_drone/body/visuals,
    # /a_5_in_drone/prop{i}/visuals) via internal references inside the payload.
    # USD's offline composition doesn't expose those mapped paths via
    # GetChildren() — so we walk the reference TARGET and translate paths.
    from pxr import UsdGeom

    physics_usd = os.path.join(_ASSET_DIR, "configuration", "5_in_drone_physics.usd")
    src_stage = Usd.Stage.Open(physics_usd, Usd.Stage.LoadAll)
    target_relpaths: list[str] = []

    # Mapping: source mesh root -> wrapper namespace path under /Drone.
    body_src, body_dst = "/visuals/body", "/Drone/body/visuals"
    prop_mapping = {
        f"/visuals/prop{i}": f"/Drone/prop{i}/visuals" for i in (1, 2, 3, 4)
    }
    # STEP 1: deactivate instancing on the visual roots BEFORE authoring any
    # descendant overrides. Source has instanceable=true on body/visuals + the
    # four prop*/visuals — that makes every Mesh + GeomSubset inside an
    # instance proxy (uneditable). Override the flag first.
    for dst_prefix in [body_dst, *prop_mapping.values()]:
        over = stage.OverridePrim(Sdf.Path(dst_prefix))
        over.SetInstanceable(False)

    # STEP 2a: props — bind tint on the parent Mesh, covers all prop subsets.
    for src_prefix, dst_prefix in prop_mapping.items():
        src_prim = src_stage.GetPrimAtPath(src_prefix)
        if not src_prim.IsValid():
            continue
        for child in Usd.PrimRange(src_prim):
            if UsdGeom.Mesh(child):
                rest = str(child.GetPath())[len(src_prefix):]
                target_relpaths.append(dst_prefix + rest)
                break  # one Mesh per prop is enough

    # STEP 2b: body — only retint subsets whose ORIGINAL material is the
    # pink/magenta arm color. Other body subsets (frame, motors, camera mount)
    # keep their stock material. Pink set is determined by parsing the source
    # DAE diffuse colors: bright magenta (Blender default "Material.00x" =
    # (1,0,1)) plus the one red-pink F_a306... arm material.
    arm_material_names = _collect_arm_material_names()
    body_prim = src_stage.GetPrimAtPath(body_src)
    if body_prim.IsValid():
        for child in Usd.PrimRange(body_prim):
            if not UsdGeom.Subset(child):
                continue
            bapi = UsdShade.MaterialBindingAPI(child)
            rel = bapi.GetDirectBindingRel()
            if not rel:
                continue
            targets = rel.GetTargets()
            if not targets:
                continue
            mat_name = str(targets[0]).rsplit("/", 1)[-1]
            if mat_name not in arm_material_names:
                continue
            rest = str(child.GetPath())[len(body_src):]
            target_relpaths.append(body_dst + rest)

    # STEP 3: bind tinted material at every selected target path.
    for tp in target_relpaths:
        over = stage.OverridePrim(Sdf.Path(tp))
        UsdShade.MaterialBindingAPI.Apply(over).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    stage.GetRootLayer().Save()
    # Atomic rename onto the cached path.
    os.replace(tmp_path, out_path)
    print(f"[swarm] baked tinted USD: {out_path}  color={color}  overrides={len(target_relpaths)}")
    return out_path


def make_drone_articulation(drone_idx: int) -> ArticulationCfg:
    """Return an ArticulationCfg for drone i with prim_path Drone_{i} and a
    pre-baked tinted USD."""
    color = DRONE_PALETTE[drone_idx % len(DRONE_PALETTE)]
    try:
        tinted_path = _bake_tinted_usd(drone_idx, color)
    except Exception as exc:
        print(f"[swarm] tinted-USD bake failed for drone {drone_idx}: {exc!r} — falling back to source")
        # Sweep up any partial wrapper file left behind so the next attempt
        # doesn't get the broken cached path.
        try:
            partial = os.path.join(_TINTED_DIR, f"_partial_drone_{drone_idx}.usd")
            if os.path.exists(partial):
                os.remove(partial)
            final = os.path.join(_TINTED_DIR, f"drone_{drone_idx}.usd")
            # Only remove final if it was just half-written (size 0 or tiny).
            if os.path.exists(final) and os.path.getsize(final) < 1024:
                os.remove(final)
        except Exception:
            pass
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
        force_threshold=10.0,  # phantom near-zero now that prop mass = 1e-6
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
