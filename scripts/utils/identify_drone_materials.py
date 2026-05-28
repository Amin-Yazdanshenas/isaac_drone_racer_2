# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: bake a wrapper drone USD where each source material gets a
unique bright color. Open the wrapper in Isaac Sim / usdview, identify which
color sits on the part you want to recolor, look up the material name in the
printed legend, then tell me the material name(s).

Run:
    conda activate isaacsim
    python3 scripts/utils/identify_drone_materials.py
    # Then open: assets/5_in_drone/_tinted/_diagnostic.usd
    # In Isaac Sim: File > Open
    # In usdview: usdview assets/5_in_drone/_tinted/_diagnostic.usd

You can also pass --only F_xxxxx to retint just that one material in red,
which is the fastest way to confirm a candidate ("does retinting this hit
the arms?").
"""

from __future__ import annotations

import argparse
import colorsys
import os

ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "5_in_drone"))
TINTED_DIR = os.path.join(ASSET_DIR, "_tinted")
SOURCE_USD = os.path.join(ASSET_DIR, "5_in_drone.usd")
PHYSICS_USD = os.path.join(ASSET_DIR, "configuration", "5_in_drone_physics.usd")


def bake_diagnostic(only: str | None = None) -> str:
    from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf

    os.makedirs(TINTED_DIR, exist_ok=True)
    out_path = os.path.join(TINTED_DIR, "_diagnostic.usd")
    if os.path.exists(out_path):
        os.remove(out_path)

    stage = Usd.Stage.CreateNew(out_path)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = stage.DefinePrim("/Drone", "Xform")
    root.GetReferences().AddReference(SOURCE_USD)
    stage.SetDefaultPrim(root)

    # Discover all source materials bound to body/prop subsets and their counts.
    src = Usd.Stage.Open(PHYSICS_USD, Usd.Stage.LoadAll)
    mat_subsets: dict[str, list[str]] = {}  # mat_name -> list of src subset paths
    parents = ["/visuals/body", "/visuals/prop1", "/visuals/prop2", "/visuals/prop3", "/visuals/prop4"]
    for parent_path in parents:
        parent = src.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            continue
        for child in Usd.PrimRange(parent):
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
            mat_subsets.setdefault(mat_name, []).append(str(child.GetPath()))

    # Deactivate instancing so descendant overrides are legal.
    for parent_path in parents:
        dst = "/Drone" + parent_path[len("/visuals"):] if False else "/Drone" + parent_path.replace("/visuals", "")
        # Actually mapping: /visuals/body -> /Drone/body/visuals etc.
        if parent_path == "/visuals/body":
            dst = "/Drone/body/visuals"
        else:
            i = parent_path[-1]
            dst = f"/Drone/prop{i}/visuals"
        over = stage.OverridePrim(Sdf.Path(dst))
        over.SetInstanceable(False)

    # Assign each material a unique color via HSV → RGB sweep.
    names = sorted(mat_subsets.keys())
    n = len(names)
    legend: list[tuple[str, tuple[float, float, float], int]] = []
    for idx, mat_name in enumerate(names):
        if only and mat_name != only:
            continue
        h = idx / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0) if not only else (1.0, 0.05, 0.05)
        # Define a tint material under /Drone/Looks/Diag_<idx>
        mat_path = Sdf.Path(f"/Drone/Looks/Diag_{idx}")
        tint = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r * 0.3, g * 0.3, b * 0.3))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tint.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        legend.append((mat_name, (r, g, b), len(mat_subsets[mat_name])))
        # Bind tint on every subset bound to this source material.
        for src_subset_path in mat_subsets[mat_name]:
            # Translate /visuals/body/... -> /Drone/body/visuals/...
            if src_subset_path.startswith("/visuals/body"):
                dst = "/Drone/body/visuals" + src_subset_path[len("/visuals/body"):]
            else:
                # /visuals/propX/...
                rest = src_subset_path[len("/visuals/prop"):]  # "X/..."
                pi = rest[0]
                dst = f"/Drone/prop{pi}/visuals" + rest[1:]
            over = stage.OverridePrim(Sdf.Path(dst))
            UsdShade.MaterialBindingAPI.Apply(over).Bind(
                tint, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )

    stage.GetRootLayer().Save()

    # Print legend so user can map color back to material name.
    print(f"\n[diag] wrote {out_path}\n")
    print(f"{'COLOR (R, G, B)':<28}  {'subsets':>7}  material_name")
    print("-" * 80)
    for mat_name, rgb, cnt in legend:
        rgb_str = f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})"
        print(f"{rgb_str:<28}  {cnt:>7}  {mat_name}")
    print()
    print("HOW TO USE:")
    print(f"  1. Open in Isaac Sim:  ./isaaclab.sh -p {out_path}   (or File > Open in Isaac UI)")
    print(f"     Or with usdview:     usdview {out_path}")
    print("  2. Find the part you want to recolor — note its color from the legend above.")
    print("  3. Tell Claude the material_name(s).")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="If set, only this material name gets retinted (bright red). Useful for confirmation.")
    args = parser.parse_args()
    bake_diagnostic(only=args.only)
