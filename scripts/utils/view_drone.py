# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim GUI with a single drone USD loaded so you can click parts
and inspect material bindings. No usdview needed.

Run:
    conda activate isaacsim
    python3 scripts/utils/view_drone.py
    python3 scripts/utils/view_drone.py --usd assets/5_in_drone/_tinted/_diagnostic.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                          "assets", "5_in_drone", "_tinted", "_diagnostic.usd")),
                    help="Path to USD file to open.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import omni.usd
ctx = omni.usd.get_context()
ctx.open_stage(args.usd)

print(f"[view] opened: {args.usd}")
print("[view] In the GUI: click a prim in the viewport, look at the property panel for 'Material Binding'.")
print("[view] Ctrl+C in this terminal to quit.")

while sim_app.is_running():
    sim_app.update()

sim_app.close()
