#!/usr/bin/env python3
"""ADR-0064 — generate `nav2_visual.yaml` from the base lidar Nav2 config.

`nav2_visual.yaml` is the Nav2 costmap profile for the VISUAL SLAM backend
(cuVSLAM + nvblox): the global + local costmaps consume the backend-agnostic
`/map` `OccupancyGrid` via `static_layer` instead of ray-casting `/scan`, so a
lidar-less robot can navigate. It is DERIVED from `nav2_panda_mobile.yaml` so
the two stay in sync — re-run this one-shot after editing the base:

    python tools/gen_nav2_visual.py

Only the costmap obstacle source differs (see the diff applied below); all
geometry / planner / controller / behaviour tuning mirrors the base.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent.parent / "packages" / "openral_nav2_bringup" / "config"
_BASE = _DIR / "nav2_panda_mobile.yaml"
_OUT = _DIR / "nav2_visual.yaml"

_HEADER = (
    "# ADR-0064 — Nav2 costmap profile for the VISUAL SLAM backend (cuVSLAM + nvblox),\n"
    "# DERIVED from nav2_panda_mobile.yaml. The ONLY differences vs the base (lidar)\n"
    "# profile: the global+local costmaps consume the backend-agnostic `/map`\n"
    "# OccupancyGrid via `static_layer` (instead of ray-casting `/scan`), with\n"
    "# `map_subscribe_transient_local: False` to match nvblox's RELIABLE+VOLATILE\n"
    "# live-updating /map; and the collision_monitor's /scan source is disabled\n"
    "# (a lidar-less robot has no /scan). Everything else (geometry, planner\n"
    "# allow_unknown, controller, behaviours) mirrors the base — see that file for\n"
    "# the tuning rationale. This makes Nav2 plan off `/map` regardless of HOW the\n"
    "# map was built (slam_toolbox lidar vs cuVSLAM+nvblox vision).\n"
    "# Regenerate after editing the base: python tools/gen_nav2_visual.py\n\n"
)

_STATIC_LAYER_ON_MAP = {
    "plugin": "nav2_costmap_2d::StaticLayer",
    "map_topic": "/map",
    # nvblox publishes /map RELIABLE+VOLATILE (live-updating), NOT latched — so
    # the static_layer must NOT request transient_local or the QoS mismatches and
    # the layer never receives the map. Verified live (ros2 topic info /map).
    "map_subscribe_transient_local": False,
}


def main() -> int:
    cfg = yaml.safe_load(_BASE.read_text())

    gc = cfg["global_costmap"]["global_costmap"]["ros__parameters"]
    gc["plugins"] = ["static_layer", "inflation_layer"]
    gc["static_layer"] = dict(_STATIC_LAYER_ON_MAP)
    gc.pop("obstacle_layer", None)  # no /scan on a lidar-less robot

    lc = cfg["local_costmap"]["local_costmap"]["ros__parameters"]
    lc["plugins"] = ["static_layer", "inflation_layer"]
    lc["static_layer"] = dict(_STATIC_LAYER_ON_MAP)
    lc.pop("voxel_layer", None)

    # collision_monitor: a lidar-less robot has no /scan. Keep the source LISTED
    # (nav2's validator aborts on an empty observation_sources) but DISABLED, so
    # the monitor runs (lifecycle_manager needs it ACTIVE) without waiting on a
    # scan that never arrives. The FootprintApproach polygon is already disabled
    # in the base. (A real visual robot would point this at a depth-derived scan.)
    cm = cfg["collision_monitor"]["ros__parameters"]
    if "scan" in cm:
        cm["scan"]["enabled"] = False

    if "map_saver" in cfg:
        cfg["map_saver"]["ros__parameters"]["map_subscribe_transient_local"] = False

    with _OUT.open("w") as f:
        f.write(_HEADER)
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, width=100)
    print(f"wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
