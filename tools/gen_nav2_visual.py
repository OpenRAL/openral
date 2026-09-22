#!/usr/bin/env python3
"""Generate `nav2_visual.yaml` from the base lidar Nav2 config.

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

import sys
from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent.parent / "packages" / "openral_nav2_bringup" / "config"
_BASE = _DIR / "nav2_panda_mobile.yaml"
_OUT = _DIR / "nav2_visual.yaml"

_HEADER = (
    "# Nav2 costmap profile for the VISUAL SLAM backend (cuVSLAM + nvblox),\n"
    "# DERIVED from nav2_panda_mobile.yaml. The ONLY differences vs the base (lidar)\n"
    "# profile: the global+local costmaps consume the backend-agnostic `/map`\n"
    "# OccupancyGrid via `static_layer` (instead of ray-casting `/scan`), with\n"
    "# `map_subscribe_transient_local: False` to match nvblox's RELIABLE+VOLATILE\n"
    "# live-updating /map; and the collision_monitor's /scan source is disabled\n"
    "# (a lidar-less robot has no /scan). Everything else (geometry, planner\n"
    "# allow_unknown, controller, behaviours) mirrors the base — see that file for\n"
    "# the tuning rationale. This makes Nav2 plan off `/map` regardless of HOW the\n"
    "# map was built (slam_toolbox lidar vs cuVSLAM+nvblox vision).\n"
    "# Regenerate after editing the base: just gen-nav2-visual\n\n"
)

_STATIC_LAYER_ON_MAP = {
    "plugin": "nav2_costmap_2d::StaticLayer",
    "map_topic": "/map",
    # nvblox publishes /map RELIABLE+VOLATILE (live-updating), NOT latched — so
    # the static_layer must NOT request transient_local or the QoS mismatches and
    # the layer never receives the map (ros2 topic info /map).
    "map_subscribe_transient_local": False,
}


def render(base_text: str) -> str:
    """Return the visual-SLAM profile derived from the base (lidar) profile text."""
    cfg = yaml.safe_load(base_text)

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

    return _HEADER + yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False, width=100)


def main(argv: list[str] | None = None) -> int:
    """Write the derived profile, or with ``--check`` exit 1 if the checked-in copy is stale."""
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    rendered = render(_BASE.read_text())
    if check:
        if _OUT.read_text() == rendered:
            return 0
        print(f"{_OUT} is stale — run `just gen-nav2-visual`", file=sys.stderr)
        return 1
    _OUT.write_text(rendered)
    print(f"wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
