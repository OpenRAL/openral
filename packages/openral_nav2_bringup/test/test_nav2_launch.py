"""ADR-0025 — hermetic checks on ``nav2.launch.py``.

These do NOT spawn a real Nav2 graph (that's the integration tier).
They assert the launch file's structural pieces remain intact and
that the panda_mobile-tuned config keeps its critical knobs.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "nav2.launch.py"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_panda_mobile.yaml"
_VISUAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "nav2_visual.yaml"


def _import_launch_module() -> ModuleType:
    """Import ``nav2.launch.py`` by file path.

    Mirrors the slam_toolbox test pattern: bypass the ROS 2 Python
    package shadowing by loading via ``spec_from_file_location``.
    """
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_nav2_bringup_launch_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_file_exists_and_parses() -> None:
    """The panda_mobile-tuned YAML ships in-tree and is valid."""
    assert _CONFIG_PATH.is_file(), _CONFIG_PATH
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    assert isinstance(data, dict)
    # MPPI uses Omni motion model — required for the holonomic base.
    controller_params = data["controller_server"]["ros__parameters"]
    follow_path = controller_params["FollowPath"]
    assert follow_path["motion_model"] == "Omni", (
        "panda_mobile is holonomic — MPPI motion_model must be 'Omni' "
        "(DiffDrive ignores lateral velocity commands and the base "
        "won't strafe). See nav2_panda_mobile.yaml."
    )
    # Symmetric lateral velocity bound — DiffDrive default is 0.0.
    assert follow_path.get("vy_min", 0.0) < 0.0, (
        "holonomic base needs symmetric vy bounds (vy_min < 0); "
        "see nav2_panda_mobile.yaml FollowPath.vy_min."
    )
    # Robot radius widened to panda_mobile's chassis envelope.
    global_costmap = data["global_costmap"]["global_costmap"]["ros__parameters"]
    local_costmap = data["local_costmap"]["local_costmap"]["ros__parameters"]
    assert global_costmap["robot_radius"] >= 0.30, global_costmap["robot_radius"]
    assert local_costmap["robot_radius"] >= 0.30, local_costmap["robot_radius"]


def test_launch_module_includes_upstream_navigation_launch() -> None:
    """The launch description includes the upstream Nav2 navigation_launch.py."""
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    try:
        desc = mod.generate_launch_description()
    except PackageNotFoundError as exc:
        # ``openral_nav2_bringup`` or upstream ``nav2_bringup`` not on
        # the ament index — legitimate skip path per CLAUDE.md §1.11
        # (the operator hasn't ``source install/setup.bash`` or
        # apt-installed Nav2). No fakes; just skip.
        pytest.skip(f"ament package missing (overlay not sourced?): {exc}")
    actions = desc.describe_sub_entities()
    # ADR-0025 — the upstream nav2 IncludeLaunchDescription is built at
    # launch time inside an OpaqueFunction (it RewrittenYaml-rewrites the
    # base params with this robot's `nav2_param_overrides()`), so it is
    # NOT a static top-level entity. Assert the single deferred function
    # plus the `robot_yaml` arg that drives the per-robot override.
    opaque = [a for a in actions if isinstance(a, OpaqueFunction)]
    assert len(opaque) == 1, f"expected exactly 1 deferred OpaqueFunction; got {len(opaque)}"
    arg_names = {a.name for a in actions if isinstance(a, DeclareLaunchArgument)}
    assert "robot_yaml" in arg_names, arg_names
    # ADR-0064 — the SLAM-backend selector arg drives the costmap profile.
    assert "slam_backend" in arg_names, arg_names
    default_path = getattr(mod, "DEFAULT_PARAMS_PATH")  # noqa: B009
    assert Path(default_path).is_file(), default_path


def test_slam_backend_selects_costmap_profile() -> None:
    """ADR-0064 — `_params_path_for_backend` maps the backend to the right config."""
    mod = _import_launch_module()
    visual = Path(mod._params_path_for_backend("visual"))
    lidar = Path(mod._params_path_for_backend("lidar"))
    assert visual.name == "nav2_visual.yaml", visual
    assert lidar.name == "nav2_panda_mobile.yaml", lidar
    # default / unknown backends fall back to the lidar (base) profile.
    assert Path(mod._params_path_for_backend("none")).name == "nav2_panda_mobile.yaml"
    # case-insensitive
    assert Path(mod._params_path_for_backend("VISUAL")).name == "nav2_visual.yaml"


def test_visual_profile_consumes_map_not_scan() -> None:
    """ADR-0064 — the visual profile's costmaps read `/map` via static_layer (no /scan).

    This is what makes Nav2 backend-agnostic: a lidar-less robot (cuVSLAM+nvblox)
    plans off the same `/map` interface slam_toolbox publishes, with no `/scan`.
    """
    assert _VISUAL_CONFIG_PATH.is_file(), _VISUAL_CONFIG_PATH
    data = yaml.safe_load(_VISUAL_CONFIG_PATH.read_text())
    for scope in ("global_costmap", "local_costmap"):
        cm = data[scope][scope]["ros__parameters"]
        assert "static_layer" in cm["plugins"], (scope, cm["plugins"])
        assert "obstacle_layer" not in cm["plugins"], (scope, cm["plugins"])
        assert "voxel_layer" not in cm["plugins"], (scope, cm["plugins"])
        sl = cm["static_layer"]
        assert sl["map_topic"] == "/map", sl
        # nvblox publishes /map RELIABLE+VOLATILE (not latched) — the static_layer
        # must NOT request transient_local or the QoS mismatches.
        assert sl["map_subscribe_transient_local"] is False, sl
    # collision_monitor must not depend on /scan (lidar-less): scan source disabled.
    cm = data["collision_monitor"]["ros__parameters"]
    assert cm["scan"]["enabled"] is False, cm["scan"]
    # geometry/planner still mirror the base (planner threads unknown space).
    assert data["planner_server"]["ros__parameters"]["GridBased"]["allow_unknown"] is True
