"""ADR-0064 (Phase 2) — hermetic checks on ``nvblox.launch.py``.

No real ROS 2 graph and no nvblox engine (an NVIDIA binary OpenRAL does not
bundle). Asserts the launch file's structural contract: pinned node/package/
plugin, a valid default params YAML carrying the frame contract, and a single
``ComposableNodeContainer`` with one nvblox ``ComposableNode`` and no
``LifecycleNode`` (nvblox is a plain composable node, like cuVSLAM — ADR-0064).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "nvblox.launch.py"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "nvblox.yaml"


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_slam_bringup_nvblox_launch_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_file_exists_and_parses() -> None:
    assert _CONFIG_PATH.is_file(), _CONFIG_PATH
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    assert isinstance(data, dict)
    assert "/**" in data, sorted(data)
    params = data["/**"]["ros__parameters"]
    # nvblox reconstructs in the same world frame cuVSLAM localizes in.
    assert params["global_frame"] == "map"
    assert params["use_tf_transforms"] is True
    assert params["publish_esdf_distance_slice"] is True
    # Occupancy mapping → the `~/static_occupancy_grid` OccupancyGrid the launch
    # remaps to `/map` (the backend-agnostic topic the static_layer consumes);
    # static_tsdf would publish only an ESDF slice, not an OccupancyGrid.
    assert params["mapping_type"] == "static_occupancy"
    assert params["use_depth"] is True and params["use_lidar"] is False
    # The `/map` prefilter derives its live global-frame band from robot geometry
    # + TF; nvblox's own workspace bounds must not bake in a scene-specific floor.
    assert params["static_mapper.workspace_bounds_type"] == "unbounded"
    assert "static_mapper.workspace_bounds_min_height_m" not in params
    assert "static_mapper.workspace_bounds_max_height_m" not in params
    # Persistent map: decay pinned to nvblox's no-decay extremes (strict bound —
    # free must be > 0.5, occupied < 0.5) so far voxels do not fade and
    # long-range Nav2 goals into previously-seen space still plan.
    assert params["static_mapper.free_region_decay_probability"] > 0.5
    assert params["static_mapper.occupied_region_decay_probability"] < 0.5


def test_launch_defaults_to_sim_depth_camera_topics() -> None:
    """ADR-0064 — deploy-sim's manifest depth camera is nvblox's default input."""
    mod = _import_launch_module()
    from launch.actions import DeclareLaunchArgument
    from launch_ros.actions import Node

    desc = mod.generate_launch_description()
    args = {a.name: a for a in desc.describe_sub_entities() if isinstance(a, DeclareLaunchArgument)}
    assert args["depth_image_topic"].default_value[0].text == (
        "/openral/cameras/front_depth/depth/image"
    )
    assert args["depth_camera_info_topic"].default_value[0].text == (
        "/openral/cameras/front_depth/depth/camera_info"
    )
    nodes = [
        a
        for a in desc.describe_sub_entities()
        if isinstance(a, Node)
        and getattr(a, "node_executable", None) == "depth_height_filter_node.py"
    ]
    assert len(nodes) == 1
    assert nodes[0].node_executable == "depth_height_filter_node.py"
    assert "robot_yaml" in args
    assert "height_filter_floor_clearance_m" in args
    assert "height_filter_min_body_height_m" in args


def test_launch_module_pins_node_package_and_plugin() -> None:
    mod = _import_launch_module()
    assert getattr(mod, "NODE_NAME") == "openral_nvblox"  # noqa: B009
    assert getattr(mod, "PACKAGE") == "nvblox_ros"  # noqa: B009
    assert getattr(mod, "PLUGIN") == "nvblox::NvbloxNode"  # noqa: B009
    assert Path(getattr(mod, "DEFAULT_PARAMS_PATH")).is_file()  # noqa: B009


def test_launch_description_shape() -> None:
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from launch_ros.actions import ComposableNodeContainer, LifecycleNode

    try:
        get_package_share_directory("openral_slam_bringup")
    except PackageNotFoundError:
        pytest.skip("openral_slam_bringup not built (run `just ros2-build`).")

    desc = mod.generate_launch_description()
    actions = desc.describe_sub_entities()
    containers = [a for a in actions if isinstance(a, ComposableNodeContainer)]
    assert len(containers) == 1, f"expected exactly 1 container, got {len(containers)}"
    lifecycle_nodes = [a for a in actions if isinstance(a, LifecycleNode)]
    assert len(lifecycle_nodes) == 0, (
        f"nvblox is a composable node, not a LifecycleNode — got {len(lifecycle_nodes)}."
    )
