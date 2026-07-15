"""Hermetic checks on ``pycuvslam.launch.py``.

These do NOT spawn a real ROS 2 graph and do NOT need the cuVSLAM wheel
(an NVIDIA-licensed engine OpenRAL does not bundle). They assert the
launch file's structural contract:

* The node name / package / executable the deployment targets stay pinned.
* The default parameter YAML exists, parses, and carries the OpenRAL
  frame contract (``map``/``odom``) plus the rectified-stereo camera bus.
* The launch description composes exactly one plain ``Node`` — no
  ``ComposableNodeContainer`` (PyCuVSLAM has no NITROS zero-copy path)
  and no ``LifecycleNode``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "pycuvslam.launch.py"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pycuvslam.yaml"


def _import_launch_module() -> ModuleType:
    """Import ``pycuvslam.launch.py`` directly by file path."""
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_slam_bringup_pycuvslam_launch_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_params_file_exists_and_parses() -> None:
    """The default YAML ships in-tree, is valid, and carries the frame contract."""
    assert _CONFIG_PATH.is_file(), _CONFIG_PATH
    data = yaml.safe_load(_CONFIG_PATH.read_text())
    assert isinstance(data, dict)
    assert "/**" in data, sorted(data)
    params = data["/**"]["ros__parameters"]
    # PyCuVSLAM fills the same map→odom edge as the other SLAM backends.
    assert params["map_frame"] == "map"
    assert params["odom_frame"] == "odom"
    assert params["sync_slop_s"] > 0.0
    # The four stereo topics are owned by the launch args (so a scene's
    # slam_stereo_cameras can retarget them), NOT pinned in the YAML.
    assert "left_image_topic" not in params
    assert "right_image_topic" not in params


def test_launch_declares_stereo_camera_topic_args() -> None:
    """The launch owns the four rectified-stereo topics as overridable args."""
    mod = _import_launch_module()
    desc = None
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        get_package_share_directory("openral_slam_bringup")
        desc = mod.generate_launch_description()
    except PackageNotFoundError:
        pytest.skip("openral_slam_bringup not built (run `just ros2-build`).")
    from launch.actions import DeclareLaunchArgument

    defaults = {
        a.name: a.default_value[0].text  # single TextSubstitution
        for a in desc.describe_sub_entities()
        if isinstance(a, DeclareLaunchArgument)
    }
    assert defaults["left_image_topic"] == "/openral/cameras/left/image"
    assert defaults["left_camera_info_topic"] == "/openral/cameras/left/camera_info"
    assert defaults["right_image_topic"] == "/openral/cameras/right/image"
    assert defaults["right_camera_info_topic"] == "/openral/cameras/right/camera_info"


def test_launch_module_pins_node_package_and_executable() -> None:
    """The deployment targets these identifiers — pin them."""
    mod = _import_launch_module()
    assert getattr(mod, "NODE_NAME") == "openral_pycuvslam"  # noqa: B009
    assert getattr(mod, "PACKAGE") == "openral_slam_bringup"  # noqa: B009
    assert getattr(mod, "EXECUTABLE") == "pycuvslam_node.py"  # noqa: B009
    default_path = getattr(mod, "DEFAULT_PARAMS_PATH")  # noqa: B009
    assert Path(default_path).is_file(), default_path


def test_launch_description_shape() -> None:
    """One plain Node — no container (no NITROS path) and no LifecycleNode."""
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node

    try:
        get_package_share_directory("openral_slam_bringup")
    except PackageNotFoundError:
        pytest.skip("openral_slam_bringup not built (run `just ros2-build`).")

    desc = mod.generate_launch_description()
    actions = desc.describe_sub_entities()
    nodes = [a for a in actions if type(a) is Node]
    assert len(nodes) == 1, f"expected exactly 1 plain Node, got {len(nodes)}"
    assert not [a for a in actions if isinstance(a, ComposableNodeContainer)]
    assert not [a for a in actions if isinstance(a, LifecycleNode)]
