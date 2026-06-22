"""ADR-0064 — hermetic checks on ``cuvslam.launch.py``.

These do NOT spawn a real ROS 2 graph and do NOT need the cuVSLAM engine
(an NVIDIA binary OpenRAL does not bundle). They assert the launch file's
structural contract so a regression surfaces at unit-test time:

* The composable node name / package / plugin the deployment targets stay
  pinned.
* The default parameter YAML exists, parses, and carries the OpenRAL frame
  contract (``map``/``odom``/``base_link``).
* The launch description composes exactly one ``ComposableNodeContainer``
  holding exactly one cuVSLAM ``ComposableNode`` — and, unlike
  ``slam_toolbox.launch.py``, there is NO ``LifecycleNode`` (cuVSLAM is a
  plain composable node, not a ROS 2 lifecycle node — ADR-0064).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "cuvslam.launch.py"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "cuvslam.yaml"


def _import_launch_module() -> ModuleType:
    """Import ``cuvslam.launch.py`` directly by file path."""
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("ament_index_python")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")

    spec = importlib.util.spec_from_file_location(
        "openral_slam_bringup_cuvslam_launch_under_test", _LAUNCH_FILE
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
    # cuVSLAM fills the same map→odom→base_link chain as slam_toolbox.
    assert params["map_frame"] == "map"
    assert params["odom_frame"] == "odom"
    assert params["base_frame"] == "base_link"
    assert params["publish_odom_to_base_tf"] is True


def test_launch_module_pins_node_package_and_plugin() -> None:
    """The deployment targets these upstream identifiers — pin them."""
    mod = _import_launch_module()
    assert getattr(mod, "NODE_NAME") == "openral_visual_slam"  # noqa: B009
    assert getattr(mod, "PACKAGE") == "isaac_ros_visual_slam"  # noqa: B009
    assert "VisualSlamNode" in getattr(mod, "PLUGIN")  # noqa: B009
    default_path = getattr(mod, "DEFAULT_PARAMS_PATH")  # noqa: B009
    assert Path(default_path).is_file(), default_path


def test_launch_description_shape() -> None:
    """One container, one cuVSLAM composable node, and NO LifecycleNode.

    cuVSLAM is a plain composable node (ADR-0064) — composing it makes it
    live, so there is no UNCONFIGURED→INACTIVE auto-transition and no
    Reasoner-driven CONFIGURE/ACTIVATE (contrast slam_toolbox, ADR-0025).
    """
    mod = _import_launch_module()
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from launch_ros.actions import ComposableNodeContainer, LifecycleNode

    # generate_launch_description() resolves the package share dir for the
    # default params path; that only exists in a *built* workspace
    # (`just ros2-build`). In a bare source checkout the package is not on
    # the ament index — skip rather than fail on the environment.
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
        "cuVSLAM is a composable node, not a LifecycleNode — ADR-0064. "
        f"Got {len(lifecycle_nodes)} LifecycleNode(s)."
    )
