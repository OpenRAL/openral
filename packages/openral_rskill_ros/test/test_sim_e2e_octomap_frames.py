"""The octomap leg's frames come from the manifest, not `odom`/`base_link`.

`octomap_server` accumulates its octree in a frame that must not move under the
robot, and `octomap_voxel_bridge` / `WorldCloudBridge` express the result in the
robot's base frame. Both were the literals ``"odom"`` and ``"base_link"`` — the
MOBILE-BASE convention, which holds only while something publishes odometry.

A fixed-base arm has neither frame. Nothing in its graph publishes `odom`, and
its base link is named by the manifest (`openarm_base`, `panda_link0`,
`pelvis`). The failure is silent in the worst way: every node comes up, reports
healthy, and drops each cloud on a TF lookup, so the octree,
``/openral/world_voxels`` and the dashboard's pointcloud card all stay empty
with no error anywhere in the graph.

Observed on hardware 2026-09-07: a ZED-M feeding `octomap_server` through a
bimanual OpenArm deploy produced exactly that — a healthy dashboard with an
empty POINTCLOUD card.

Hermetic (no live ROS graph). Same import/skip pattern as
``test_sim_e2e_visual_slam``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

_LAUNCH_FILE = Path(__file__).resolve().parent.parent / "launch" / "sim_e2e.launch.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_launch_module() -> ModuleType:
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    pytest.importorskip("openral_foxglove_bringup.topics")
    if not os.environ.get("ROS_DISTRO"):
        pytest.skip("ROS_DISTRO not set — launch_ros requires a sourced ROS 2 install.")
    spec = importlib.util.spec_from_file_location(
        "openral_sim_e2e_octomap_under_test", _LAUNCH_FILE
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"failed to build module spec for {_LAUNCH_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _description(robot_id: str):
    from openral_core import RobotDescription

    return RobotDescription.from_yaml(_REPO_ROOT / "robots" / robot_id / "robot.yaml")


class TestFixedBaseRobots:
    """No odometry exists, so the manifest's base frame is the fixed frame."""

    def test_openarm_maps_in_its_own_base_frame(self) -> None:
        module = _import_launch_module()
        description = _description("openarm")
        assert description.capabilities.locomotion == ["none"]

        fixed_frame, base_frame = module._octomap_frames(description)

        assert fixed_frame == description.base_frame == "openarm_base"
        assert base_frame == "openarm_base"
        # The regression this guards: nothing publishes `odom` for this robot.
        assert fixed_frame != "odom"
        assert base_frame != "base_link"

    def test_a_fixed_arm_that_names_its_base_differently_is_honoured(self) -> None:
        """franka_panda's base is `panda_link0` — neither old literal applies."""
        module = _import_launch_module()
        description = _description("franka_panda")
        assert description.capabilities.locomotion == ["none"]

        fixed_frame, base_frame = module._octomap_frames(description)

        assert fixed_frame == base_frame == description.base_frame == "panda_link0"


class TestMobileRobots:
    """A robot that actually moves keeps accumulating in its odometry frame."""

    def test_panda_mobile_still_maps_in_odom(self) -> None:
        module = _import_launch_module()
        description = _description("panda_mobile")
        assert any(kind != "none" for kind in description.capabilities.locomotion)

        fixed_frame, base_frame = module._octomap_frames(description)

        assert fixed_frame == description.odom_frame == "odom"
        assert base_frame == description.base_frame == "base_link"
        # Behaviour-preserving: exactly the two literals this change replaced.


def test_every_intree_manifest_resolves_to_frames_it_declares() -> None:
    """Whatever the robot, both frames are names the manifest itself carries."""
    module = _import_launch_module()
    manifests = sorted((_REPO_ROOT / "robots").glob("*/robot.yaml"))
    assert manifests, "no in-tree robot manifests found"
    for manifest in manifests:
        description = _description(manifest.parent.name)
        fixed_frame, base_frame = module._octomap_frames(description)
        assert base_frame == description.base_frame
        assert fixed_frame in {description.base_frame, description.odom_frame}
