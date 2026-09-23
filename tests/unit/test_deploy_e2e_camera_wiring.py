"""The deploy launch derives its camera topics from the robot manifest, not from a guess.

Two consumers used to keep a node-default camera name that most robots do not
declare — the reasoner's completion camera (``top``) and the world-state
object-lift depth fallback (``front_depth``) — so on every other robot they
subscribed to a topic nothing publishes and silently did nothing. The launch
now resolves both from the manifest; these tests pin that resolution on real
manifests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from openral_core import RobotDescription

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH = REPO_ROOT / "packages/openral_rskill_ros/launch/deploy_e2e.launch.py"


@pytest.fixture(scope="module")
def launch_module() -> object:
    """The real launch file, loaded by path — it is not an importable package."""
    for dep in ("launch", "launch_ros", "lifecycle_msgs", "openral_foxglove_bringup"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location("deploy_e2e_launch_cameras", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_e2e_launch_cameras"] = module
    spec.loader.exec_module(module)
    return module


def _robot(name: str) -> RobotDescription:
    return RobotDescription.from_yaml(REPO_ROOT / "robots" / name / "robot.yaml")


@pytest.mark.parametrize(
    ("robot", "camera"),
    [
        ("panda_mobile", "shoulder_left"),  # optical-framed RGB wins over the first (head)
        ("so101_follower", "top"),  # no optical frame: the manifest's first RGB sensor
        ("ur5e", ""),  # no RGB sensor at all: empty disables the subscription
    ],
)
def test_primary_rgb_camera_follows_the_manifest(
    launch_module: object, robot: str, camera: str
) -> None:
    assert launch_module._primary_rgb_camera(_robot(robot)) == camera  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("robot", "topic"),
    [
        ("panda_mobile", "/openral/cameras/front_depth/points"),
        ("openarm", "/openral/cameras/head_zed/points"),
        ("so101_follower", ""),
    ],
)
def test_depth_points_topic_names_the_manifests_depth_sensor(
    launch_module: object, robot: str, topic: str
) -> None:
    assert launch_module._depth_points_topic(_robot(robot)) == topic  # type: ignore[attr-defined]


def test_the_launch_no_longer_hardcodes_either_camera() -> None:
    """The consumers' node defaults must not survive as literals in the launch."""
    source = LAUNCH.read_text(encoding="utf-8")
    assert "/openral/cameras/top/image" not in source
    assert 'reasoner_params["completion_camera_topic"]' in source
    assert '"object_depth_points_topic": _depth_points_topic(description)' in source
    # ``octomap_cloud_topic`` keeps its documented sim default (scenes pin it on hardware);
    # the object-lift fallback is the only ``front_depth`` literal that was silently dead.
    # (launch-arg default + the helper's docstring example)
    assert source.count("/openral/cameras/front_depth/points") == 2
