"""The deploy launch derives its camera topics from the robot manifest, not from a guess.

Three consumers used to keep a node-default camera name that most robots do not
declare — the reasoner's completion camera (``top``) and the world-state
object-lift depth fallback and octomap's ``cloud_in`` (``front_depth``) — so
on every other robot they subscribed to a topic nothing publishes and silently
did nothing. The launch now resolves them from the manifest; these tests pin that resolution on real
manifests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from openral_core import CameraTopicKind, RobotDescription, camera_topic

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
    assert launch_module._primary_rgb_camera(_robot(robot).sensors) == camera  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("robot", "camera"),
    [("panda_mobile", "front_depth"), ("openarm", "head_zed"), ("so101_follower", "")],
)
def test_depth_points_topic_names_the_manifests_depth_sensor(
    launch_module: object, robot: str, camera: str
) -> None:
    description = _robot(robot)
    assert launch_module._depth_camera(description) == camera  # type: ignore[attr-defined]
    topic = camera_topic(camera, CameraTopicKind.POINTS) if camera else ""
    assert launch_module._depth_points_topic(description) == topic  # type: ignore[attr-defined]


def test_the_launch_no_longer_hardcodes_either_camera() -> None:
    """The consumers' node defaults must not survive as literals in the launch."""
    source = LAUNCH.read_text(encoding="utf-8")
    assert "/openral/cameras/top/image" not in source
    assert 'reasoner_params["completion_camera_topic"]' in source
    assert '"object_depth_points_topic": _depth_points_topic(description)' in source
    assert "octomap_cloud_topic = _octomap_cloud_topic(octomap_cloud_topic, description)" in source
    # Only the helper's docstring example names ``front_depth`` — no launch default does.
    assert source.count("/openral/cameras/front_depth/points") == 1


@pytest.mark.parametrize(
    ("pinned", "robot", "topic"),
    [
        ("", "openarm", "/openral/cameras/head_zed/points"),  # derived: the sim bridge's cloud
        ("", "panda_mobile", "/openral/cameras/front_depth/points"),
        # A scene pin (a real depth driver) wins over the derivation.
        (
            "/zed/zed_node/point_cloud/cloud_registered",
            "openarm",
            "/zed/zed_node/point_cloud/cloud_registered",
        ),
        ("/camera/depth/color/points", "so101_follower", "/camera/depth/color/points"),
    ],
)
def test_octomap_cloud_topic_prefers_the_pin_then_the_manifest(
    launch_module: object, pinned: str, robot: str, topic: str
) -> None:
    got = launch_module._octomap_cloud_topic(pinned, _robot(robot))  # type: ignore[attr-defined]
    assert got == topic


def test_octomap_without_a_depth_cloud_fails_loud(launch_module: object) -> None:
    """``--enable-octomap`` on a depth-less robot must not map silence behind healthy nodes."""
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match="so101_follower"):
        launch_module._octomap_cloud_topic("", _robot("so101_follower"))  # type: ignore[attr-defined]


def test_real_deploy_picks_a_bound_camera(launch_module: object) -> None:
    """On a real deploy only a camera with a ``deploy_binding`` has a topic.

    OpenArm's manifest binds no camera (``top`` is a MuJoCo render), so its sim pick
    (``top``) is a dead topic on the real cell. With a scene that binds only
    ``wrist_left`` (host binding only, as ``check_scene_sensor_overrides`` requires),
    the completion/detector camera must be ``wrist_left``; with no binding at all it is
    empty, which disables the subscription instead of subscribing to silence.
    """
    from openral_core import SensorDeployBinding, SensorSpec

    arm = _robot("openarm")
    manifest_left = next(s for s in arm.sensors if s.name == "wrist_left")
    bound_left = SensorSpec(
        name="wrist_left",
        modality="rgb",
        frame_id=manifest_left.frame_id,
        rate_hz=30.0,
        deploy_binding=SensorDeployBinding(backend_params={"device": "/dev/video0"}),
    )
    publishing = launch_module._publishing_sensors  # type: ignore[attr-defined]
    pick = launch_module._primary_rgb_camera  # type: ignore[attr-defined]
    assert pick(publishing(arm, [], "sim")) == "top"
    assert pick(publishing(arm, [], "real")) == ""
    assert pick(publishing(arm, [bound_left], "real")) == "wrist_left"
