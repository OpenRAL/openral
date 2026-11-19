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
    return RobotDescription.from_yaml(str(REPO_ROOT / "robots" / name / "robot.yaml"))


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
def test_depth_camera_names_the_manifests_depth_sensor(
    launch_module: object, robot: str, camera: str
) -> None:
    assert launch_module._depth_camera(_robot(robot)) == camera  # type: ignore[attr-defined]


def test_the_launch_no_longer_hardcodes_either_camera() -> None:
    """The consumers' node defaults must not survive as literals in the launch."""
    source = LAUNCH.read_text(encoding="utf-8")
    assert "/openral/cameras/top/image" not in source
    assert 'reasoner_params["completion_camera_topic"]' in source
    # The lift fallback and octomap share one derivation (openral_core.deploy_cloud_topic).
    assert '"object_depth_points_topic": deploy_cloud_topic(' in source
    assert "octomap_cloud_topic = _octomap_cloud_topic(\n" in source
    assert "/openral/cameras/front_depth/points" not in source


@pytest.mark.parametrize(
    ("pinned", "robot", "hal_mode", "topic"),
    [
        ("", "openarm", "sim", "/openral/cameras/head_zed/points"),  # the sim bridge's cloud
        ("", "panda_mobile", "sim", "/openral/cameras/front_depth/points"),
        # A scene pin (a real depth driver) wins over the derivation.
        (
            "/zed/zed_node/point_cloud/cloud_registered",
            "openarm",
            "real",
            "/zed/zed_node/point_cloud/cloud_registered",
        ),
        ("/camera/depth/color/points", "so101_follower", "sim", "/camera/depth/color/points"),
    ],
)
def test_octomap_cloud_topic_prefers_the_pin_then_the_manifest(
    launch_module: object, pinned: str, robot: str, hal_mode: str, topic: str
) -> None:
    got = launch_module._octomap_cloud_topic(pinned, _robot(robot), hal_mode)  # type: ignore[attr-defined]
    assert got == topic


@pytest.mark.parametrize(
    ("robot", "hal_mode"),
    [
        ("so101_follower", "sim"),  # no depth sensor at all
        # A depth camera, but on real hardware only the sim bridge's name would be derived.
        ("panda_mobile", "real"),
        ("openarm", "real"),
    ],
)
def test_octomap_without_a_published_cloud_fails_loud(
    launch_module: object, robot: str, hal_mode: str
) -> None:
    """Octomap must never map silence behind healthy nodes, even from a direct ros2 launch."""
    from openral_core.exceptions import ROSConfigError

    with pytest.raises(ROSConfigError, match=robot):
        launch_module._octomap_cloud_topic("", _robot(robot), hal_mode)  # type: ignore[attr-defined]


def test_real_deploy_picks_a_bound_camera(launch_module: object) -> None:
    """On a real deploy only a camera with a ``deploy_binding`` has a topic.

    The OpenArm manifest binds all its cameras (a deploy scene never touches a robot
    camera), so the real pick equals the sim pick, ``top``. Strip the bindings
    and a camera the manifest leaves unbound is a dead topic on the real cell: with only
    ``wrist_left`` bound the completion/detector camera must be ``wrist_left``; with no
    binding at all it is empty, which disables the subscription instead of subscribing
    to silence.
    """
    from openral_core import publishing_sensors

    arm = _robot("openarm")
    only_left = [
        s if s.name == "wrist_left" else s.model_copy(update={"deploy_binding": None})
        for s in arm.sensors
    ]
    unbound = [s.model_copy(update={"deploy_binding": None}) for s in arm.sensors]
    pick = launch_module._primary_rgb_camera  # type: ignore[attr-defined]
    assert pick(publishing_sensors(arm.sensors, [], "sim")) == "top"
    assert pick(publishing_sensors(arm.sensors, [], "real")) == "top"
    assert pick(publishing_sensors(only_left, [], "real")) == "wrist_left"
    assert pick(publishing_sensors(unbound, [], "real")) == ""


@pytest.mark.parametrize("robot", ["makermods_metal", "makermods_metal_station"])
def test_real_deploy_names_the_top_cameras_optical_child(launch_module: object, robot: str) -> None:
    """#627: the launch's optical child for ``top`` is exactly the lift's frame.

    The object lift looks the body-convention ``top`` camera up in its optical child
    (``image_frame_id``); on hardware only the launch publishes it. In sim the sensor
    bridge does, so the launch must not.
    """
    from openral_core.depth_extrinsic import image_frame_id

    sensors = _robot(robot).sensors
    top = next(s for s in sensors if s.name == "top")
    mounts = launch_module._optical_mounts  # type: ignore[attr-defined]
    assert mounts(sensors, "real") == {"top_camera": image_frame_id(top, sensors)}
    assert image_frame_id(top, sensors) == "top_camera_optical_frame"
    assert mounts(sensors, "sim") == {}


@pytest.mark.parametrize(
    "robot", sorted(p.parent.name for p in (REPO_ROOT / "robots").glob("*/robot.yaml"))
)
def test_real_launch_publishes_every_optical_child_a_consumer_names(
    launch_module: object, robot: str
) -> None:
    """#641: on every manifest the real launch hangs exactly the optical children of
    ``test_optical_mount.MOUNTS`` (written out by hand) under their mounts, and those
    are the frames the object lift (``image_frame_id``) and the DA3 resolver name."""
    from openral_core.depth_extrinsic import image_frame_id, resolve_estimated_depth_camera

    from tests.unit.test_optical_mount import MOUNTS, lift_sensors

    sensors = lift_sensors(robot)
    want = MOUNTS.get(robot, {})
    mounts = launch_module._optical_mounts(sensors, "real")  # type: ignore[attr-defined]
    assert mounts == {frame: optical for _, frame, optical in want.values()}
    assert launch_module._optical_mounts(sensors, "sim") == {}  # type: ignore[attr-defined]
    for spec in sensors:
        if spec.modality != "rgb":
            continue
        frame = image_frame_id(spec, sensors)
        assert frame == spec.frame_id or frame in mounts.values()
        da3 = resolve_estimated_depth_camera(sensors, spec.name, hal_mode="real")
        if da3.optical_parent is not None:
            assert mounts[da3.optical_parent] == da3.cloud_frame == frame


def _compose(launch_module: object, monkeypatch: pytest.MonkeyPatch, **overrides: str) -> list[str]:
    """Node names of the real ``compose_runtime_graph``, every other arg at its default."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument

    for dep in ("openral_safety",):
        pytest.importorskip(dep, reason=f"{dep} is imported by compose_runtime_graph")
    # A real deploy refuses a ROS graph that reaches other hosts.
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.delenv("ROS_STATIC_PEERS", raising=False)
    context = LaunchContext()
    for action in launch_module.generate_launch_description().entities:  # type: ignore[attr-defined]
        if isinstance(action, DeclareLaunchArgument):
            default = "".join(s.perform(context) for s in action.default_value or [])
            context.launch_configurations[action.name] = overrides.get(action.name, default)
    actions = launch_module.compose_runtime_graph(context)  # type: ignore[attr-defined]
    return [n for a in actions if (n := getattr(a, "_Node__node_name", None))]


def test_real_launch_publishes_the_optical_child_once(
    launch_module: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#627: a real MakerMods launch publishes ``top_camera -> top_camera_optical_frame``.

    With DA3 estimated depth on ``top`` too, whose cloud frame is that same child, the
    estimated-depth leg reuses it instead of starting a second, same-named publisher.
    ``top`` is bound to a driver topic here: a real estimated-depth camera needs one.
    """
    import yaml

    manifest = yaml.safe_load((REPO_ROOT / "robots/makermods_metal/robot.yaml").read_text())
    top = next(s for s in manifest["sensors"] if s["name"] == "top")
    top["deploy_binding"] = {"backend": "ros2_image", "backend_params": {"topic": "/top/image"}}
    robot_yaml = tmp_path / "robot.yaml"
    robot_yaml.write_text(yaml.safe_dump(manifest))
    names = _compose(
        launch_module,
        monkeypatch,
        robot_yaml=str(robot_yaml),
        hal_mode="real",
        estimated_depth_camera="top",
    )
    optical = [n for n in names if n.endswith("_optical_frame")]
    assert optical == ["static_top_camera_to_top_camera_optical_frame"]


def test_sim_launch_leaves_the_optical_child_to_the_bridge(
    launch_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In sim the sensor bridge publishes the optical child; the launch adds none."""
    names = _compose(
        launch_module,
        monkeypatch,
        robot_yaml=str(REPO_ROOT / "robots/makermods_metal/robot.yaml"),
        hal_mode="sim",
    )
    assert not [n for n in names if n.endswith("_optical_frame")]
