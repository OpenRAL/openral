"""The dual Rizon's real HAL and its bringup — the actuation path to two Flexiv Rizon 4s.

The robot is config-only: ``robots/rizon_dual/robot.yaml`` points ``hal.real`` at the
generic ``GroupedRosControlHAL`` and names its two controllers in
``hal.parameters.defaults.groups``. Doubles only at the transport boundary
(``publish_fn`` / ``state_fn``, the adapter's own injection points, and ``SimTransport``,
the in-repo controller simulator); the manifest, URDF, controller config and launch are the
committed ones.

What these pin down:

- The committed manifest builds the HAL through ``build_hal``, and one 14-D action
  reaches **both** arm controllers, each with its own slice and joint names.
- ``/joint_states`` is matched by name (two controllers publish into it independently),
  and a sample without names is refused, never read by position.
- A publish that fails after the left arm took its target stops both controllers and
  commits nothing; a second ``estop`` keeps the acknowledged report.
- The manifest's groups, the bringup's controllers yaml, its spawner list and the committed
  URDF agree — a mismatch is a silent no-op on hardware.
"""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

pytest.importorskip("lerobot", reason="openral_hal imports lerobot at package import")

from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription
from openral_hal import build_hal
from openral_hal.ros_control import GroupedRosControlHAL
from openral_hal.sim_transport import SimTransport

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "robots" / "rizon_dual" / "robot.yaml"
BRINGUP = REPO_ROOT / "packages" / "openral_hal_rizon"
CONTROLLERS = BRINGUP / "config" / "rizon_dual_controllers.yaml"
LEFT = [f"left_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_joint{i}" for i in range(1, 8)]
ARM_CONTROLLERS = ["left_arm_controller", "right_arm_controller"]


@pytest.fixture(scope="module")
def desc() -> RobotDescription:
    return RobotDescription.from_yaml(str(MANIFEST))


def _hal(desc: RobotDescription, **kw: object) -> GroupedRosControlHAL:
    hal = build_hal(desc, mode="real", transport=kw)
    assert type(hal) is GroupedRosControlHAL
    return hal


def _transport() -> SimTransport:
    return SimTransport(n_joints=14, controllers=ARM_CONTROLLERS, joint_names=LEFT + RIGHT)


def _action(value: float, *, tick: int = 0) -> Action:
    return Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[value] * 14],
        tick_index=tick,
    )


def test_manifest_builds_the_hal_with_two_controllers(desc: RobotDescription) -> None:
    hal = _hal(desc)
    assert hal.controller_names() == ARM_CONTROLLERS
    assert hal.ros2_control_joint_names() == LEFT + RIGHT


def test_one_action_reaches_both_controllers_sliced_and_named(desc: RobotDescription) -> None:
    sent: list[tuple[str, dict[str, object]]] = []
    hal = _hal(desc, publish_fn=lambda t, m: sent.append((t, m)))
    hal.connect()
    hal.send_action(
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            horizon=1,
            joint_targets=[[float(i) for i in range(14)]],
        )
    )
    assert [t for t, _ in sent] == hal.command_topics()
    (_, left), (_, right) = sent
    assert left["joint_names"] == LEFT
    assert right["joint_names"] == RIGHT
    assert left["joint_targets"] == [[0.0, 1, 2, 3, 4, 5, 6]]
    assert right["joint_targets"] == [[7.0, 8, 9, 10, 11, 12, 13]]


def test_a_wrong_width_action_publishes_nothing(desc: RobotDescription) -> None:
    sent: list[object] = []
    hal = _hal(desc, publish_fn=lambda t, m: sent.append(t))
    hal.connect()
    with pytest.raises(ROSConfigError):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.0] * 7])
        )
    assert sent == []


def test_state_is_matched_by_name_and_a_missing_arm_is_an_error(desc: RobotDescription) -> None:
    names = RIGHT + LEFT  # the broadcaster's order is not the manifest's
    raw: dict[str, object] = {
        "name": names,
        "position": [float(i) for i in range(14)],
        "velocity": [0.0] * 14,
    }
    hal = _hal(desc, state_fn=lambda: raw)
    hal.connect()
    assert hal.read_state().position[:7] == [7.0, 8, 9, 10, 11, 12, 13]
    raw["name"] = RIGHT
    with pytest.raises(ROSRuntimeError, match="left_joint1"):
        hal.read_state()


def test_a_sample_without_names_is_refused_not_read_by_position(desc: RobotDescription) -> None:
    hal = _hal(desc, state_fn=lambda: {"position": [0.0] * 14, "velocity": [0.0] * 14})
    hal.connect()
    with pytest.raises(ROSRuntimeError, match="no joint names"):
        hal.read_state()


@pytest.mark.parametrize(
    "groups",
    [
        [("left_arm_controller", 7)],  # short
        [("left_arm_controller", 14), ("right_arm_controller", 0)],  # zero-width
        [("arm_controller", 7), ("arm_controller", 7)],  # duplicate
    ],
)
def test_groups_must_be_distinct_and_cover_every_joint(
    desc: RobotDescription, groups: list[tuple[str, int]]
) -> None:
    with pytest.raises(ROSConfigError, match="cover"):
        GroupedRosControlHAL(desc, groups=groups)


def test_a_publish_failing_after_one_arm_took_its_target_stops_both(
    desc: RobotDescription,
) -> None:
    transport = _transport()

    def publish(topic: str, msg: dict[str, object]) -> None:
        if topic.startswith("/right"):
            raise RuntimeError("publisher gone")
        transport.publish(topic, msg)

    hal = _hal(desc, publish_fn=publish, state_fn=transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    with pytest.raises(ROSEStopRequested):
        hal.send_action(_action(0.5, tick=1))
    assert [t for t, _ in transport.calls] == ["/left_arm_controller/joint_trajectory"]
    assert all(transport.controller_state(n) == "inactive" for n in ARM_CONTROLLERS)
    assert hal.last_committed_tick == 0  # tick 1 never committed
    assert hal.last_applied_action is None
    with pytest.raises(ROSRuntimeError):  # disconnected: nothing more reaches either arm
        hal.send_action(_action(0.5, tick=2))


def test_a_second_estop_keeps_the_acknowledged_report(desc: RobotDescription) -> None:
    """A STRICT manager refuses re-deactivating inactive controllers: never ask it twice."""
    transport = _transport()
    hal = _hal(desc, publish_fn=transport.publish, state_fn=transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    for _ in range(2):
        with pytest.raises(ROSEStopRequested):
            hal.estop()
    assert [op for op, _ in transport.switch_calls if op == "deactivate"] == ["deactivate"]
    report = hal.last_stop_report
    assert report is not None and report.stopped


def test_an_unacknowledged_estop_is_retried(desc: RobotDescription) -> None:
    hal = _hal(desc)  # no stop seam: the first stop cannot be acknowledged
    hal.connect()
    with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
        hal.estop()
    transport = _transport()
    hal.attach_controller_stop(transport)
    with pytest.raises(ROSEStopRequested, match="downstream stop acknowledged"):
        hal.estop()
    assert all(transport.controller_state(n) == "inactive" for n in ARM_CONTROLLERS)


def test_bringup_controller_config_matches_the_manifest(desc: RobotDescription) -> None:
    cfg = yaml.safe_load(CONTROLLERS.read_text(encoding="utf-8"))
    hal = _hal(desc)
    for name, joints in zip(hal.controller_names(), (LEFT, RIGHT), strict=True):
        assert name in cfg["controller_manager"]["ros__parameters"]
        assert cfg[name]["ros__parameters"]["joints"] == joints


def test_bringup_spawns_exactly_the_manifest_controllers(desc: RobotDescription) -> None:
    for dep in ("launch", "launch_ros"):
        pytest.importorskip(dep, reason=f"{dep} is a module-level import of the launch file")
    spec = importlib.util.spec_from_file_location(
        "rizon_real_bringup", BRINGUP / "launch" / "real_bringup.launch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spawned = list(module._CONTROLLERS)
    assert spawned == ["joint_state_broadcaster", *_hal(desc).controller_names()]


def test_committed_urdf_declares_the_manifest_joints(desc: RobotDescription) -> None:
    urdf = MANIFEST.parent / "rizon_dual.urdf"
    root = ET.parse(urdf).getroot()
    revolute = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    assert revolute == [j.name for j in desc.joints]


def test_committed_urdf_keeps_every_link_inertia() -> None:
    """flexiv_description emits bare origin/mass/inertia; parsers read them only in <inertial>."""
    root = ET.parse(MANIFEST.parent / "rizon_dual.urdf").getroot()
    inertials = root.findall("link/inertial")
    assert len(inertials) == 16
    for link in root.findall("link"):
        for tag in ("origin", "mass", "inertia"):
            assert link.find(tag) is None, f"{link.get('name')}: {tag} outside <inertial>"
    for inertial in inertials:
        assert [c.tag for c in inertial] == ["origin", "mass", "inertia"]
