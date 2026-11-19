"""`RizonDualRealHAL` and its bringup — the actuation path to two real Flexiv Rizon 4s.

Doubles only at the transport boundary (``publish_fn`` / ``state_fn``, the adapter's own
injection points); the manifest, URDF and controller config are the committed ones.

What these pin down:

- The committed manifest builds this HAL through ``build_hal``, and one 14-D action
  reaches **both** arm controllers, each with its own slice and joint names.
- ``/joint_states`` is matched by name: two controllers publish into it independently.
- The HAL, the bringup's controller config and the committed URDF agree on controller
  and joint names — a mismatch is a silent no-op on hardware (a controller that claims
  no joint the HAL commands, or a URDF joint ros2_control never exports).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

pytest.importorskip("lerobot", reason="openral_hal imports lerobot at package import")

from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, RobotDescription
from openral_hal import build_hal
from openral_hal.flexiv_rizon_real import RizonDualRealHAL
from openral_hal.ros_control import GroupedRosControlHAL

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "robots" / "rizon_dual" / "robot.yaml"
CONTROLLERS = REPO_ROOT / "packages/openral_hal_rizon/config/rizon_dual_controllers.yaml"
LEFT = [f"left_joint{i}" for i in range(1, 8)]
RIGHT = [f"right_joint{i}" for i in range(1, 8)]


@pytest.fixture(scope="module")
def desc() -> RobotDescription:
    return RobotDescription.from_yaml(str(MANIFEST))


def _hal(desc: RobotDescription, **kw: object) -> RizonDualRealHAL:
    hal = build_hal(desc, mode="real", transport=kw)
    assert isinstance(hal, RizonDualRealHAL)
    return hal


def test_manifest_builds_the_hal_with_two_controllers(desc: RobotDescription) -> None:
    hal = _hal(desc)
    assert hal.controller_names() == ["left_arm_controller", "right_arm_controller"]
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


def test_groups_must_cover_every_joint(desc: RobotDescription) -> None:
    with pytest.raises(ROSConfigError, match="cover"):
        GroupedRosControlHAL(desc, groups=[("left_arm_controller", 7)])


def test_bringup_controller_config_matches_the_hal(desc: RobotDescription) -> None:
    cfg = yaml.safe_load(CONTROLLERS.read_text(encoding="utf-8"))
    hal = _hal(desc)
    for name, joints in zip(hal.controller_names(), (LEFT, RIGHT), strict=True):
        assert name in cfg["controller_manager"]["ros__parameters"]
        assert cfg[name]["ros__parameters"]["joints"] == joints


def test_committed_urdf_declares_the_manifest_joints(desc: RobotDescription) -> None:
    urdf = MANIFEST.parent / "rizon_dual.urdf"
    root = ET.parse(urdf).getroot()
    revolute = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    assert revolute == [j.name for j in desc.joints]


def test_a_publish_failing_after_one_arm_took_its_target_stops_both(
    desc: RobotDescription,
) -> None:
    sent: list[str] = []

    def publish(topic: str, msg: dict[str, object]) -> None:
        if topic.startswith("/right"):
            raise RuntimeError("publisher gone")
        sent.append(topic)

    hal = _hal(desc, publish_fn=publish)
    hal.connect()
    before = hal.last_committed_tick
    with pytest.raises(ROSEStopRequested):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.5] * 14])
        )
    assert sent == ["/left_arm_controller/joint_trajectory"]
    assert hal.last_committed_tick == before  # the half-applied tick is never committed
    with pytest.raises(ROSRuntimeError):  # disconnected: nothing more reaches either arm
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[0.5] * 14])
        )


def test_committed_urdf_keeps_every_link_inertia() -> None:
    """flexiv_description emits bare origin/mass/inertia; parsers read them only in <inertial>."""
    root = ET.parse(MANIFEST.parent / "rizon_dual.urdf").getroot()
    for link in root.findall("link"):
        assert link.find("mass") is None, f"{link.get('name')}: mass outside <inertial>"
    assert len(root.findall("link/inertial/mass")) == 16
