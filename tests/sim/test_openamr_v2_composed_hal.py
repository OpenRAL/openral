"""``ComposedHAL`` over real parts: the OpenArm MuJoCo twin as the arm of openamr_v2.

The arm part is ``OpenArmMujocoHAL``, the real 16-joint OpenArm twin, unchanged. The
base and the lift are ``HeldJointsHAL`` parts, so every assertion about routing,
merging and refusal runs against real physics. No mocks. The real-hardware composite
(``OpenArmRealHAL`` + ``PlatformBaseHAL``) is checked for its wiring surface only;
it needs hardware to run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

try:
    import mujoco  # noqa: F401
except Exception as exc:  # mujoco's eager renderer probe can raise non-ImportError types
    _MUJOCO_ERROR: str | None = str(exc)
else:
    _MUJOCO_ERROR = None

from openral_core import Action, ControlMode, RobotDescription, ROSConfigError
from openral_core.exceptions import ROSEStopRequested
from openral_hal import build_hal
from openral_hal.composed import ComposedHAL

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_HELD = {"base_x": 0.5, "base_y": -0.25, "base_yaw": 0.3, "lift_joint": 0.2}

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(_MUJOCO_ERROR is not None, reason=f"mujoco unavailable: {_MUJOCO_ERROR}"),
]


@pytest.fixture(scope="module")
def description() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / "openamr_v2" / "robot.yaml"))


def _parts() -> list[dict[str, object]]:
    return [
        {
            "entry": "openral_hal.openarm:OpenArmMujocoHAL",
            "robot": "openarm",
            "params": {"gravity_enabled": False},
        },
        {
            "entry": "openral_hal.composed:HeldJointsHAL",
            "joints": ["base_x", "base_y", "base_yaw"],
            "params": {"positions": {k: _HELD[k] for k in ("base_x", "base_y", "base_yaw")}},
        },
        {
            "entry": "openral_hal.composed:HeldJointsHAL",
            "joints": ["lift_joint"],
            "params": {"positions": {"lift_joint": _HELD["lift_joint"]}},
        },
    ]


@pytest.fixture
def hal(description: RobotDescription) -> Iterator[ComposedHAL]:
    composed = ComposedHAL(description=description, parts=_parts())
    try:
        composed.connect()
    except ROSConfigError as exc:  # pinned OpenArm MJCF clone unreachable (offline host)
        pytest.skip(f"OpenArm twin unavailable: {exc}")
    yield composed
    composed.disconnect()


def _padded(names: list[str], values: list[float], robot: list[str]) -> list[float]:
    """A slot row as the runner sends it: robot dof, values at each joint's index."""
    row = [0.0] * len(robot)
    for name, value in zip(names, values, strict=True):
        row[robot.index(name)] = value
    return row


def _openarm_tick(
    robot: list[str], tick: int, left: list[float], right: list[float], grip: float
) -> list[Action]:
    """The four ADR-0102 slots a 16-D OpenArm rSkill emits for one tick."""
    common = {"horizon": 1, "tick_index": tick, "tick_group_size": 4, "runner_session_id": 7}
    left_names = [f"left_joint{i}" for i in range(1, 8)]
    right_names = [f"right_joint{i}" for i in range(1, 8)]
    return [
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[_padded(left_names, left, robot)],
            joint_names=left_names,
            **common,
        ),
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[grip],
            ee_name="left_gripper",
            **common,
        ),
        Action(
            control_mode=ControlMode.JOINT_POSITION,
            joint_targets=[_padded(right_names, right, robot)],
            joint_names=right_names,
            **common,
        ),
        Action(
            control_mode=ControlMode.GRIPPER_POSITION,
            gripper=[-grip],
            ee_name="right_gripper",
            **common,
        ),
    ]


def test_state_merges_every_part_by_name(hal: ComposedHAL) -> None:
    state = hal.read_state()
    assert state.name == [j.name for j in hal.description.joints]
    for name, value in _HELD.items():
        assert state.position[state.name.index(name)] == value


def test_openarm_slot_tick_drives_the_arm_part(hal: ComposedHAL) -> None:
    """A 16-D OpenArm tick reaches the OpenArm twin as its own 4-slot group."""
    left = [0.0, -0.4, 0.0, 0.9, 0.0, 0.0, 0.0]
    right = [0.0, 0.4, 0.0, 0.9, 0.0, 0.0, 0.0]
    robot = [j.name for j in hal.description.joints]
    for tick in range(1, 91):  # 3 s at 30 Hz
        for slot in _openarm_tick(robot, tick, left, right, 0.5):
            hal.send_action(slot)
    state = hal.read_state()
    got = dict(zip(state.name, state.position, strict=True))
    assert got["left_joint2"] == pytest.approx(-0.4, abs=0.02)
    assert got["right_joint4"] == pytest.approx(0.9, abs=0.02)
    assert got["left_gripper"] == pytest.approx(0.5, abs=0.05)
    assert got["lift_joint"] == _HELD["lift_joint"]
    assert hal.last_committed_tick == 90
    applied = hal.last_applied_action
    assert applied is not None and applied.joint_targets is not None
    composed = dict(zip(robot, applied.joint_targets[-1], strict=True))
    assert composed["left_joint2"] == -0.4  # the slot's target
    assert composed["lift_joint"] == _HELD["lift_joint"]  # uncommanded: measured


def test_whole_robot_row_is_split_by_name(hal: ComposedHAL) -> None:
    names = [j.name for j in hal.description.joints]
    row = list(hal.read_state().position)
    row[names.index("left_joint4")] = 1.0
    action = Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
    for _ in range(90):
        hal.send_action(action)
    assert hal.read_state().position[names.index("left_joint4")] == pytest.approx(1.0, abs=0.02)


def test_a_held_joint_refuses_to_move(hal: ComposedHAL) -> None:
    names = [j.name for j in hal.description.joints]
    row = list(hal.read_state().position)
    row[names.index("lift_joint")] = 0.3
    with pytest.raises(ROSConfigError, match="no command"):
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[row])
        )


def test_estop_stops_every_part_and_raises(hal: ComposedHAL) -> None:
    arm = hal._parts[0]
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert not getattr(arm, "_connected", True)  # the OpenArm twin released its model


def test_joint_ownership_must_be_complete_and_disjoint(description: RobotDescription) -> None:
    parts = _parts()
    with pytest.raises(ROSConfigError, match="no part owns"):
        ComposedHAL(description=description, parts=parts[:2])
    overlap = [*parts, {**parts[2]}]
    with pytest.raises(ROSConfigError, match="owned by two parts"):
        ComposedHAL(description=description, parts=overlap)


def test_real_composite_exposes_the_openarm_ros2_control_seam(
    description: RobotDescription,
) -> None:
    """build_hal(real) gives the lifecycle node the OpenArm's own transport surface."""
    from openral_hal.ros_control import ControllerStoppable
    from openral_hal.ros_control_transport import RosControlDrivable

    real = build_hal(description, mode="real")
    assert isinstance(real, ComposedHAL)
    assert isinstance(real, RosControlDrivable)
    assert isinstance(real, ControllerStoppable)
    openarm = build_hal(
        RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml")), mode="real"
    )
    assert real.controller_names() == openarm.controller_names()  # type: ignore[attr-defined]
    assert real.ros2_control_joint_names() == openarm.ros2_control_joint_names()  # type: ignore[attr-defined]
