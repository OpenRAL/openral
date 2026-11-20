"""``ComposedHAL`` wiring that needs no hardware and no ROS graph.

The parts are the real ``HeldJointsHAL`` and ``PlatformBaseHAL`` classes over the real
``robots/openamrobot_v2`` manifest, narrowed to the base and lift joints. Nothing here
connects a platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core import RobotDescription, ROSConfigError
from openral_core.exceptions import ROSCapabilityMismatch, ROSEStopRequested
from openral_hal.composed import ComposedHAL
from openral_hal.platform_base import PlatformBaseHAL
from openral_hal.resolver import vendor_publishes_urdf_joint_states

_ROBOTS = Path(__file__).resolve().parents[2] / "robots"
_BASE_LIFT = ("base_x", "base_y", "base_yaw", "lift_joint")


@pytest.fixture(scope="module")
def amr() -> RobotDescription:
    return RobotDescription.from_yaml(str(_ROBOTS / "openamrobot_v2" / "robot.yaml"))


@pytest.fixture(scope="module")
def base_lift(amr: RobotDescription) -> RobotDescription:
    return amr.model_copy(
        update={
            "joints": [j for j in amr.joints if j.name in _BASE_LIFT],
            "end_effectors": [],
        }
    )


def _parts(base_params: dict[str, object] | None = None) -> list[dict[str, object]]:
    return [
        {
            "entry": "openral_hal.platform_base:PlatformBaseHAL",
            "joints": ["base_x", "base_y", "base_yaw"],
            "params": base_params or {},
        },
        {
            "entry": "openral_hal.composed:HeldJointsHAL",
            "joints": ["lift_joint"],
            "params": {"positions": {"lift_joint": 0.0}},
        },
    ]


def test_a_misspelled_part_key_is_refused(base_lift: RobotDescription) -> None:
    """``param:`` for ``params:`` would start the part on its default topics."""
    parts = _parts()
    parts[0] = {**parts[0], "param": parts[0].pop("params")}
    with pytest.raises(ROSConfigError, match="param"):
        ComposedHAL(description=base_lift, parts=parts)


def test_staleness_and_overrides_reach_the_parts_that_take_them(
    base_lift: RobotDescription,
) -> None:
    hal = ComposedHAL(
        description=base_lift,
        parts=_parts({"odom_topic": "/platform/odom"}),
        staleness_limit_s=0.1,
        odom_topic="/override/odom",  # a deploy-run transport override wins over params
        left_can_interface="can9",  # taken by no part here: dropped, as build_hal drops it
    )
    base = hal._parts[0]
    assert isinstance(base, PlatformBaseHAL)
    assert base._staleness_limit_s == 0.1
    assert base._odom_topic == "/override/odom"


def test_build_hal_threads_the_manifest_staleness_window(amr: RobotDescription) -> None:
    from openral_hal import build_hal

    real = build_hal(amr, mode="real", transport={"require_can_links": False})
    assert isinstance(real, ComposedHAL)
    from openral_hal.openarm_real import OpenArmRealHAL

    arm = next(p for p in real._parts if isinstance(p, OpenArmRealHAL))
    assert arm._staleness_limit_s == amr.safety.joint_state_staleness_limit_s
    # The base keeps its own odometry window: 0.1 s would fail 10 Hz /odom.
    base = next(p for p in real._parts if isinstance(p, PlatformBaseHAL))
    assert base._staleness_limit_s == 0.5


def test_a_parts_own_staleness_window_beats_the_manifest_default(
    base_lift: RobotDescription,
) -> None:
    hal = ComposedHAL(
        description=base_lift, parts=_parts({"staleness_limit_s": 0.5}), staleness_limit_s=0.1
    )
    base = hal._parts[0]
    assert isinstance(base, PlatformBaseHAL)
    assert base._staleness_limit_s == 0.5


def test_the_base_refuses_a_joint_target_before_any_part_is_commanded(
    base_lift: RobotDescription,
) -> None:
    """``check_action`` runs on every part first; the base takes only twists."""
    from openral_core import Action, ControlMode

    hal = ComposedHAL(description=base_lift, parts=_parts())
    with pytest.raises(ROSConfigError, match="BODY_TWIST only"):
        hal.send_action(Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[[0.0] * 4]))


def test_the_manifest_real_base_is_odometry_only(amr: RobotDescription) -> None:
    """The OpenAMRobot platform owns its final base command (openral#497).

    The manifest gives the base part no ``cmd_vel_topic``, so a non-zero twist is
    refused before any part, the arm included, is commanded; a zero twist passes.
    """
    from openral_core import Action, ControlMode
    from openral_hal import build_hal

    real = build_hal(amr, mode="real", transport={"require_can_links": False})
    assert isinstance(real, ComposedHAL)
    base = next(p for p in real._parts if isinstance(p, PlatformBaseHAL))
    assert base._cmd_vel_topic is None

    def twist(vx: float) -> Action:
        return Action(
            control_mode=ControlMode.BODY_TWIST, horizon=1, body_twist=[(vx, 0, 0, 0, 0, 0)]
        )

    with pytest.raises(ROSCapabilityMismatch, match="odometry-only"):
        base.check_action(twist(0.1))
    assert base.check_action(twist(0.0)) == (0.0,) * 6
    hal = ComposedHAL(
        description=amr.model_copy(
            update={"joints": [j for j in amr.joints if j.name in _BASE_LIFT], "end_effectors": []}
        ),
        parts=_parts(),
    )
    with pytest.raises(ROSCapabilityMismatch):
        hal.send_action(twist(0.1))


def test_without_a_ros2_control_part_the_stop_surface_falls_back(
    base_lift: RobotDescription,
) -> None:
    """The lifecycle node reads these inside its e-stop handler; they must not raise."""
    hal = ComposedHAL(description=base_lift, parts=_parts())
    assert hal.last_stop_report is None  # "unproven"
    assert getattr(hal, "estop_recovery", "undeclared") == "undeclared"
    with pytest.raises(ROSConfigError, match="ros2_control"):
        hal.controller_names()
    with pytest.raises(ROSConfigError, match="ros2_control"):  # typed, not AttributeError
        hal.reset_estop()


def test_a_clean_stop_of_every_part_raises_estop_requested(base_lift: RobotDescription) -> None:
    hal = ComposedHAL(description=base_lift, parts=_parts())
    with pytest.raises(ROSEStopRequested):
        hal.estop()


def test_the_composite_publishes_its_own_urdf_joint_states(amr: RobotDescription) -> None:
    """The arm's broadcaster has no lift state, so robot_state_publisher reads the HAL's."""
    assert not vendor_publishes_urdf_joint_states(amr, mode="real")
    openarm = RobotDescription.from_yaml(str(_ROBOTS / "openarm" / "robot.yaml"))
    assert vendor_publishes_urdf_joint_states(openarm, mode="real")


def test_a_part_that_stops_itself_mid_send_stops_the_whole_robot(amr: RobotDescription) -> None:
    """The OpenArm part publishes its left arm, then a later controller's publish fails.

    The part stops its own controllers and raises; the composite must then stop every
    part, not let the error out as one part's affair. The arm's second ``estop`` (from
    the composite) is idempotent: one deactivate, the acknowledged report kept.
    """
    from openral_core import Action, ControlMode
    from openral_hal import build_hal
    from openral_hal.openarm_real import OpenArmRealHAL
    from openral_hal.sim_transport import SimTransport

    real = build_hal(amr, mode="real", transport={"require_can_links": False})
    assert isinstance(real, ComposedHAL)
    arm = next(p for p in real._parts if isinstance(p, OpenArmRealHAL))
    transport = SimTransport(n_joints=16, controllers=arm.controller_names())
    first = arm.command_topics()[0]

    def publish(topic: str, msg: dict[str, object]) -> None:
        if topic != first:
            raise RuntimeError("publisher gone")
        transport.publish(topic, msg)

    arm.attach_transport(publish, transport.state)
    arm.attach_controller_stop(transport)
    arm.connect()
    names = [j.name for j in arm.description.joints]
    with pytest.raises(ROSEStopRequested, match=f"robot '{amr.name}'"):  # the composite's
        real.send_action(
            Action(
                control_mode=ControlMode.JOINT_POSITION,
                horizon=1,
                joint_targets=[[0.1] * len(names)],
                joint_names=names,
            )
        )
    assert [op for op, _ in transport.switch_calls if op == "deactivate"] == ["deactivate"]
    report = arm.last_stop_report
    assert report is not None and report.stopped


def test_a_slot_that_shrinks_the_tick_never_reaches_a_part(base_lift: RobotDescription) -> None:
    """The composite's stager refuses a resized tick before any part is dispatched (#683).

    Releasing it at the smaller size would hand the parts a partial tick on the real
    actuation path. Nothing is connected: the refusal comes before routing.
    """
    from openral_core.exceptions import ROSRuntimeError
    from openral_core.schemas import Action, ControlMode

    hal = ComposedHAL(description=base_lift, parts=_parts())
    lift = Action(
        control_mode=ControlMode.JOINT_POSITION,
        horizon=1,
        joint_targets=[[0.0, 0.0, 0.0, 0.0]],
        joint_names=["lift_joint"],
        tick_index=1,
        tick_group_size=3,
        runner_session_id=0xA,
    )
    hal.send_action(lift)
    with pytest.raises(ROSRuntimeError, match="changed its declared size from 3 to 2"):
        hal.send_action(lift.model_copy(update={"tick_group_size": 2}))
    assert hal.last_committed_tick == 0
    assert hal.discard_staged_slots() == 0  # the tick was discarded, not left staged
