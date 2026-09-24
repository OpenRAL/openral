"""The ros2_control downstream e-stop seam, proven on the generic adapter (issue #295).

`/openral/estop` used to end at `RosControlHAL.estop()` flipping a flag: the controller that
already held a trajectory was never told anything, so the physical stop was unproven for
every ros2_control robot. The HAL now deactivates its controllers through an attached
`ControllerStopSeam` and records whether that was acknowledged.

The seam here is `SimTransport`, which is the in-repo controller simulator, not a mock: it
implements the same protocol `RosControlTransport` does on a live `controller_manager`, and
it reproduces the two verified facts about a deactivated `JointTrajectoryController` — it
drops trajectories it is sent and writes nothing until re-activated. The live counterpart of
this file is `tests/integration/test_real_hal_estop_ros2_control_live.py`.
"""

from __future__ import annotations

import pytest
from openral_core.exceptions import ROSConfigError, ROSEStopRequested, ROSRuntimeError
from openral_core.schemas import (
    Action,
    ActionSpec,
    ControlMode,
    EmbodimentKind,
    JointSpec,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
)
from openral_hal.protocol import EStopRecovery, LifecycleEStopHAL, ResettableLifecycleEStopHAL
from openral_hal.ros_control import (
    ControllerStoppable,
    ControllerStopSeam,
    DownstreamStopReporting,
    RosControlHAL,
)
from openral_hal.sim_transport import SimTransport

_CONTROLLER = "arm_controller"


def _description(n_joints: int = 2) -> RobotDescription:
    """A minimal two-joint ``RobotDescription`` for the generic adapter."""
    return RobotDescription(
        name="seam_robot",
        embodiment_kind=EmbodimentKind.MANIPULATOR,
        joints=[
            JointSpec(
                name=f"j{i}",
                joint_type=JointType.REVOLUTE,
                parent_link="base" if i == 0 else f"link_{i - 1}",
                child_link=f"link_{i}",
            )
            for i in range(n_joints)
        ],
        capabilities=RobotCapabilities(supported_control_modes=[ControlMode.JOINT_POSITION]),
        safety=SafetyEnvelope(),
        # A ros2_control HAL refuses a manifest without a control rate (#303).
        action_spec=ActionSpec(dim=n_joints, control_freq_hz=30.0),
    )


class _ResettableHAL(RosControlHAL):
    """The generic adapter with the OpenArm's policy, to exercise `reset_estop`."""

    estop_recovery = EStopRecovery.RESETTABLE


def _wired(
    *, controllers: list[str] | None = None, cls: type[RosControlHAL] = RosControlHAL
) -> tuple[RosControlHAL, SimTransport]:
    """A connected ``RosControlHAL`` with a ``SimTransport`` as drive and stop seam."""
    transport = SimTransport(n_joints=2, controllers=controllers)
    hal = cls(_description(), controller_name=_CONTROLLER)
    hal.attach_transport(transport.publish, transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    return hal, transport


def _move(v: float) -> Action:
    """A one-step ``JOINT_POSITION`` action to ``(v, -v)``."""
    return Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[[v, -v]])


def test_the_generic_adapter_declares_the_lifecycle_contract() -> None:
    """The generic adapter declares the lifecycle contract."""
    hal = RosControlHAL(_description(), controller_name=_CONTROLLER)
    assert isinstance(hal, LifecycleEStopHAL)
    assert isinstance(hal, ControllerStoppable)
    assert isinstance(hal, DownstreamStopReporting)
    assert hal.estop_recovery is EStopRecovery.RESTART_REQUIRED
    assert hal.controller_names() == [_CONTROLLER]
    assert hal.vendor_stop_services() == [] and hal.vendor_stop_topics() == []
    assert hal.last_stop_report is None


def test_estop_deactivates_the_controller_and_the_report_is_acknowledged() -> None:
    """Estop deactivates the controller and the report is acknowledged."""
    hal, transport = _wired(controllers=[_CONTROLLER])
    hal.send_action(_move(0.3))
    assert transport.state()["position"] == [0.3, -0.3]

    with pytest.raises(ROSEStopRequested, match="downstream stop acknowledged"):
        hal.estop()

    assert transport.switch_calls == [("deactivate", (_CONTROLLER,))]
    assert transport.controller_state(_CONTROLLER) == "inactive"
    report = hal.last_stop_report
    assert report is not None
    assert report.stopped
    assert report.controllers == (_CONTROLLER,)
    assert report.controller_states == {_CONTROLLER: "inactive"}
    assert report.fields()["downstream_stop"] == "acknowledged"


def test_after_the_stop_no_command_can_reach_the_controller_from_either_side() -> None:
    """Both layers hold: the HAL refuses, and a stray publish is dropped by the controller."""
    hal, transport = _wired(controllers=[_CONTROLLER])
    hal.send_action(_move(0.3))
    with pytest.raises(ROSEStopRequested):
        hal.estop()

    with pytest.raises(ROSRuntimeError):
        hal.send_action(_move(0.9))
    # A publisher that bypassed the HAL (the controller's own topic) is dropped too.
    transport.publish(f"/{_CONTROLLER}/joint_trajectory", {"joint_targets": [[0.9, -0.9]]})
    assert transport.state()["position"] == [0.3, -0.3]
    assert len(transport.dropped_calls) == 1


def test_a_controller_the_manager_does_not_list_is_an_unacknowledged_stop() -> None:
    """STRICT semantics: a switch that cannot move every controller is not a stop."""
    hal, _transport = _wired(controllers=["some_other_controller"])
    with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
        hal.estop()
    report = hal.last_stop_report
    assert report is not None and not report.stopped
    assert "not loaded" in report.detail
    assert report.fields()["downstream_stop"] == "unacknowledged"
    # The local latch still holds regardless.
    with pytest.raises(ROSRuntimeError):
        hal.send_action(_move(0.1))


def test_without_a_seam_the_stop_is_reported_unproven_never_as_stopped() -> None:
    """Without a seam the stop is reported unproven never as stopped."""
    transport = SimTransport(n_joints=2)
    hal = RosControlHAL(_description(), controller_name=_CONTROLLER)
    hal.attach_transport(transport.publish, transport.state)
    hal.connect()
    with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
        hal.estop()
    report = hal.last_stop_report
    assert report is not None and not report.stopped
    assert "no controller stop seam" in report.detail
    assert transport.switch_calls == []


def test_attaching_something_that_is_not_a_seam_is_refused_at_wire_up() -> None:
    """Attaching something that is not a seam is refused at wire up."""
    hal = RosControlHAL(_description(), controller_name=_CONTROLLER)

    class NotASeam:
        pass

    with pytest.raises(ROSConfigError, match="ControllerStopSeam"):
        hal.attach_controller_stop(NotASeam())  # type: ignore[arg-type]  # reason: the refusal is the test


def test_restart_required_refuses_an_in_process_reset() -> None:
    """Restart required refuses an in process reset."""
    hal, _ = _wired(controllers=[_CONTROLLER])
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    with pytest.raises(ROSRuntimeError, match="in-process reset is forbidden"):
        hal.reset_estop()
    with pytest.raises(ROSRuntimeError):
        hal.send_action(_move(0.1))


def test_resettable_reconnects_only_after_the_controllers_are_confirmed_active() -> None:
    """Resettable reconnects only after the controllers are confirmed active."""
    hal, transport = _wired(controllers=[_CONTROLLER], cls=_ResettableHAL)
    assert isinstance(hal, ResettableLifecycleEStopHAL)
    with pytest.raises(ROSEStopRequested):
        hal.estop()
    assert transport.controller_state(_CONTROLLER) == "inactive"

    hal.reset_estop()

    assert transport.switch_calls[-1] == ("activate", (_CONTROLLER,))
    assert transport.controller_state(_CONTROLLER) == "active"
    assert hal.last_stop_report is None
    hal.send_action(_move(0.5))
    assert transport.state()["position"] == [0.5, -0.5]


def test_a_refused_reactivation_leaves_the_hal_disconnected() -> None:
    """The latch can never clear ahead of the vendor reset (issue #295 acceptance)."""
    transport = SimTransport(n_joints=2, controllers=[_CONTROLLER])
    hal = _ResettableHAL(_description(), controller_name=_CONTROLLER)
    hal.attach_transport(transport.publish, transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()
    with pytest.raises(ROSEStopRequested):
        hal.estop()

    # Simulate the manager unloading the controller between stop and reset.
    transport._controller_states.pop(_CONTROLLER)
    with pytest.raises(ROSRuntimeError, match="re-activation not acknowledged"):
        hal.reset_estop()
    with pytest.raises(ROSRuntimeError):
        hal.send_action(_move(0.1))


def test_sim_transport_satisfies_the_seam_protocol_structurally() -> None:
    """Sim transport satisfies the seam protocol structurally."""
    assert isinstance(SimTransport(n_joints=1), ControllerStopSeam)


class _VendorStopHAL(RosControlHAL):
    """The generic adapter with a vendor step, to prove the step still runs after a fault."""

    def vendor_stop_topics(self) -> list[str]:
        """Declare the halt topic so the seam creates its publisher."""
        return ["/vendor/halt"]

    def _vendor_stop(self, seam: ControllerStopSeam) -> str:
        """Publish the halt topic and return its label."""
        seam.publish_empty("/vendor/halt")
        return "/vendor/halt"


def test_a_plain_exception_from_the_seam_is_contained_in_the_report() -> None:
    """rclpy raises plain Exceptions (ShutdownException, InvalidHandle); they must not escape.

    An escaping exception would skip the vendor step, leave ``last_stop_report`` unassigned
    and break the Protocol's "estop() always raises ROSEStopRequested" promise — the
    lifecycle node would then log a bare "hardware estop failed" with no downstream verdict.
    """
    transport = SimTransport(
        n_joints=2,
        controllers=[_CONTROLLER],
        switch_faults={"deactivate": RuntimeError("executor was shut down")},
    )
    hal = _VendorStopHAL(_description(), controller_name=_CONTROLLER)
    hal.attach_transport(transport.publish, transport.state)
    hal.attach_controller_stop(transport)
    hal.connect()

    with pytest.raises(ROSEStopRequested, match="NOT acknowledged"):
        hal.estop()

    report = hal.last_stop_report
    assert report is not None and not report.stopped
    assert "RuntimeError: executor was shut down" in report.detail
    # The vendor step still ran after the controller step blew up.
    assert transport.empty_publishes == ["/vendor/halt"]
    assert report.vendor_stop == "/vendor/halt"
    with pytest.raises(ROSRuntimeError):
        hal.send_action(_move(0.1))
