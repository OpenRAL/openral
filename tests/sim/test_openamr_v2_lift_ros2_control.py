"""The OpenAMR lift as a ros2_control part: ready for when upstream exposes it on real.

Upstream (``openamr-upperbody-sw`` ``ros2_control.xacro``, issue #17) drives ``lift_joint``
through ``lift_controller``, a position ``JointTrajectoryController``, in its mock and
Gazebo profiles; its real profile has no lift interface yet (the lift node firmware owns
it), which is why the manifest's real composite holds the lift (``HeldJointsHAL``). The
day the lift gets an interface, the swap is one ``parts`` entry: the stock
``RosControlHAL`` on ``lift_controller``, beside the arm, on the same controller_manager.

This brings that up for real: a ``controller_manager`` with a ``joint_state_broadcaster``
and two real ``JointTrajectoryController`` s (one arm, the lift) over
``mock_components/GenericSystem`` (upstream's own mock profile, the boundary double
CLAUDE.md §1.11 allows), wired the way the HAL lifecycle node wires a ros2_control HAL:
one ``RosControlTransport`` for the whole ``ComposedHAL``. Skipped without ROS 2.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from openral_core import Action, ControlMode, RobotDescription
from openral_core.exceptions import ROSEStopRequested

from tests.integration._ros2_control_stack import (
    bring_up,
    driver_node,
    unavailable,
    wait_until,
    write_controller_config,
    write_generic_system_urdf,
)

pytestmark = pytest.mark.sim

_MANIFEST = Path(__file__).resolve().parents[2] / "robots" / "openamr_v2" / "robot.yaml"
_ARM = [f"left_joint{i}" for i in range(1, 8)]
_LIFT = ["lift_joint"]
_SKIP_REASON = unavailable()


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    if _SKIP_REASON is not None:
        pytest.skip(_SKIP_REASON)
    tmp = tmp_path_factory.mktemp("openamr_lift")
    urdf = write_generic_system_urdf(tmp / "amr.urdf", name="amr", joints=[*_LIFT, *_ARM])
    config = write_controller_config(
        tmp / "controllers.yaml", {"left_arm_controller": _ARM, "lift_controller": _LIFT}
    )
    with bring_up(
        tmp, urdf=urdf, config=config, controllers=["left_arm_controller", "lift_controller"]
    ):
        yield


def _composite():  # type: ignore[no-untyped-def]  # reason: returns (ComposedHAL, transport class)
    """openamr_v2 narrowed to the lift and the left arm, each a stock ``RosControlHAL``."""
    from openral_hal.composed import ComposedHAL
    from openral_hal.ros_control_transport import RosControlTransport

    robot = RobotDescription.from_yaml(str(_MANIFEST))
    keep = {*_LIFT, *_ARM}
    description = robot.model_copy(
        update={
            "joints": [j for j in robot.joints if j.name in keep],
            "end_effectors": [],
            "base_joints": None,
        }
    )
    hal = ComposedHAL(
        description=description,
        parts=[
            {
                "entry": "openral_hal.ros_control:RosControlHAL",
                "joints": _ARM,
                "params": {"controller_name": "left_arm_controller"},
            },
            {
                "entry": "openral_hal.ros_control:RosControlHAL",
                "joints": _LIFT,
                "params": {"controller_name": "lift_controller"},
            },
        ],
        staleness_limit_s=robot.safety.joint_state_staleness_limit_s,
    )
    return hal, RosControlTransport


def test_the_lift_moves_through_lift_controller_beside_the_arm(stack: None) -> None:
    """One transport, two controllers: a whole row moves the lift and the arm by name."""
    hal, transport_cls = _composite()
    names = [j.name for j in hal.description.joints]
    with driver_node("openamr_lift_drive") as (node, spin):
        # As ManifestHALLifecycleNode._attach_ros_control_transport wires a real HAL.
        transport = transport_cls(
            node,
            command_topics=list(hal.command_bindings()),
            joint_names=hal.ros2_control_joint_names(),
            joint_state_topic=hal.joint_state_topic,
            command_kinds=hal.command_bindings(),
            controller_names=hal.controller_names(),
        )
        hal.attach_transport(transport.publish, transport.state, transport.last_arrival)
        hal.attach_controller_stop(transport)
        hal.connect()
        assert sorted(hal.controller_names()) == ["left_arm_controller", "lift_controller"]
        assert wait_until(spin, lambda: transport.last_arrival() != 0.0, 15.0)
        assert transport.missing_joints() == []
        spin(0.5)

        target = [0.0] * len(names)
        target[names.index("lift_joint")] = 0.2  # L1175-ish, inside [0, 0.35]
        target[names.index("left_joint2")] = -0.4
        hal.send_action(
            Action(control_mode=ControlMode.JOINT_POSITION, horizon=1, joint_targets=[target])
        )

        def _arrived() -> bool:
            got = hal.read_state().position
            return max(abs(a - b) for a, b in zip(got, target, strict=True)) < 1e-3

        assert wait_until(spin, _arrived, 10.0), list(hal.read_state().position)
        state = dict(zip(hal.read_state().name, hal.read_state().position, strict=True))
        assert state["lift_joint"] == pytest.approx(0.2, abs=1e-3)

        # /openral/estop deactivates BOTH controllers through the one seam, and the merged
        # report is acknowledged only because both were.
        stop = threading.Event()

        def _spin_bg() -> None:
            while not stop.is_set():
                spin(0.05)

        spinner = threading.Thread(target=_spin_bg, daemon=True)
        spinner.start()
        try:
            with pytest.raises(ROSEStopRequested):
                hal.estop()
        finally:
            stop.set()
            spinner.join(timeout=5.0)
        report = hal.last_stop_report
        assert report is not None and report.stopped, report
        assert set(report.controllers) == {"left_arm_controller", "lift_controller"}
        assert all(v != "active" for v in report.controller_states.values()), report
        time.sleep(0.1)
