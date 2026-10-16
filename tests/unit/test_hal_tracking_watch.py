"""``TrackingWatch``: a position-commanded joint that does not follow its command faults.

Replays real OpenArm data from the Orin cell (``fixtures/openarm_tracking_20261016.npz``):
run 2, the drives energised and the arms following the policy, must stay quiet; run 3, the
drives limp while the stack kept commanding, must fault within one window. Plus the HAL
node's wiring (a fault publishes ``/openral/estop``) and the launch arming it for a real
HAL only. Real manifests and recorded hardware data; no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from openral_core import RobotDescription
from openral_core.schemas import Action, ControlMode
from openral_hal._tracking_watch import TrackingWatch, joint_row_targets

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "tests" / "unit" / "fixtures" / "openarm_tracking_20261016.npz"


def _openarm() -> RobotDescription:
    return RobotDescription.from_yaml(str(_REPO / "robots" / "openarm" / "robot.yaml"))


def _command(names: list[str], row: list[float]) -> Action:
    return Action(control_mode=ControlMode.JOINT_POSITION, joint_targets=[row], joint_names=names)


def _replay(run: str) -> tuple[Any, float]:
    """Feed a recorded run through the launch's watch; (first fault, seconds replayed)."""
    z = np.load(_FIXTURE)
    names = [str(n) for n in z[f"{run}_names"]]
    watch = TrackingWatch(_openarm(), tolerance_rad=0.05, window_s=1.0)
    events = sorted(
        [(float(t), 0, i) for i, t in enumerate(z[f"{run}_cmd_t"])]
        + [(float(t), 1, i) for i, t in enumerate(z[f"{run}_state_t"])]
    )
    for t, kind, i in events:
        if kind == 0:
            watch.observe_command(_command(names, [float(v) for v in z[f"{run}_cmd_q"][i]]))
        elif fault := watch.observe_state(names, [float(v) for v in z[f"{run}_state_q"][i]], t):
            return fault, t
    return None, float(z[f"{run}_state_t"][-1])


def test_a_real_run_that_tracked_stays_quiet() -> None:
    fault, replayed_s = _replay("run2")
    assert fault is None
    assert replayed_s > 4.5


def test_a_real_run_with_limp_drives_faults_within_one_window() -> None:
    fault, at_s = _replay("run3")
    assert fault is not None
    assert at_s <= 1.1
    assert fault.error > 0.05
    assert fault.progress < 0.2 * fault.error


def test_grippers_are_not_watched() -> None:
    watch = TrackingWatch(_openarm(), tolerance_rad=0.05, window_s=1.0)
    assert "left_gripper" not in watch.watched
    assert "right_gripper" not in watch.watched
    assert {"left_joint1", "right_joint7"} <= watch.watched


def _held(watch: TrackingWatch, joint: str, positions: list[float], dt: float) -> Any:
    fault = None
    for k, q in enumerate(positions):
        fault = fault or watch.observe_state([joint], [q], k * dt)
    return fault


def test_closing_the_error_is_progress_and_sagging_away_is_not() -> None:
    watch = TrackingWatch(_openarm(), tolerance_rad=0.05, window_s=1.0)
    watch.observe_command(_command(["left_joint4"], [0.5]))
    converging = [0.5 * (1 - 0.8**k) for k in range(60)]  # slow but steady, 30 Hz
    assert _held(watch, "left_joint4", converging, 1 / 30) is None

    watch.clear()
    watch.observe_command(_command(["left_joint4"], [0.5]))
    sagging = [-0.01 * k for k in range(60)]  # moving, but away from the target
    fault = _held(watch, "left_joint4", sagging, 1 / 30)
    assert fault is not None and fault.progress < 0


def test_clear_forgets_targets() -> None:
    watch = TrackingWatch(_openarm(), tolerance_rad=0.05, window_s=0.5)
    watch.observe_command(_command(["right_joint2"], [0.4]))
    watch.clear()
    assert _held(watch, "right_joint2", [0.0] * 40, 1 / 30) is None


def test_full_dof_and_compact_rows_place_the_same_target() -> None:
    order = [j.name for j in _openarm().joints]
    padded = [0.0] * len(order)
    padded[order.index("right_joint3")] = -0.2
    assert joint_row_targets(padded, ["right_joint3"], order) == {"right_joint3": -0.2}
    assert joint_row_targets([-0.2], ["right_joint3"], order) == {"right_joint3": -0.2}
    assert joint_row_targets([0.1, 0.2], ["right_joint3"], order) is None


def test_bad_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="TrackingWatch"):
        TrackingWatch(_openarm(), tolerance_rad=0.0, window_s=1.0)


@pytest.mark.usefixtures("_rclpy_ctx")
def test_the_hal_node_raises_estop_on_a_fault() -> None:
    """The real lifecycle node: a joint held off its command publishes /openral/estop."""
    pytest.importorskip("rclpy")
    import time

    import rclpy
    from openral_core.schemas import JointState
    from openral_hal.lifecycle import _ManifestHALLifecycleNode
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from std_msgs.msg import Empty

    node = _ManifestHALLifecycleNode("t_tracking_watch")
    listener = rclpy.create_node("t_tracking_listener")
    try:
        node.set_parameters(
            [
                Parameter("robot_yaml", value=str(_REPO / "robots" / "openarm" / "robot.yaml")),
                Parameter("hal_mode", value="sim"),
                Parameter("tracking_fault_window_s", value=0.2),
            ]
        )
        node._hal = node._create_hal()
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        got: list[int] = []
        listener.create_subscription(Empty, "/openral/estop", lambda _m: got.append(1), qos)
        node._arm_tracking_watch(qos)
        assert node._tracking is not None
        node._tracking.observe_command(_command(["left_joint4"], [0.5]))
        names = [j.name for j in node._hal.description.joints]
        still = JointState(name=names, position=[0.0] * len(names), stamp_ns=0)
        node._judge_tracking(still)
        time.sleep(0.25)
        node._judge_tracking(still)
        deadline = time.monotonic() + 2.0
        while not got and time.monotonic() < deadline:
            rclpy.spin_once(listener, timeout_sec=0.05)
        assert got, "no /openral/estop after a tracking fault"
    finally:
        listener.destroy_node()
        node.destroy_node()
