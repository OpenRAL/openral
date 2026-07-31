"""Full-graph HIL gate for the Galaxea A1 deployment path.

Run this test *inside* an already-running ``openral deploy run`` container.
The ROS 1 sidecar must be running separately. Unlike the HAL-only A1 HIL
fixture, this gate proves the standard OpenRAL control path:

``candidate_action -> C++ safety kernel -> safe_action -> HAL -> ROS 1 relay``.

Environment:
    GALAXEA_A1_DEPLOY_HIL: Must be ``"1"`` to connect to the live ROS 2 graph.
    GALAXEA_A1_ALLOW_HOLD: Must also be ``"1"``. The only candidate action is
        the freshly measured current joint pose.
    GALAXEA_A1_ALLOW_NUDGE: When ``"1"``, moves ``arm_joint1`` by +0.01 rad
        through the same candidate-action path, bounds every joint's excursion,
        and returns to the measured start before e-stop.

The test is bounded and always publishes ``/openral/estop`` during teardown.
It requires the relay to begin LOCKED, the kernel and HAL diagnostics to be
healthy, the safe action to equal the candidate exactly, and every joint to
remain within one degree of its measured start.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION

if TYPE_CHECKING:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

_ENABLED = os.environ.get("GALAXEA_A1_DEPLOY_HIL", "0") == "1"
_ALLOW_HOLD = os.environ.get("GALAXEA_A1_ALLOW_HOLD", "0") == "1"
_ALLOW_NUDGE = os.environ.get("GALAXEA_A1_ALLOW_NUDGE", "0") == "1"
_ROS2_AVAILABLE = importlib.util.find_spec("rclpy") is not None
_EXPECTED_NAMES = tuple(joint.name for joint in GALAXEA_A1_DESCRIPTION.joints)
_FEEDBACK_TOLERANCE_RAD = float(
    GALAXEA_A1_DESCRIPTION.hal.parameters.defaults["feedback_limit_tolerance_rad"]
)
_MAX_HOLD_DRIFT_RAD = 0.0175
_GRAPH_TIMEOUT_S = 10.0
_HOLD_TIMEOUT_S = 10.0
_ESTOP_TIMEOUT_S = 5.0
_COMMAND_PERIOD_S = 0.05
_NUDGE_RAD = 0.01
_SETTLE_TOLERANCE_RAD = 0.008
_MOTION_TIMEOUT_S = 3.0
_MAX_EXCURSION_RAD = 0.025

pytestmark = [
    pytest.mark.skipif(
        not _ENABLED,
        reason="GALAXEA_A1_DEPLOY_HIL=1 is required for the live deploy-graph test.",
    ),
    pytest.mark.skipif(
        not _ALLOW_HOLD,
        reason="GALAXEA_A1_ALLOW_HOLD=1 is required for the current-pose hold.",
    ),
    pytest.mark.skipif(
        not _ROS2_AVAILABLE,
        reason="rclpy is unavailable; run this test inside the OpenRAL deploy image.",
    ),
]


@dataclass
class _Observed:
    """Latest real graph evidence collected by one single-threaded ROS node."""

    joint_state: JointState | None = None
    diagnostics: dict[str, DiagnosticStatus] = field(default_factory=dict)
    diagnostic_history: dict[str, list[DiagnosticStatus]] = field(default_factory=dict)
    safe_actions: list[ActionChunk] = field(default_factory=list)
    failures: list[FailureTrigger] = field(default_factory=list)

    def on_joint_state(self, message: JointState) -> None:
        self.joint_state = message

    def on_diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            self.diagnostics[status.name] = status
            self.diagnostic_history.setdefault(status.name, []).append(status)

    def on_safe_action(self, message: ActionChunk) -> None:
        self.safe_actions.append(message)

    def on_failure(self, message: FailureTrigger) -> None:
        self.failures.append(message)


def _diagnostic_values(status: DiagnosticStatus) -> dict[str, str]:
    return {item.key: item.value for item in status.values}


def _spin_until(node: Node, predicate: Callable[[], bool], timeout_s: float) -> bool:
    import rclpy

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def _ordered_positions(message: JointState) -> tuple[float, ...]:
    assert len(message.name) == len(set(message.name)), "duplicate joint name in /joint_states"
    by_name = dict(zip(message.name, message.position, strict=True))
    assert set(_EXPECTED_NAMES).issubset(by_name), (
        f"/joint_states missing A1 joints: {sorted(set(_EXPECTED_NAMES) - set(by_name))}"
    )
    values = tuple(float(by_name[name]) for name in _EXPECTED_NAMES)
    assert all(math.isfinite(value) for value in values)
    return values


def _project_feedback_to_command_limits(position: tuple[float, ...]) -> tuple[float, ...]:
    projected: list[float] = []
    for value, joint in zip(position, GALAXEA_A1_DESCRIPTION.joints, strict=True):
        lower, upper = joint.position_limits
        command = min(upper, max(lower, value))
        assert abs(command - value) <= _FEEDBACK_TOLERANCE_RAD, (
            f"{joint.name} feedback {value:.6f} is too far outside its command limits"
        )
        projected.append(command)
    return tuple(projected)


def test_galaxea_a1_full_deploy_current_hold_and_estop() -> None:
    """Prove a measured hold traverses the real kernel and relay unchanged."""
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from openral_msgs.msg import ActionChunk, FailureTrigger
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty

    rclpy.init()
    node = rclpy.create_node("openral_hil_galaxea_a1_deploy")
    observed = _Observed()
    control_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    safety_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    candidate_pub = node.create_publisher(ActionChunk, "/openral/candidate_action", control_qos)
    estop_pub = node.create_publisher(Empty, "/openral/estop", safety_qos)
    candidate: ActionChunk | None = None
    subscriptions = [
        node.create_subscription(
            JointState,
            "/joint_states",
            observed.on_joint_state,
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            DiagnosticArray,
            "/diagnostics",
            observed.on_diagnostics,
            10,
        ),
        node.create_subscription(
            ActionChunk,
            "/openral/safe_action",
            observed.on_safe_action,
            control_qos,
        ),
        node.create_subscription(
            FailureTrigger,
            "/openral/failure/safety",
            observed.on_failure,
            safety_qos,
        ),
    ]
    try:
        assert _spin_until(
            node,
            lambda: (
                observed.joint_state is not None
                and "openral_safety_kernel" in observed.diagnostics
                and "openral_hal_galaxea_a1" in observed.diagnostics
                and node.count_subscribers("/openral/candidate_action") >= 1
                and node.count_subscribers("/openral/estop") >= 1
                and node.count_publishers("/openral/safe_action") >= 1
            ),
            _GRAPH_TIMEOUT_S,
        ), "full OpenRAL graph did not become observable within 10 s"

        kernel = observed.diagnostics["openral_safety_kernel"]
        kernel_values = _diagnostic_values(kernel)
        assert kernel.message == "passthrough active"
        assert kernel_values["envelope_loaded"] == "true"
        assert kernel_values["n_dof"] == "6"

        hal = observed.diagnostics["openral_hal_galaxea_a1"]
        hal_values = _diagnostic_values(hal)
        assert hal.message == "A1 feedback and motor status healthy"
        assert hal_values["estopped"] == "false"
        assert hal_values["command_relay_state"] == "LOCKED"

        assert observed.joint_state is not None
        initial = _ordered_positions(observed.joint_state)
        target = _project_feedback_to_command_limits(initial)
        trace_id = f"galaxea-a1-deploy-hil-{time.time_ns()}"
        candidate = ActionChunk()
        candidate.header.stamp = node.get_clock().now().to_msg()
        candidate.control_mode = 0
        candidate.horizon = 1
        candidate.flat = list(target)
        candidate.n_dof = len(target)
        candidate.confidence = 1.0
        candidate.rskill_id = "galaxea_a1_current_hold_hil"
        candidate.rskill_revision = "local"
        candidate.trace_id = trace_id
        candidate.tick_index = 1

        deadline = time.monotonic() + _HOLD_TIMEOUT_S
        while time.monotonic() < deadline:
            cycle_started = time.monotonic()
            candidate.header.stamp = node.get_clock().now().to_msg()
            candidate_pub.publish(candidate)
            rclpy.spin_once(node, timeout_sec=0.01)
            if observed.joint_state is not None:
                current = _ordered_positions(observed.joint_state)
                assert (
                    max(abs(now - start) for now, start in zip(current, initial, strict=True))
                    < _MAX_HOLD_DRIFT_RAD
                )
            hal_status = observed.diagnostics.get("openral_hal_galaxea_a1")
            matching_safe = [
                action for action in observed.safe_actions if action.trace_id == trace_id
            ]
            if (
                hal_status is not None
                and _diagnostic_values(hal_status).get("command_relay_state") == "ACTIVE"
                and matching_safe
            ):
                break
            time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - cycle_started)))
        else:
            hal_status = observed.diagnostics.get("openral_hal_galaxea_a1")
            hal_debug = (
                _diagnostic_values(hal_status) if hal_status is not None else {"status": "missing"}
            )
            hal_history = [
                _diagnostic_values(status)
                for status in observed.diagnostic_history.get("openral_hal_galaxea_a1", [])
            ]
            kernel_status = observed.diagnostics.get("openral_safety_kernel")
            kernel_debug = (
                _diagnostic_values(kernel_status)
                if kernel_status is not None
                else {"status": "missing"}
            )
            pytest.fail(
                "current hold did not traverse the kernel and activate the relay; "
                f"matching_safe={len(matching_safe)}, hal={hal_debug}, "
                f"hal_history={hal_history}, kernel={kernel_debug}"
            )

        matching_safe = [action for action in observed.safe_actions if action.trace_id == trace_id]
        assert matching_safe
        assert tuple(matching_safe[-1].flat) == target
        assert matching_safe[-1].n_dof == len(target)
        assert matching_safe[-1].control_mode == 0
        assert not [failure for failure in observed.failures if failure.trace_id == trace_id]

        hal_values = _diagnostic_values(observed.diagnostics["openral_hal_galaxea_a1"])
        expected_csv = ",".join(f"{value:.6f}" for value in target)
        assert hal_values["command_relay_state"] == "ACTIVE"
        assert hal_values["staged_position"] == expected_csv
        assert hal_values["forwarded_position"] == expected_csv

        if _ALLOW_NUDGE:
            nudge_target = list(target)
            nudge_target[0] += _NUDGE_RAD
            assert nudge_target[0] <= GALAXEA_A1_DESCRIPTION.joints[0].position_limits[1]
            nudge_trace_id = f"galaxea-a1-deploy-hil-nudge-{time.time_ns()}"
            candidate.flat = nudge_target
            candidate.trace_id = nudge_trace_id
            candidate.rskill_id = "galaxea_a1_joint1_nudge_hil"
            candidate.tick_index = 2

            deadline = time.monotonic() + _MOTION_TIMEOUT_S
            settled = False
            while time.monotonic() < deadline:
                cycle_started = time.monotonic()
                candidate.header.stamp = node.get_clock().now().to_msg()
                candidate_pub.publish(candidate)
                rclpy.spin_once(node, timeout_sec=0.01)
                if observed.joint_state is not None:
                    current = _ordered_positions(observed.joint_state)
                    for index, (start, actual) in enumerate(zip(initial, current, strict=True)):
                        assert abs(actual - start) < _MAX_EXCURSION_RAD, (
                            f"A1 joint {index + 1} excursion "
                            f"{abs(actual - start):.4f} rad exceeded "
                            f"{_MAX_EXCURSION_RAD:.4f} rad"
                        )
                    settled = abs(current[0] - nudge_target[0]) <= _SETTLE_TOLERANCE_RAD and any(
                        action.trace_id == nudge_trace_id for action in observed.safe_actions
                    )
                if settled:
                    break
                time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - cycle_started)))
            assert settled, "joint1 did not settle at the full-graph +0.01 rad target"
            nudge_safe = [
                action for action in observed.safe_actions if action.trace_id == nudge_trace_id
            ]
            assert nudge_safe
            assert tuple(nudge_safe[-1].flat) == tuple(nudge_target)
            assert not [
                failure for failure in observed.failures if failure.trace_id == nudge_trace_id
            ]

            return_trace_id = f"galaxea-a1-deploy-hil-return-{time.time_ns()}"
            candidate.flat = list(target)
            candidate.trace_id = return_trace_id
            candidate.rskill_id = "galaxea_a1_joint1_return_hil"
            candidate.tick_index = 3
            deadline = time.monotonic() + _MOTION_TIMEOUT_S
            returned = False
            while time.monotonic() < deadline:
                cycle_started = time.monotonic()
                candidate.header.stamp = node.get_clock().now().to_msg()
                candidate_pub.publish(candidate)
                rclpy.spin_once(node, timeout_sec=0.01)
                if observed.joint_state is not None:
                    current = _ordered_positions(observed.joint_state)
                    returned = max(
                        abs(actual - start) for actual, start in zip(current, initial, strict=True)
                    ) <= _SETTLE_TOLERANCE_RAD and any(
                        action.trace_id == return_trace_id for action in observed.safe_actions
                    )
                if returned:
                    break
                time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - cycle_started)))
            assert returned, "joint1 did not return to the measured full-graph start"
            return_safe = [
                action for action in observed.safe_actions if action.trace_id == return_trace_id
            ]
            assert return_safe
            assert tuple(return_safe[-1].flat) == target
            assert not [
                failure for failure in observed.failures if failure.trace_id == return_trace_id
            ]
    finally:
        # The real A1 HAL requires a fresh sidecar and lifecycle after estop.
        # Keep the already-validated current hold alive until the e-stop is
        # observed. The sidecar's short command lease must not race DDS
        # delivery of the stop request and turn a clean e-stop into an
        # unacknowledged transport fault.
        estop_confirmed = False
        try:
            deadline = time.monotonic() + _ESTOP_TIMEOUT_S
            while time.monotonic() < deadline:
                cycle_started = time.monotonic()
                if candidate is not None:
                    candidate.header.stamp = node.get_clock().now().to_msg()
                    candidate_pub.publish(candidate)
                estop_pub.publish(Empty())
                rclpy.spin_once(node, timeout_sec=0.01)
                status = observed.diagnostics.get("openral_hal_galaxea_a1")
                if status is not None and _diagnostic_values(status).get("estopped") == "true":
                    estop_confirmed = True
                    break
                time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - cycle_started)))
        finally:
            for subscription in subscriptions:
                node.destroy_subscription(subscription)
            node.destroy_publisher(candidate_pub)
            node.destroy_publisher(estop_pub)
            node.destroy_node()
            rclpy.try_shutdown()
        assert estop_confirmed, "HAL diagnostics did not confirm the downstream e-stop"
