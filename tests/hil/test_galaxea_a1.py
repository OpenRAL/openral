"""Hardware-in-loop bring-up test for the Galaxea A1 ROS 1 sidecar.

The operator starts ``tools/run_galaxea_a1_sidecar.sh`` first. This test then
uses the real OpenRAL HAL and the real loopback protocol; it never imports or
bundles the vendor SDK.

Environment:
    GALAXEA_A1_HIL: Must be ``"1"`` to connect. This explicit gate prevents a
        developer laptop from claiming a nearby sidecar accidentally.
    GALAXEA_A1_HOST / GALAXEA_A1_PORT: Sidecar endpoint (defaults to
        ``127.0.0.1:46011``).
    GALAXEA_A1_ALLOW_HOLD: When ``"1"``, repeatedly sends the measured current
        pose for two seconds and verifies that no joint moves by 1 degree or
        more. This duration also proves that the sidecar command lease is
        renewed after the relay becomes active.
        Feedback within the tracked endpoint tolerance is projected to the
        exact command limit; larger projections fail before publication. The
        default is observation-only and sends no action.
    GALAXEA_A1_ALLOW_NUDGE: When ``"1"``, implies the hold gate, then moves
        ``arm_joint1`` by +0.01 rad and returns to the measured start. Both
        legs must settle within 0.006 rad while every joint remains inside a
        bounded excursion from the initial feedback.
    GALAXEA_A1_ALLOW_GRIPPER: When ``"1"``, implies the hold gate, moves the
        G2 gripper by the vendor example's 10 mm step away from the nearest
        endpoint, verifies feedback within the measured 2.5 mm steady-state
        tolerance, and returns to the measured opening.

The single test owns one sidecar session. It validates feedback, cached health,
and downstream e-stop in one bounded sequence because the sidecar deliberately
stops its ROS 1 driver/tracker stack when its sole client disconnects.
"""

from __future__ import annotations

import math
import os
import time

import pytest
from openral_core import Action, ControlMode
from openral_core.exceptions import ROSEStopRequested
from openral_hal.galaxea_a1 import GALAXEA_A1_DESCRIPTION, GalaxeaA1HAL

_ENABLED = os.environ.get("GALAXEA_A1_HIL", "0") == "1"
_ALLOW_HOLD = os.environ.get("GALAXEA_A1_ALLOW_HOLD", "0") == "1"
_ALLOW_NUDGE = os.environ.get("GALAXEA_A1_ALLOW_NUDGE", "0") == "1"
_ALLOW_GRIPPER = os.environ.get("GALAXEA_A1_ALLOW_GRIPPER", "0") == "1"
_HOST = os.environ.get("GALAXEA_A1_HOST", "127.0.0.1")
_PORT = int(os.environ.get("GALAXEA_A1_PORT", "46011"))
_NUDGE_RAD = 0.01
_SETTLE_TOLERANCE_RAD = 0.006
_MOTION_TIMEOUT_S = 3.0
_MAX_EXCURSION_RAD = 0.025
_GRIPPER_NUDGE_MM = 10.0
# This G2's measured mid-stroke steady-state error was 2.08-2.21 mm across
# repeated 10 mm commands. Keep modest headroom while still requiring at least
# 7.5 mm of the commanded excursion and the same bound on the return leg.
_GRIPPER_TOLERANCE_MM = 2.5
_HOLD_DURATION_S = 2.0

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="GALAXEA_A1_HIL=1 is required for the live A1 sidecar test.",
)


def test_galaxea_a1_observe_optional_hold_and_estop() -> None:
    """Validate one complete, bounded hardware session and stop it downstream."""
    hal = GalaxeaA1HAL(host=_HOST, port=_PORT, connect_timeout_s=20.0)
    connected = False
    try:
        hal.connect()
        connected = True

        start = time.monotonic()
        samples = [hal.read_state()]
        for _ in range(2):
            time.sleep(0.05)
            samples.append(hal.read_state())
        elapsed = time.monotonic() - start

        expected_names = [joint.name for joint in GALAXEA_A1_DESCRIPTION.joints]
        defaults = GALAXEA_A1_DESCRIPTION.hal.parameters.defaults
        feedback_tolerance = float(defaults["feedback_limit_tolerance_rad"])
        assert elapsed < 1.0, f"three cached A1 reads took {elapsed:.3f} s"
        for state in samples:
            assert state.name == expected_names
            assert len(state.position) == 6
            assert all(math.isfinite(value) for value in state.position)
            for value, joint in zip(state.position, GALAXEA_A1_DESCRIPTION.joints, strict=True):
                lower, upper = joint.position_limits
                assert lower - feedback_tolerance <= value <= upper + feedback_tolerance

        report = hal.health()
        assert report.message == "A1 feedback and motor status healthy"
        assert len(report.fields["motor_status_codes"].split(",")) >= 7

        if _ALLOW_HOLD or _ALLOW_NUDGE or _ALLOW_GRIPPER:
            before = samples[-1]
            hold_target: list[float] = []
            for value, joint in zip(before.position, GALAXEA_A1_DESCRIPTION.joints, strict=True):
                lower, upper = joint.position_limits
                projected = min(upper, max(lower, value))
                assert abs(projected - value) <= feedback_tolerance
                hold_target.append(projected)
            relay_deadline = time.monotonic() + float(defaults["tracker_alignment_timeout_s"])
            while time.monotonic() < relay_deadline:
                hal.send_action(
                    Action(
                        control_mode=ControlMode.JOINT_POSITION,
                        horizon=1,
                        joint_targets=[hold_target],
                        stamp_ns=time.time_ns(),
                    )
                )
                time.sleep(0.05)
                if hal.health().fields["command_relay_state"] == "ACTIVE":
                    break
            assert hal.health().fields["command_relay_state"] == "ACTIVE", (
                "A1 staged command relay did not become ACTIVE"
            )
            hold_deadline = time.monotonic() + _HOLD_DURATION_S
            while time.monotonic() < hold_deadline:
                hal.send_action(
                    Action(
                        control_mode=ControlMode.JOINT_POSITION,
                        horizon=1,
                        joint_targets=[hold_target],
                        stamp_ns=time.time_ns(),
                    )
                )
                time.sleep(0.05)
                current = hal.read_state()
                for previous, actual in zip(before.position, current.position, strict=True):
                    assert abs(actual - previous) < 0.0175, (
                        f"A1 joint moved {abs(actual - previous):.4f} rad during hold"
                    )

            if _ALLOW_NUDGE:
                nudge_target = list(hold_target)
                joint1_upper = GALAXEA_A1_DESCRIPTION.joints[0].position_limits[1]
                nudge_target[0] += _NUDGE_RAD
                assert nudge_target[0] <= joint1_upper

                nudge_samples = []
                last_report = hal.health()
                deadline = time.monotonic() + _MOTION_TIMEOUT_S
                while time.monotonic() < deadline:
                    hal.send_action(
                        Action(
                            control_mode=ControlMode.JOINT_POSITION,
                            horizon=1,
                            joint_targets=[nudge_target],
                            stamp_ns=time.time_ns(),
                        )
                    )
                    time.sleep(0.05)
                    state = hal.read_state()
                    nudge_samples.append(state)
                    last_report = hal.health()
                    if abs(state.position[0] - nudge_target[0]) <= _SETTLE_TOLERANCE_RAD:
                        break
                assert nudge_samples
                assert abs(nudge_samples[-1].position[0] - nudge_target[0]) <= (
                    _SETTLE_TOLERANCE_RAD
                ), (
                    f"joint1 target={nudge_target[0]:.6f}, "
                    f"feedback={nudge_samples[-1].position[0]:.6f}, "
                    f"staged={last_report.fields['staged_position']}, "
                    f"forwarded={last_report.fields['forwarded_position']}"
                )
                for state in nudge_samples:
                    for index, (initial, current) in enumerate(
                        zip(before.position, state.position, strict=True)
                    ):
                        assert abs(current - initial) < _MAX_EXCURSION_RAD, (
                            f"A1 joint {index + 1} excursion "
                            f"{abs(current - initial):.4f} rad exceeded "
                            f"{_MAX_EXCURSION_RAD:.4f} rad"
                        )

                return_samples = []
                deadline = time.monotonic() + _MOTION_TIMEOUT_S
                while time.monotonic() < deadline:
                    hal.send_action(
                        Action(
                            control_mode=ControlMode.JOINT_POSITION,
                            horizon=1,
                            joint_targets=[hold_target],
                            stamp_ns=time.time_ns(),
                        )
                    )
                    time.sleep(0.05)
                    state = hal.read_state()
                    return_samples.append(state)
                    if (
                        max(
                            abs(current - initial)
                            for current, initial in zip(
                                state.position, before.position, strict=True
                            )
                        )
                        <= _SETTLE_TOLERANCE_RAD
                    ):
                        break
                assert return_samples
                final_error = max(
                    abs(current - initial)
                    for current, initial in zip(
                        return_samples[-1].position, before.position, strict=True
                    )
                )
                assert final_error <= _SETTLE_TOLERANCE_RAD

            if _ALLOW_GRIPPER:
                initial_gripper = float(hal.health().fields["gripper_position_normalized"])
                gripper_stroke = float(defaults["gripper_stroke_max_mm"]) - float(
                    defaults["gripper_stroke_min_mm"]
                )
                gripper_nudge = _GRIPPER_NUDGE_MM / gripper_stroke
                gripper_tolerance = _GRIPPER_TOLERANCE_MM / gripper_stroke
                direction = 1.0 if initial_gripper <= 0.5 else -1.0
                gripper_target = initial_gripper + direction * gripper_nudge
                assert 0.0 <= gripper_target <= 1.0

                gripper_feedback = initial_gripper
                returned_gripper = initial_gripper
                try:
                    deadline = time.monotonic() + _MOTION_TIMEOUT_S
                    while time.monotonic() < deadline:
                        hal.send_action(
                            Action(
                                control_mode=ControlMode.GRIPPER_POSITION,
                                gripper=[gripper_target],
                                stamp_ns=time.time_ns(),
                            )
                        )
                        time.sleep(0.05)
                        gripper_feedback = float(hal.health().fields["gripper_position_normalized"])
                        if abs(gripper_feedback - gripper_target) <= gripper_tolerance:
                            break
                finally:
                    deadline = time.monotonic() + _MOTION_TIMEOUT_S
                    while time.monotonic() < deadline:
                        hal.send_action(
                            Action(
                                control_mode=ControlMode.GRIPPER_POSITION,
                                gripper=[initial_gripper],
                                stamp_ns=time.time_ns(),
                            )
                        )
                        time.sleep(0.05)
                        returned_gripper = float(hal.health().fields["gripper_position_normalized"])
                        if abs(returned_gripper - initial_gripper) <= gripper_tolerance:
                            break
                assert abs(returned_gripper - initial_gripper) <= gripper_tolerance, (
                    f"G2 return target={initial_gripper:.6f}, feedback={returned_gripper:.6f}"
                )
                assert abs(gripper_feedback - gripper_target) <= gripper_tolerance, (
                    f"G2 target={gripper_target:.6f}, feedback={gripper_feedback:.6f}"
                )

        with pytest.raises(ROSEStopRequested):
            hal.estop()
        connected = False
    finally:
        if connected:
            hal.disconnect()
