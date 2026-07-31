#!/usr/bin/env python3
"""Galaxea A1 ROS 1 sidecar for :class:`openral_hal.galaxea_a1.GalaxeaA1HAL`.

Run this file inside a ROS Noetic environment with the operator-provided A1
SDK sourced.  It owns roscore, the official serial driver, and the official
joint tracker; every exit path stops all three.  No vendor code is bundled in
OpenRAL.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

JOINT_NAMES = tuple(f"arm_joint{i}" for i in range(1, 7))
JOINT_LIMITS = (
    (-2.8798, 2.8798),
    (0.0, 3.1415),
    (-3.3161, 0.0),
    (-2.8798, 2.8798),
    (-1.6581, 1.6581),
    (-2.8798, 2.8798),
)
PROTOCOL_VERSION = 1
GRIPPER_STATUS_INDEX = len(JOINT_NAMES)
MAX_STATUS_MASK = 0xFFFFFFFF
TRACKER_NODE = "openral_a1_joint_tracker"
STAGED_COMMAND_TOPIC = "/openral/arm_joint_command_staged"
HOST_COMMAND_TOPIC = "/arm_joint_command_host"
TRACKER_ALLOWED_MODES = (0,)


def _driver_argv(serial: str) -> list[str]:
    """Build the official driver launch command with every live topic explicit."""
    return [
        "roslaunch",
        "signal_arm",
        "single_arm_node.launch",
        f"single_arm_serial_port_path:={serial}",
        "single_arm_joint_states_topic:=/joint_states_host",
        "single_arm_gripper_feedback_topic:=/gripper_stroke_host",
        "single_arm_control_topic:=/arm_joint_command_host",
        "single_arm_arm_status_topic:=/arm_status_host",
        "single_arm_gripper_position_control_topic:=/gripper_position_control_host",
        "single_arm_gripper_version:=G2",
    ]


def _tracker_argv() -> list[str]:
    """Run only the official joint tracker node under an OpenRAL-owned name."""
    return [
        "rosrun",
        "mobiman",
        "jointTracker_demo_node",
        f"__name:={TRACKER_NODE}",
    ]


def _tracker_parameters(sdk_root: str) -> dict[str, str]:
    """Return the official tracker parameters with every live topic explicit.

    The current vendor ``jointTrackerdemo.launch`` declares its arguments with
    ``value=`` rather than ``default=``, so ROS 1 rejects command-line
    overrides. Starting the same official binary directly avoids that
    SDK-version trap and also omits the unrelated EE-pose publisher.
    """
    share = os.path.join(sdk_root, "install", "share", "mobiman")
    private = f"/{TRACKER_NODE}"
    return {
        f"{private}/joint_states_sub_topic": "/joint_states_host",
        f"{private}/arm_joint_command_topic": STAGED_COMMAND_TOPIC,
        f"{private}/arm_joint_target_position": "/arm_joint_target_position",
        f"{private}/taskFile": os.path.join(share, "config", "task.info"),
        f"{private}/urdfFile": os.path.join(share, "urdf", "A1", "urdf", "A1_URDF_0607_0028.urdf"),
        f"{private}/libFolder": os.path.join(share, "auto_generated", "x1_robot"),
    }


class Stack:
    """Own and tear down the ROS 1 process group in reverse start order."""

    def __init__(self, shutdown_grace_s: float = 3.0, kill_wait_s: float = 1.0) -> None:
        self._children: list[Any] = []
        self._shutdown_grace_s = shutdown_grace_s
        self._kill_wait_s = kill_wait_s

    def start(self, argv: Sequence[str]) -> Any:
        child = subprocess.Popen(argv, start_new_session=True)
        self._children.append(child)
        return child

    def stop(self) -> None:
        for child in reversed(self._children):
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGINT)
        deadline = time.monotonic() + self._shutdown_grace_s
        for child in reversed(self._children):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                # An e-stop acknowledgement is meaningful only after the
                # tracker/driver process has actually exited, not merely after
                # SIGKILL was requested.
                child.wait(timeout=self._kill_wait_s)
        self._children = []


class Bridge:
    def __init__(
        self,
        rospy: Any,
        joint_state_type: Callable[[], Any],
        status_type: Any,
        gripper_type: Callable[[], Any],
        arm_control_type: Callable[[], Any],
    ) -> None:
        self._rospy = rospy
        self._lock = threading.Lock()
        self._joint: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int] | None = (
            None
        )
        self._gripper_mm: float | None = None
        self._status: tuple[int, ...] | None = None
        self._joint_updated = 0.0
        self._gripper_updated = 0.0
        self._status_updated = 0.0
        self._relay_config: dict[str, Any] | None = None
        self._relay_state = "LOCKED"
        self._relay_fault = ""
        self._activation_target: tuple[float, ...] | None = None
        self._activation_deadline = 0.0
        self._published_target: tuple[float, ...] | None = None
        self._published_gripper_mm: float | None = None
        self._staged_position: tuple[float, ...] | None = None
        self._forwarded_position: tuple[float, ...] | None = None
        self._accepted_action_count = 0
        self._joint_pub = rospy.Publisher(
            "/arm_joint_target_position", joint_state_type, queue_size=1
        )
        self._arm_command_pub = rospy.Publisher(HOST_COMMAND_TOPIC, arm_control_type, queue_size=1)
        self._gripper_pub = rospy.Publisher(
            "/gripper_position_control_host", gripper_type, queue_size=1
        )
        rospy.Subscriber("/joint_states_host", joint_state_type, self._on_joint, queue_size=1)
        rospy.Subscriber("/gripper_stroke_host", joint_state_type, self._on_gripper, queue_size=1)
        rospy.Subscriber("/arm_status_host", status_type, self._on_status, queue_size=1)
        self._staged_sub = rospy.Subscriber(
            STAGED_COMMAND_TOPIC, arm_control_type, self._on_staged, queue_size=1
        )
        self._joint_state_type = joint_state_type
        self._gripper_type = gripper_type

    def configure(self, config: dict[str, Any]) -> None:
        """Install the validated client contract while keeping motion locked."""
        with self._lock:
            self._relay_config = config
            self._relay_state = "LOCKED"
            self._relay_fault = ""
            self._activation_target = None
            self._activation_deadline = 0.0
            self._published_target = None
            self._published_gripper_mm = None
            self._staged_position = None
            self._forwarded_position = None
            self._accepted_action_count = 0

    def _on_joint(self, msg: Any) -> None:
        try:
            position = self._ordered(msg.name, msg.position, "position")
            velocity = self._ordered_optional(msg.name, msg.velocity)
            effort = self._ordered_optional(msg.name, msg.effort)
        except ValueError as exc:
            self._rospy.logerr("invalid /joint_states_host: %s", exc)
            return
        with self._lock:
            self._joint = (position, velocity, effort, msg.header.stamp.to_nsec())
            self._joint_updated = time.monotonic()

    def _on_gripper(self, msg: Any) -> None:
        if not msg.position or not math.isfinite(float(msg.position[0])):
            return
        with self._lock:
            self._gripper_mm = float(msg.position[0])
            self._gripper_updated = time.monotonic()

    def _on_status(self, msg: Any) -> None:
        codes = tuple(int(item.error_code) for item in msg.data.motor_errors)
        if len(codes) < 7:
            self._rospy.logerr("invalid /arm_status_host: expected 7 motors, got %d", len(codes))
            return
        if any(code < 0 or code > MAX_STATUS_MASK for code in codes):
            self._rospy.logerr("invalid /arm_status_host: motor code outside uint32 range")
            return
        with self._lock:
            self._status = codes
            self._status_updated = time.monotonic()

    def _on_staged(self, msg: Any) -> None:
        try:
            position = self._decode_tracker_command(msg)
        except (AttributeError, OverflowError, TypeError, ValueError) as exc:
            self._latch_relay_fault(str(exc))
            return

        now = time.monotonic()
        output = None
        with self._lock:
            self._staged_position = position
            config = self._relay_config
            target = self._activation_target
            joint = self._joint
            status = self._status
            feedback_age = now - self._joint_updated
            status_age = now - self._status_updated
            position_in_command_limits = all(
                lower <= value <= upper for value, (lower, upper) in zip(position, JOINT_LIMITS)
            )
            if not position_in_command_limits:
                tolerance = config["feedback_limit_tolerance"] if config is not None else 0.0
                position_in_feedback_limits = all(
                    lower - tolerance <= value <= upper + tolerance
                    for value, (lower, upper) in zip(position, JOINT_LIMITS)
                )
                # The official tracker starts its interpolation at measured
                # feedback. Encoder zero/quantization can put that sample just
                # beyond a nominal endpoint even though the requested target
                # has already been projected to the exact command limit.
                # Keep such samples staged while ARMING; they are never
                # forwarded to the host command topic. Any larger excursion,
                # or any out-of-range command after activation, remains a
                # fail-closed relay fault.
                if self._relay_state == "ARMING" and position_in_feedback_limits:
                    return
                for name, value, (lower, upper) in zip(JOINT_NAMES, position, JOINT_LIMITS):
                    if not lower <= value <= upper:
                        self._latch_relay_fault_locked(
                            f"tracker command {name}={value:.6f} is outside "
                            f"[{lower:.6f}, {upper:.6f}]"
                        )
                        return
            if (
                config is not None
                and target is not None
                and joint is not None
                and status is not None
                and feedback_age <= config["state_timeout"]
                and status_age <= config["status_timeout"]
            ):
                try:
                    self._check_status(status, config)
                except RuntimeError as exc:
                    self._latch_relay_fault_locked(str(exc))
                else:
                    current = joint[0]
                    for name, value, lower, upper in zip(
                        JOINT_NAMES,
                        current,
                        config["joint_position_min"],
                        config["joint_position_max"],
                    ):
                        tolerance = config["feedback_limit_tolerance"]
                        if not lower - tolerance <= value <= upper + tolerance:
                            self._latch_relay_fault_locked(
                                f"{name} feedback {value:.6f} is outside "
                                f"[{lower:.6f}, {upper:.6f}] by more than "
                                f"{tolerance:.3f} rad"
                            )
                            return
                    target_error = max(abs(a - b) for a, b in zip(position, target))
                    feedback_error = max(abs(a - b) for a, b in zip(position, current))
                    if (
                        self._relay_state == "ARMING"
                        and target_error <= config["alignment"]
                        and feedback_error <= config["alignment"]
                    ):
                        self._relay_state = "ACTIVE"
                        self._rospy.logwarn(
                            "OpenRAL A1 relay ACTIVE after staged hold alignment "
                            "(target %.6f rad, feedback %.6f rad)",
                            target_error,
                            feedback_error,
                        )
                    if self._relay_state == "ACTIVE":
                        output = copy.deepcopy(msg)
                        self._forwarded_position = position
        if output is not None:
            output.header.stamp = self._rospy.Time.now()
            self._arm_command_pub.publish(output)

    @staticmethod
    def _decode_tracker_command(msg: Any) -> tuple[float, ...]:
        vectors = {
            "p_des": msg.p_des,
            "v_des": msg.v_des,
            "kp": msg.kp,
            "kd": msg.kd,
            "t_ff": msg.t_ff,
        }
        decoded: dict[str, tuple[float, ...]] = {}
        for label, values in vectors.items():
            if len(values) != len(JOINT_NAMES):
                raise ValueError(
                    f"tracker command {label} has {len(values)} values, "
                    f"need exactly {len(JOINT_NAMES)}"
                )
            decoded[label] = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in decoded[label]):
                raise ValueError(f"tracker command {label} contains non-finite values")
        if any(value < 0.0 for value in decoded["kp"]):
            raise ValueError("tracker command kp contains negative gains")
        if any(value < 0.0 for value in decoded["kd"]):
            raise ValueError("tracker command kd contains negative gains")
        if isinstance(msg.mode, bool) or not isinstance(msg.mode, int):
            raise ValueError(f"tracker command mode is not an integer: {msg.mode!r}")
        if msg.mode not in TRACKER_ALLOWED_MODES:
            raise ValueError(
                f"tracker command mode {msg.mode} is not allowed; "
                f"expected one of {list(TRACKER_ALLOWED_MODES)}"
            )
        return decoded["p_des"]

    def _latch_relay_fault(self, reason: str) -> None:
        with self._lock:
            self._latch_relay_fault_locked(reason)

    def _latch_relay_fault_locked(self, reason: str) -> None:
        if not self._relay_fault:
            self._relay_fault = reason
            self._relay_state = "FAULT"
            self._rospy.logerr("OpenRAL A1 relay FAULT: %s", reason)

    @staticmethod
    def _ordered(names: Sequence[str], values: Sequence[float], label: str) -> tuple[float, ...]:
        if len(values) < len(JOINT_NAMES):
            raise ValueError(f"{label} has {len(values)} values")
        if names:
            if len(names) != len(set(names)):
                raise ValueError("duplicate joint names")
            by_name = dict(zip(names, values))
            missing = [name for name in JOINT_NAMES if name not in by_name]
            if missing:
                raise ValueError(f"missing joints {missing!r}")
            result = tuple(float(by_name[name]) for name in JOINT_NAMES)
        else:
            result = tuple(float(value) for value in values[: len(JOINT_NAMES)])
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label} contains non-finite values")
        return result

    @classmethod
    def _ordered_optional(cls, names: Sequence[str], values: Sequence[float]) -> tuple[float, ...]:
        if not values:
            return ()
        return cls._ordered(names, values, "optional vector")

    def ready(self) -> bool:
        with self._lock:
            return (
                self._joint is not None
                and self._gripper_mm is not None
                and self._status is not None
            )

    def command_paths_ready(self) -> bool:
        """Return whether staged and host command consumers are connected."""
        return bool(
            self._joint_pub.get_num_connections() > 0
            and self._staged_sub.get_num_connections() > 0
            and self._arm_command_pub.get_num_connections() > 0
            and self._gripper_pub.get_num_connections() > 0
        )

    def relay_state(self) -> str:
        with self._lock:
            return self._relay_state

    def state(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            joint = self._joint
            gripper_mm = self._gripper_mm
            status = self._status
            updated = (self._joint_updated, self._gripper_updated, self._status_updated)
            relay_state = self._relay_state
            relay_fault = self._relay_fault
            activation_deadline = self._activation_deadline
            staged_position = self._staged_position
            forwarded_position = self._forwarded_position
            accepted_action_count = self._accepted_action_count
        if joint is None or gripper_mm is None or status is None:
            raise RuntimeError("hardware feedback is incomplete")
        now = time.monotonic()
        if relay_fault:
            raise RuntimeError(f"command relay fault: {relay_fault}")
        if relay_state == "ARMING" and now > activation_deadline:
            self._latch_relay_fault(
                f"tracker did not align its staged hold within "
                f"{config['tracker_alignment_timeout']:.3f} s"
            )
            raise RuntimeError("command relay fault: tracker staged-hold alignment timed out")
        if now - updated[0] > config["state_timeout"]:
            raise RuntimeError("joint feedback is stale")
        if now - updated[1] > config["state_timeout"]:
            raise RuntimeError("gripper feedback is stale")
        if now - updated[2] > config["status_timeout"]:
            raise RuntimeError("motor status is stale")
        position, velocity, effort, stamp_ns = joint
        for name, value, lower, upper in zip(
            JOINT_NAMES,
            position,
            config["joint_position_min"],
            config["joint_position_max"],
        ):
            tolerance = config["feedback_limit_tolerance"]
            if not lower - tolerance <= value <= upper + tolerance:
                raise RuntimeError(
                    f"{name} feedback {value:.6f} is outside [{lower:.6f}, {upper:.6f}] "
                    f"by more than {tolerance:.3f} rad"
                )
        self._check_status(status, config)
        gripper_norm = (gripper_mm - config["gripper_min"]) / (
            config["gripper_max"] - config["gripper_min"]
        )
        if not 0.0 <= gripper_norm <= 1.0:
            raise RuntimeError(f"gripper feedback {gripper_mm:.3f} mm is outside configured range")
        return {
            "op": "state",
            "name": list(JOINT_NAMES),
            "position": list(position),
            "velocity": list(velocity),
            "effort": list(effort),
            "stamp_ns": int(stamp_ns),
            "status_codes": list(status),
            "gripper_norm": gripper_norm,
            "relay_state": relay_state,
            "staged_position": list(staged_position or ()),
            "forwarded_position": list(forwarded_position or ()),
            "accepted_action_count": accepted_action_count,
        }

    def apply(
        self, packet: dict[str, Any], config: dict[str, Any], first_joint_command: bool
    ) -> bool:
        state = self.state(config)
        self._check_status(state["status_codes"], config)
        target = packet.get("joint_targets")
        if target is not None:
            if not isinstance(target, list) or len(target) != len(JOINT_NAMES):
                raise ValueError("joint_targets must contain six values")
            target = tuple(float(value) for value in target)
            if not all(math.isfinite(value) for value in target):
                raise ValueError("joint target contains a non-finite value")
            for name, value, lower, upper in zip(
                JOINT_NAMES,
                target,
                config["joint_position_min"],
                config["joint_position_max"],
            ):
                if not lower <= value <= upper:
                    raise ValueError(
                        f"{name} target {value:.6f} is outside [{lower:.6f}, {upper:.6f}]"
                    )
            with self._lock:
                relay_state = self._relay_state
            bound = (
                config["alignment"]
                if first_joint_command or relay_state == "ARMING"
                else config["max_step"]
            )
            if any(abs(target[i] - state["position"][i]) > bound for i in range(len(JOINT_NAMES))):
                raise ValueError(f"joint target exceeds {bound:.3f} rad feedback-relative bound")
            if first_joint_command:
                with self._lock:
                    self._relay_state = "ARMING"
                    self._activation_target = target
                    self._activation_deadline = (
                        time.monotonic() + config["tracker_alignment_timeout"]
                    )
            elif relay_state == "ARMING":
                # Closed-loop IK recomputes a feedback-relative substep on
                # every tick, so encoder noise can move the target slightly
                # while the official tracker is still converging. Keep the
                # original bounded deadline, but align against the latest
                # target. Every ARMING target remains subject to the tighter
                # initial-alignment bound above.
                with self._lock:
                    self._activation_target = target
            elif relay_state != "ACTIVE":
                raise RuntimeError(f"command relay is not ACTIVE: {relay_state}")
            with self._lock:
                publish_target = self._published_target != target
                if publish_target:
                    self._published_target = target
            if publish_target:
                msg = self._joint_state_type()
                msg.header.stamp = self._rospy.Time.now()
                msg.header.frame_id = "base_link"
                msg.name = list(JOINT_NAMES)
                msg.position = list(target)
                self._joint_pub.publish(msg)
            first_joint_command = False
        gripper = packet.get("gripper")
        if gripper is not None:
            gripper = float(gripper)
            if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
                raise ValueError("gripper must be within [0, 1]")
            with self._lock:
                gripper_relay_state = self._relay_state
            if gripper_relay_state != "ACTIVE":
                # The gripper bypasses the tracker's staged-hold interpolation,
                # so it must not actuate before the arm relay has finished
                # alignment. Same fail-closed contract as the joint path above;
                # the deploy flow always streams the joint hold to ACTIVE
                # before the first gripper setpoint.
                raise RuntimeError(
                    f"gripper command refused: command relay is not ACTIVE: {gripper_relay_state}"
                )
            gripper_mm = config["gripper_min"] + gripper * (
                config["gripper_max"] - config["gripper_min"]
            )
            with self._lock:
                publish_gripper = self._published_gripper_mm != gripper_mm
                if publish_gripper:
                    self._published_gripper_mm = gripper_mm
            if publish_gripper:
                msg = self._gripper_type()
                msg.header.stamp = self._rospy.Time.now()
                msg.gripper_stroke = gripper_mm
                self._gripper_pub.publish(msg)
        if target is None and gripper is None:
            raise ValueError("action has no joint_targets or gripper")
        with self._lock:
            self._accepted_action_count += 1
        return first_joint_command

    @staticmethod
    def _check_status(codes: Sequence[int], config: dict[str, Any]) -> None:
        for index, code in enumerate(codes[: GRIPPER_STATUS_INDEX + 1]):
            allowed = config["idle_mask"] | (
                config["gripper_mask"] if index == GRIPPER_STATUS_INDEX else 0
            )
            if int(code) & ~allowed:
                raise RuntimeError(f"actuator {index + 1} reports error code {code}")


def _send(conn: socket.socket, message: dict[str, Any]) -> None:
    conn.sendall(json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n")


def _hello_config(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("op") != "hello" or packet.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("expected OpenRAL Galaxea A1 protocol hello v1")
    if packet.get("robot") != "galaxea_a1":
        raise ValueError("robot identity mismatch")
    if packet.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("joint_names mismatch")
    joint_min = tuple(float(value) for value in packet["joint_position_min"])
    joint_max = tuple(float(value) for value in packet["joint_position_max"])
    if len(joint_min) != len(JOINT_NAMES) or len(joint_max) != len(JOINT_NAMES):
        raise ValueError("joint position limits must contain six lower and upper values")
    if any(
        not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
        for lower, upper in zip(joint_min, joint_max)
    ):
        raise ValueError("joint position limits must be finite ordered pairs")
    if tuple(zip(joint_min, joint_max)) != JOINT_LIMITS:
        raise ValueError("joint position limits do not match the sidecar's official A1 limits")
    idle_mask = packet["idle_timeout_error_mask"]
    gripper_mask = packet["gripper_ignored_error_mask"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (idle_mask, gripper_mask)
    ):
        raise ValueError("motor status masks must be integers")
    config: dict[str, Any] = {
        "alignment": float(packet["initial_alignment_tolerance_rad"]),
        "max_step": float(packet["max_target_step_rad"]),
        "lease": float(packet["command_lease_s"]),
        "state_timeout": float(packet["state_timeout_s"]),
        "status_timeout": float(packet["status_timeout_s"]),
        "feedback_limit_tolerance": float(packet["feedback_limit_tolerance_rad"]),
        "tracker_alignment_timeout": float(packet["tracker_alignment_timeout_s"]),
        "idle_mask": idle_mask,
        "gripper_mask": gripper_mask,
        "gripper_min": float(packet["gripper_stroke_min_mm"]),
        "gripper_max": float(packet["gripper_stroke_max_mm"]),
        "joint_position_min": joint_min,
        "joint_position_max": joint_max,
    }
    positive = (
        "alignment",
        "max_step",
        "lease",
        "state_timeout",
        "status_timeout",
        "tracker_alignment_timeout",
    )
    if any(not math.isfinite(config[key]) or config[key] <= 0.0 for key in positive):
        raise ValueError("motion limits and command lease must be positive")
    if (
        not math.isfinite(config["feedback_limit_tolerance"])
        or config["feedback_limit_tolerance"] < 0.0
    ):
        raise ValueError("feedback limit tolerance must be finite and non-negative")
    if (
        not math.isfinite(config["gripper_min"])
        or not math.isfinite(config["gripper_max"])
        or config["gripper_max"] <= config["gripper_min"]
    ):
        raise ValueError("invalid gripper range")
    if any(
        isinstance(config[key], bool) or not 0 <= config[key] <= MAX_STATUS_MASK
        for key in ("idle_mask", "gripper_mask")
    ):
        raise ValueError("motor status masks must be uint32 values")
    return config


def _serve(
    listener: socket.socket,
    bridge: Bridge,
    rospy: Any,
    stop_stack: Callable[[], None],
) -> None:
    conn, address = listener.accept()
    rospy.loginfo("OpenRAL client connected from %s:%d", *address)
    conn.setblocking(False)
    buffer = b""
    config = None
    last_command = None
    first_joint_command = True
    last_state_send = 0.0
    while not rospy.is_shutdown():
        readable, _, _ = select.select([conn], [], [], 0.01)
        if readable:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                packet = json.loads(line)
                if config is None:
                    config = _hello_config(packet)
                    bridge.configure(config)
                    _send(
                        conn,
                        {"op": "hello", "protocol": PROTOCOL_VERSION, "robot": "galaxea_a1"},
                    )
                elif packet.get("op") == "action":
                    first_joint_command = bridge.apply(packet, config, first_joint_command)
                    last_command = time.monotonic()
                elif packet.get("op") == "estop":
                    stop_stack()
                    _send(conn, {"op": "estopped"})
                    return
                else:
                    raise ValueError(f"unsupported operation {packet.get('op')!r}")
        # Drain a readable packet before evaluating the lease. The ROS tracker
        # callback may briefly starve this thread while the relay transitions
        # ARMING -> ACTIVE; current actions or an e-stop can already be buffered
        # on the socket at that point. Checking first would fault on an old
        # timestamp without consuming the fresh safety input.
        now = time.monotonic()
        if (
            config is not None
            and last_command is not None
            and bridge.relay_state() == "ACTIVE"
            and now - last_command > config["lease"]
        ):
            raise RuntimeError("command lease expired after %.3f s" % (now - last_command))
        if config is not None and now - last_state_send >= 0.02:
            _send(conn, bridge.state(config))
            last_state_send = now


def _wait_for(predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {label}")


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.1)
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=46011)
    parser.add_argument("--serial", default="/dev/a1")
    parser.add_argument("--startup-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    stack = Stack()
    listener = None
    try:
        if _port_is_open("127.0.0.1", 11311):
            raise RuntimeError("ROS master port 11311 is already occupied inside the sidecar")
        roscore = stack.start(["roscore"])
        _wait_for(
            lambda: roscore.poll() is None and _port_is_open("127.0.0.1", 11311),
            args.startup_timeout_s,
            "ROS master",
        )
        import rospy  # type: ignore[import-not-found]  # reason: provided by ROS 1 Noetic sidecar
        from sensor_msgs.msg import JointState
        from signal_arm.msg import (  # type: ignore[import-not-found]  # reason: operator-provided vendor SDK
            arm_control,
            gripper_position_control,
            status_stamped,
        )

        rospy.init_node("openral_galaxea_a1_sidecar", disable_signals=True)
        bridge = Bridge(
            rospy,
            JointState,
            status_stamped,
            gripper_position_control,
            arm_control,
        )
        driver = stack.start(_driver_argv(args.serial))
        _wait_for(
            lambda: driver.poll() is None and bridge.ready(),
            args.startup_timeout_s,
            "fresh A1 feedback and motor status",
        )
        tracker_parameters = _tracker_parameters(os.environ["A1_SDK_ROOT"])
        missing_tracker_assets = [
            value
            for key, value in tracker_parameters.items()
            if key.endswith(("taskFile", "urdfFile", "libFolder")) and not os.path.exists(value)
        ]
        if missing_tracker_assets:
            raise RuntimeError(f"A1 SDK tracker assets are missing: {missing_tracker_assets!r}")
        for name, value in tracker_parameters.items():
            rospy.set_param(name, value)
        tracker = stack.start(_tracker_argv())
        _wait_for(
            lambda: tracker.poll() is None and bridge.command_paths_ready(),
            args.startup_timeout_s,
            "A1 joint tracker and gripper command subscribers",
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen(1)
        print(f"OpenRAL Galaxea A1 sidecar ready on {args.bind}:{args.port}", flush=True)
        _serve(listener, bridge, rospy, stack.stop)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Galaxea A1 sidecar fault: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if listener is not None:
            listener.close()
        stack.stop()


if __name__ == "__main__":
    raise SystemExit(main())
