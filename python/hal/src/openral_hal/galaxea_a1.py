"""Real-hardware HAL for the Galaxea A1 arm via an isolated ROS 1 sidecar.

The vendor stack is ROS 1 Noetic while OpenRAL is ROS 2 / Python 3.12.  This
adapter therefore owns no vendor imports: an operator-provisioned sidecar runs
the official SDK and streams typed JSON records over loopback TCP.  Network I/O
lives on a worker thread so ``read_state`` and ``send_action`` never block.
"""

from __future__ import annotations

import contextlib
import json
import math
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import structlog
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import (
    Action,
    ControlMode,
    EmbodimentKind,
    EndEffectorSpec,
    HalEntrypoints,
    IntrinsicsPinhole,
    JointSpec,
    JointState,
    JointType,
    RobotCapabilities,
    RobotDescription,
    SafetyEnvelope,
    SensorModality,
    SensorSpec,
)

from openral_hal._base import HALBase
from openral_hal.protocol import EStopRecovery, HALHealthReport

__all__ = ["GALAXEA_A1_DESCRIPTION", "GalaxeaA1HAL"]

log = structlog.get_logger(__name__)

_JOINT_NAMES = tuple(f"arm_joint{i}" for i in range(1, 7))
_JOINT_LIMITS = (
    (-2.8798, 2.8798),
    (0.0, 3.1415),
    (-3.3161, 0.0),
    (-2.8798, 2.8798),
    (-1.6581, 1.6581),
    (-2.8798, 2.8798),
)
_PROTOCOL_VERSION = 1
_GRIPPER_STATUS_INDEX = len(_JOINT_NAMES)
_MAX_TCP_PORT = 65536
_MAX_STATUS_MASK = 0xFFFFFFFF


GALAXEA_A1_DESCRIPTION = RobotDescription(
    name="galaxea_a1",
    embodiment_kind=EmbodimentKind.MANIPULATOR,
    base_frame="base_link",
    joints=[
        JointSpec(
            name=name,
            joint_type=JointType.REVOLUTE,
            parent_link="base_link" if index == 1 else f"arm_seg{index - 1}",
            child_link=f"arm_seg{index}",
            position_limits=limits,
            velocity_limit=velocity,
            effort_limit=effort,
            actuator_kind="bldc",
        )
        for index, (name, limits, velocity, effort) in enumerate(
            zip(
                _JOINT_NAMES,
                _JOINT_LIMITS,
                (20.944, 20.944, 7.5398, 25.133, 25.133, 25.133),
                (40.0, 40.0, 27.0, 7.0, 7.0, 7.0),
                strict=True,
            ),
            start=1,
        )
    ],
    end_effectors=[
        EndEffectorSpec(
            name="gripper",
            kind="parallel_gripper",
            n_dof=1,
        )
    ],
    sensors=[
        SensorSpec(
            name="front",
            modality=SensorModality.RGB,
            frame_id="a1_front_camera",
            rate_hz=30.0,
            encoding="rgb8",
            intrinsics=IntrinsicsPinhole(
                width=480,
                height=480,
                fx=387.67059326171875,
                fy=386.86669921875,
                cx=220.1776123046875,
                cy=250.0211944580078,
                distortion_model="plumb_bob",
                distortion_coeffs=[
                    -0.05492031201720238,
                    0.06242402642965317,
                    -0.0008095245575532317,
                    0.00015016942052170634,
                    -0.020708303898572922,
                ],
            ),
            vla_feature_key="observation.images.front",
            vendor="Intel",
            model="RealSense D455",
            metadata={
                "output_width": 480,
                "output_height": 480,
                "crop": "x=103,y=0,width=480,height=480",
            },
        ),
        SensorSpec(
            name="wrist",
            modality=SensorModality.RGB,
            frame_id="a1_wrist_camera",
            parent_frame="arm_seg6",
            rate_hz=30.0,
            encoding="rgb8",
            intrinsics=IntrinsicsPinhole(
                width=640,
                height=480,
                fx=392.2943115234375,
                fy=391.7977294921875,
                cx=320.1477355957031,
                cy=241.63327026367188,
                distortion_model="plumb_bob",
                distortion_coeffs=[
                    -0.05426114425063133,
                    0.05990355461835861,
                    0.0004118939395993948,
                    0.000653077382594347,
                    -0.019962556660175323,
                ],
            ),
            vla_feature_key="observation.images.wrist",
            vendor="Intel",
            model="RealSense D405",
            metadata={"output_width": 640, "output_height": 480},
        ),
    ],
    capabilities=RobotCapabilities(
        can_lift_kg=5.0,
        has_vision=True,
        supported_control_modes=[ControlMode.JOINT_POSITION, ControlMode.GRIPPER_POSITION],
        embodiment_tags=["galaxea_a1"],
    ),
    safety=SafetyEnvelope(
        max_ee_speed_m_s=0.25,
        max_joint_speed_factor=0.05,
        max_force_n=40.0,
        max_torque_nm=40.0,
        deadman_required=True,
    ),
    sdk_kind="closed_with_api",
    hal=HalEntrypoints(
        real="openral_hal.galaxea_a1:GalaxeaA1HAL",
        parameters={
            "defaults": {
                "host": "127.0.0.1",
                "port": 46011,
                "state_timeout_s": 0.5,
                "status_timeout_s": 1.0,
                "feedback_limit_tolerance_rad": 0.01,
                "initial_alignment_tolerance_rad": 0.05,
                "tracker_alignment_timeout_s": 5.0,
                "max_target_step_rad": 0.08,
                "policy_substep_rad": 0.045,
                "command_lease_s": 0.5,
                "idle_timeout_error_mask": 64,
                "gripper_ignored_error_mask": 8,
                "gripper_stroke_min_mm": 0.0,
                "gripper_stroke_max_mm": 104.0,
            }
        },
    ),
)


@dataclass(frozen=True)
class _Snapshot:
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]
    stamp_ns: int
    received_monotonic: float
    status_codes: tuple[int, ...]
    status_received_monotonic: float
    gripper_norm: float
    relay_state: str = "LOCKED"
    staged_position: tuple[float, ...] = ()
    forwarded_position: tuple[float, ...] = ()
    sidecar_action_count: int = 0


class _A1Transport(Protocol):
    def connect(self, hello: dict[str, object], timeout_s: float) -> None: ...

    def close(self) -> None: ...

    def snapshot(self) -> _Snapshot: ...

    def send(self, message: dict[str, object]) -> None: ...

    def estop(self, timeout_s: float) -> None: ...


class _SocketTransport:
    """Loopback JSON-lines client with one reader and one priority writer."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._estop_ack = threading.Event()
        self._outbound_ready = threading.Event()
        self._estop_confirmed = False
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._snapshot: _Snapshot | None = None
        self._pending_joint: dict[str, object] | None = None
        self._pending_gripper: dict[str, object] | None = None
        self._estop_pending = False
        self._identity_verified = False
        self._error: str | None = None

    def connect(self, hello: dict[str, object], timeout_s: float) -> None:
        self._stop.clear()
        self._ready.clear()
        self._estop_ack.clear()
        self._outbound_ready.clear()
        with self._lock:
            self._snapshot = None
            self._pending_joint = None
            self._pending_gripper = None
            self._estop_pending = False
            self._identity_verified = False
            self._estop_confirmed = False
            self._error = None
        try:
            sock = socket.create_connection((self._host, self._port), timeout=timeout_s)
        except OSError as exc:
            raise ROSRuntimeError(
                f"Galaxea A1 sidecar is unavailable at {self._host}:{self._port}: {exc}"
            ) from exc
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
            self._sock = sock
            self._write(hello)
            self._reader_thread = threading.Thread(
                target=self._read_loop, name="openral_a1_sidecar_reader", daemon=True
            )
            self._writer_thread = threading.Thread(
                target=self._write_loop, name="openral_a1_sidecar_writer", daemon=True
            )
            self._reader_thread.start()
            self._writer_thread.start()
        except Exception:
            self.close()
            raise
        if not self._ready.wait(timeout_s):
            self.close()
            raise ROSRuntimeError(
                f"Galaxea A1 sidecar returned no valid state within {timeout_s:.1f} s."
            )
        self._raise_if_failed()

    def close(self) -> None:
        self._stop.set()
        self._outbound_ready.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        current = threading.current_thread()
        for thread in (self._reader_thread, self._writer_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.0)
        self._reader_thread = None
        self._writer_thread = None

    def snapshot(self) -> _Snapshot:
        self._raise_if_failed()
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise ROSRuntimeError("Galaxea A1 sidecar has not produced a state snapshot.")
        return snapshot

    def send(self, message: dict[str, object]) -> None:
        self._raise_if_failed()
        with self._lock:
            if "joint_targets" in message:
                self._pending_joint = message
            if "gripper" in message:
                self._pending_gripper = message
        self._outbound_ready.set()

    def estop(self, timeout_s: float) -> None:
        with self._lock:
            self._estop_pending = True
            self._pending_joint = None
            self._pending_gripper = None
        self._outbound_ready.set()
        self._estop_ack.wait(timeout_s)
        with self._lock:
            confirmed = self._estop_confirmed
        self.close()
        if not confirmed:
            raise ROSRuntimeError(
                "Galaxea A1 sidecar did not confirm that its ROS 1 stack stopped."
            )

    def _read_loop(self) -> None:
        buffer = b""
        try:
            while not self._stop.is_set():
                sock = self._sock
                if sock is None:
                    return
                readable, _, _ = select.select([sock], [], [], 0.02)
                if not readable:
                    continue
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("sidecar closed the connection")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        self._handle(json.loads(line))
        except Exception as exc:
            if not self._stop.is_set():
                self._fail(f"Galaxea A1 sidecar transport failed: {exc}")

    def _write_loop(self) -> None:
        """Drain the latest command independently of inbound state traffic."""
        try:
            while not self._stop.is_set():
                self._outbound_ready.wait()
                self._outbound_ready.clear()
                while not self._stop.is_set():
                    message = self._take_pending_message()
                    if message is None:
                        break
                    self._write(message)
        except Exception as exc:
            if not self._stop.is_set():
                self._fail(f"Galaxea A1 sidecar transport failed: {exc}")

    def _take_pending_message(self) -> dict[str, object] | None:
        """Atomically coalesce the latest arm and gripper targets.

        OpenRAL publishes per-mode action chunks back-to-back. Keeping one
        mailbox per mode prevents a gripper chunk from overwriting the arm
        chunk before the socket worker runs, while still bounding backlog.
        """
        with self._lock:
            if self._estop_pending:
                message: dict[str, object] | None = {"op": "estop"}
                self._estop_pending = False
            else:
                joint = self._pending_joint
                gripper = self._pending_gripper
                self._pending_joint = None
                self._pending_gripper = None
                if joint is None and gripper is None:
                    return None
                stamps: list[int] = []
                for item in (joint, gripper):
                    stamp = None if item is None else item.get("stamp_ns")
                    if isinstance(stamp, int) and not isinstance(stamp, bool):
                        stamps.append(stamp)
                message = {"op": "action", "stamp_ns": max(stamps, default=time.time_ns())}
                if joint is not None:
                    message["joint_targets"] = joint["joint_targets"]
                if gripper is not None:
                    message["gripper"] = gripper["gripper"]
        return message

    def _write(self, message: dict[str, object]) -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionError("sidecar socket is closed")
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        view = memoryview(payload)
        while view:
            _, writable, _ = select.select([], [sock], [], 0.2)
            if not writable:
                raise TimeoutError("sidecar send timed out")
            view = view[sock.send(view) :]

    def _handle(self, raw: object) -> None:
        if not isinstance(raw, dict):
            raise ValueError("sidecar record must be a JSON object")
        op = raw.get("op")
        if op == "hello":
            if raw.get("protocol") != _PROTOCOL_VERSION or raw.get("robot") != "galaxea_a1":
                raise ValueError("sidecar identity does not match Galaxea A1 protocol v1")
            with self._lock:
                self._identity_verified = True
        elif op == "state":
            with self._lock:
                identity_verified = self._identity_verified
            if not identity_verified:
                raise ValueError("sidecar sent state before its identity handshake")
            snapshot = _decode_snapshot(raw)
            with self._lock:
                self._snapshot = snapshot
            self._ready.set()
        elif op == "fault":
            self._fail(str(raw.get("reason", "unknown sidecar fault")))
        elif op == "estopped":
            with self._lock:
                self._estop_confirmed = True
            self._estop_ack.set()
        else:
            raise ValueError(f"unsupported sidecar operation {op!r}")

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._error = reason
        self._ready.set()
        self._estop_ack.set()

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise ROSRuntimeError(error)


def _decode_snapshot(raw: dict[str, object]) -> _Snapshot:
    def _floats(key: str, *, required: bool = True) -> tuple[float, ...]:
        value = raw.get(key)
        if not isinstance(value, list):
            if not required:
                return ()
            raise ValueError(f"state.{key} must be a list")
        result = tuple(float(v) for v in value)
        if not all(math.isfinite(v) for v in result):
            raise ValueError(f"state.{key} contains a non-finite value")
        return result

    names = raw.get("name")
    if names != list(_JOINT_NAMES):
        raise ValueError(f"state.name must be {list(_JOINT_NAMES)!r}, got {names!r}")
    position = _floats("position")
    if len(position) != len(_JOINT_NAMES):
        raise ValueError(f"state.position must have {len(_JOINT_NAMES)} values")
    velocity = _floats("velocity", required=False)
    effort = _floats("effort", required=False)
    for label, values in (("velocity", velocity), ("effort", effort)):
        if values and len(values) != len(_JOINT_NAMES):
            raise ValueError(f"state.{label} must be empty or have {len(_JOINT_NAMES)} values")
    status = raw.get("status_codes")
    if not isinstance(status, list) or len(status) < len(_JOINT_NAMES) + 1:
        raise ValueError("state.status_codes must include six arm motors and the gripper")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_STATUS_MASK
        for value in status
    ):
        raise ValueError("state.status_codes must contain uint32 integers")
    gripper_raw = raw.get("gripper_norm")
    if isinstance(gripper_raw, bool) or not isinstance(gripper_raw, (int, float)):
        raise ValueError("state.gripper_norm must be numeric")
    gripper = float(gripper_raw)
    if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
        raise ValueError("state.gripper_norm must be finite and within [0, 1]")
    relay_state = raw.get("relay_state")
    if relay_state not in {"LOCKED", "ARMING", "ACTIVE", "FAULT"}:
        raise ValueError(f"state.relay_state is invalid: {relay_state!r}")
    staged_position = _floats("staged_position", required=False)
    forwarded_position = _floats("forwarded_position", required=False)
    for label, values in (
        ("staged_position", staged_position),
        ("forwarded_position", forwarded_position),
    ):
        if values and len(values) != len(_JOINT_NAMES):
            raise ValueError(f"state.{label} must be empty or have {len(_JOINT_NAMES)} values")
    now = time.monotonic()
    stamp_raw = raw.get("stamp_ns")
    if isinstance(stamp_raw, bool) or not isinstance(stamp_raw, int) or stamp_raw < 0:
        raise ValueError("state.stamp_ns must be a non-negative integer")
    sidecar_action_count = _decode_nonnegative_int(raw, "accepted_action_count", default=0)
    return _Snapshot(
        position=position,
        velocity=velocity,
        effort=effort,
        stamp_ns=stamp_raw,
        received_monotonic=now,
        status_codes=tuple(status),
        status_received_monotonic=now,
        gripper_norm=gripper,
        relay_state=str(relay_state),
        staged_position=staged_position,
        forwarded_position=forwarded_position,
        sidecar_action_count=sidecar_action_count,
    )


def _decode_nonnegative_int(raw: dict[str, object], key: str, *, default: int | None = None) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"state.{key} must be a non-negative integer")
    return value


class GalaxeaA1HAL(HALBase):
    """OpenRAL real HAL for the six-axis Galaxea A1 plus parallel gripper."""

    description = GALAXEA_A1_DESCRIPTION
    estop_recovery = EStopRecovery.RESTART_REQUIRED

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 46011,
        state_timeout_s: float = 0.5,
        status_timeout_s: float = 1.0,
        feedback_limit_tolerance_rad: float = 0.01,
        initial_alignment_tolerance_rad: float = 0.05,
        tracker_alignment_timeout_s: float = 5.0,
        max_target_step_rad: float = 0.08,
        policy_substep_rad: float = 0.045,
        command_lease_s: float = 0.5,
        idle_timeout_error_mask: int = 64,
        gripper_ignored_error_mask: int = 8,
        gripper_stroke_min_mm: float = 0.0,
        gripper_stroke_max_mm: float = 104.0,
        connect_timeout_s: float = 10.0,
        transport: _A1Transport | None = None,
    ) -> None:
        """Configure the sidecar endpoint and explicit safety deadlines."""
        if not host.strip() or not 0 < int(port) < _MAX_TCP_PORT:
            raise ROSConfigError("GalaxeaA1HAL requires a valid sidecar host and TCP port.")
        positive = {
            "state_timeout_s": state_timeout_s,
            "status_timeout_s": status_timeout_s,
            "initial_alignment_tolerance_rad": initial_alignment_tolerance_rad,
            "tracker_alignment_timeout_s": tracker_alignment_timeout_s,
            "max_target_step_rad": max_target_step_rad,
            "policy_substep_rad": policy_substep_rad,
            "command_lease_s": command_lease_s,
            "connect_timeout_s": connect_timeout_s,
        }
        if any(not math.isfinite(v) or v <= 0.0 for v in positive.values()):
            raise ROSConfigError(
                f"GalaxeaA1HAL timing and motion limits must be positive: {positive}"
            )
        if policy_substep_rad >= max_target_step_rad:
            raise ROSConfigError("policy_substep_rad must be smaller than max_target_step_rad.")
        if not math.isfinite(feedback_limit_tolerance_rad) or feedback_limit_tolerance_rad < 0.0:
            raise ROSConfigError("feedback_limit_tolerance_rad must be finite and non-negative.")
        if (
            not math.isfinite(gripper_stroke_min_mm)
            or not math.isfinite(gripper_stroke_max_mm)
            or gripper_stroke_max_mm <= gripper_stroke_min_mm
        ):
            raise ROSConfigError("gripper_stroke_max_mm must exceed gripper_stroke_min_mm.")
        masks = {
            "idle_timeout_error_mask": idle_timeout_error_mask,
            "gripper_ignored_error_mask": gripper_ignored_error_mask,
        }
        if any(isinstance(v, bool) or not 0 <= int(v) <= _MAX_STATUS_MASK for v in masks.values()):
            raise ROSConfigError(f"GalaxeaA1HAL motor status masks must be uint32 values: {masks}")
        self._host = host
        self._port = int(port)
        self._state_timeout_s = state_timeout_s
        self._status_timeout_s = status_timeout_s
        self._feedback_limit_tolerance = feedback_limit_tolerance_rad
        self._alignment = initial_alignment_tolerance_rad
        self._tracker_alignment_timeout = tracker_alignment_timeout_s
        self._max_step = max_target_step_rad
        self._policy_substep = policy_substep_rad
        self._lease = command_lease_s
        self._idle_mask = int(idle_timeout_error_mask)
        self._gripper_mask = int(gripper_ignored_error_mask)
        self._gripper_min = gripper_stroke_min_mm
        self._gripper_max = gripper_stroke_max_mm
        self._connect_timeout_s = connect_timeout_s
        self._transport: _A1Transport = transport or _SocketTransport(host, self._port)
        self._connected = False
        self._first_joint_command = True
        self._accepted_action_count = 0

    def connect(self) -> None:
        """Connect to the sidecar and require one complete fresh snapshot."""
        if self._connected:
            raise ROSRuntimeError("GalaxeaA1HAL is already connected.")
        hello: dict[str, object] = {
            "op": "hello",
            "protocol": _PROTOCOL_VERSION,
            "robot": "galaxea_a1",
            "joint_names": list(_JOINT_NAMES),
            "joint_position_min": [limits[0] for limits in _JOINT_LIMITS],
            "joint_position_max": [limits[1] for limits in _JOINT_LIMITS],
            "initial_alignment_tolerance_rad": self._alignment,
            "tracker_alignment_timeout_s": self._tracker_alignment_timeout,
            "max_target_step_rad": self._max_step,
            "command_lease_s": self._lease,
            "state_timeout_s": self._state_timeout_s,
            "status_timeout_s": self._status_timeout_s,
            "feedback_limit_tolerance_rad": self._feedback_limit_tolerance,
            "idle_timeout_error_mask": self._idle_mask,
            "gripper_ignored_error_mask": self._gripper_mask,
            "gripper_stroke_min_mm": self._gripper_min,
            "gripper_stroke_max_mm": self._gripper_max,
        }
        try:
            self._transport.connect(hello, self._connect_timeout_s)
            self._fresh_snapshot(require_status=True)
        except Exception:
            self._transport.close()
            raise
        self._connected = True
        self._first_joint_command = True
        self._accepted_action_count = 0
        log.info("hal.connect", robot=self.description.name, sidecar=f"{self._host}:{self._port}")

    def disconnect(self) -> None:
        """Close the sidecar transport idempotently."""
        self._transport.close()
        self._connected = False

    def read_state(self) -> JointState:
        """Return the latest non-stale six-joint feedback snapshot."""
        self._require_connected("read_state")
        state = self._fresh_snapshot(require_status=True)
        return JointState(
            name=list(_JOINT_NAMES),
            position=list(state.position),
            velocity=list(state.velocity),
            effort=list(state.effort),
            stamp_ns=state.stamp_ns,
        )

    def send_action(self, action: Action) -> None:
        """Queue one validated joint or normalized-gripper target."""
        self._require_connected("send_action")
        state = self._fresh_snapshot(require_status=True)
        message: dict[str, object] = {"op": "action", "stamp_ns": action.stamp_ns}
        if action.control_mode == ControlMode.JOINT_POSITION:
            message["joint_targets"] = list(self._validated_joint_target(action, state))
        elif action.control_mode == ControlMode.GRIPPER_POSITION:
            message["gripper"] = self._validated_gripper_target(action)
        else:
            raise ROSConfigError(
                f"GalaxeaA1HAL does not support control mode {action.control_mode.value!r}."
            )
        self._transport.send(message)
        self._accepted_action_count += 1
        if action.control_mode == ControlMode.JOINT_POSITION:
            self._first_joint_command = False

    def _validated_joint_target(self, action: Action, state: _Snapshot) -> tuple[float, ...]:
        self._validate_action_dims(action, len(_JOINT_NAMES))
        if action.gripper:
            raise ROSConfigError("gripper data require GRIPPER_POSITION control mode.")
        if not action.joint_targets:
            raise ROSConfigError("GalaxeaA1HAL requires joint_targets in JOINT_POSITION mode.")
        target = tuple(float(v) for v in action.joint_targets[0])
        if not all(math.isfinite(v) for v in target):
            raise ROSConfigError("Galaxea A1 joint target contains a non-finite value.")
        for name, value, (lower, upper) in zip(_JOINT_NAMES, target, _JOINT_LIMITS, strict=True):
            if not lower <= value <= upper:
                raise ROSConfigError(
                    f"Galaxea A1 joint target {name}={value:.6f} is outside "
                    f"[{lower:.6f}, {upper:.6f}]."
                )
        bound = self._alignment if self._first_joint_command else self._max_step
        if any(
            abs(value - current) > bound
            for value, current in zip(target, state.position, strict=True)
        ):
            label = "initial alignment" if self._first_joint_command else "target step"
            raise ROSConfigError(
                f"Galaxea A1 {label} exceeds the configured {bound:.3f} rad limit."
            )
        return target

    @staticmethod
    def _validated_gripper_target(action: Action) -> float:
        if action.joint_targets:
            raise ROSConfigError("joint_targets require JOINT_POSITION control mode.")
        if not action.gripper:
            raise ROSConfigError("GalaxeaA1HAL requires gripper data in GRIPPER_POSITION mode.")
        gripper = float(action.gripper[0])
        if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ROSConfigError("Galaxea A1 gripper target must be finite and within [0, 1].")
        return gripper

    def health(self) -> HALHealthReport:
        """Return cached hardware health for the lifecycle diagnostics heartbeat."""
        state = self._fresh_snapshot(require_status=True)
        now = time.monotonic()
        return HALHealthReport(
            message="A1 feedback and motor status healthy",
            fields={
                "sidecar": f"{self._host}:{self._port}",
                "policy_substep_rad": f"{self._policy_substep:.6f}",
                "joint_state_age_s": f"{now - state.received_monotonic:.3f}",
                "motor_status_age_s": f"{now - state.status_received_monotonic:.3f}",
                "motor_status_codes": ",".join(str(code) for code in state.status_codes),
                "command_relay_state": state.relay_state,
                "hal_accepted_action_count": str(self._accepted_action_count),
                "sidecar_accepted_action_count": str(state.sidecar_action_count),
                "gripper_position_normalized": f"{state.gripper_norm:.6f}",
                "staged_position": ",".join(f"{value:.6f}" for value in state.staged_position),
                "forwarded_position": ",".join(
                    f"{value:.6f}" for value in state.forwarded_position
                ),
            },
        )

    def estop(self) -> None:
        """Stop the sidecar-owned ROS 1 stack and always raise."""
        log.critical("hal.estop", robot=self.description.name)
        try:
            # The sidecar acknowledges only after tracker, driver, and roscore
            # have exited. Its bounded shutdown allows up to three seconds.
            self._transport.estop(timeout_s=4.0)
        finally:
            self._connected = False
        raise ROSEStopRequested("Emergency stop triggered on Galaxea A1; ROS 1 sidecar stopped.")

    def _fresh_snapshot(self, *, require_status: bool = False) -> _Snapshot:
        state = self._transport.snapshot()
        age = time.monotonic() - state.received_monotonic
        if age > self._state_timeout_s:
            raise ROSPerceptionStale(
                f"Galaxea A1 joint state is {age:.3f} s old (limit {self._state_timeout_s:.3f} s)."
            )
        if require_status:
            status_age = time.monotonic() - state.status_received_monotonic
            if status_age > self._status_timeout_s:
                raise ROSPerceptionStale(
                    f"Galaxea A1 motor status is {status_age:.3f} s old "
                    f"(limit {self._status_timeout_s:.3f} s)."
                )
            for index, code in enumerate(state.status_codes[: _GRIPPER_STATUS_INDEX + 1]):
                allowed = self._idle_mask | (
                    self._gripper_mask if index == _GRIPPER_STATUS_INDEX else 0
                )
                if code & ~allowed:
                    raise ROSRuntimeError(
                        f"Galaxea A1 actuator {index + 1} reports error code {code}."
                    )
        for name, value, (lower, upper) in zip(
            _JOINT_NAMES, state.position, _JOINT_LIMITS, strict=True
        ):
            if not (
                lower - self._feedback_limit_tolerance
                <= value
                <= upper + self._feedback_limit_tolerance
            ):
                raise ROSRuntimeError(
                    f"Galaxea A1 feedback {name}={value:.6f} is outside "
                    f"[{lower:.6f}, {upper:.6f}] by more than the configured "
                    f"{self._feedback_limit_tolerance:.3f} rad tolerance."
                )
        return state
