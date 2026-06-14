"""ADR-0018 F1 — HAL adapter that publishes ``ActionChunk`` on ROS.

`ROSPublishingHAL` satisfies the existing
:class:`openral_hal.protocol.HAL` Protocol but **does not drive motors
directly**. Instead:

* ``send_action`` serialises the :class:`openral_core.Action` into
  ``openral_msgs/ActionChunk`` and publishes it on
  ``/openral/candidate_action`` with RELIABLE / VOLATILE / KL=1 QoS.
* ``read_state`` returns the most recent ``JointState`` cached from a
  ``/joint_states`` subscription opened on a host
  ``rclpy.lifecycle.LifecycleNode``.
* ``connect`` / ``disconnect`` open / close the publisher and
  subscription on that host node — the adapter holds no rclpy node of
  its own (composing into the host's executor keeps QoS / lifecycle /
  shutdown in one place per CLAUDE.md §6.1).

This is the **single change** to the in-process hot path mandated by
ADR-0018 §F1: ``HardwareRunner._tick_impl`` keeps calling
``hal.send_action(action)`` — only the sink moves from motors to a ROS
topic, behind which sits ``safety_node`` (F5) → ``<robot>_hal_node``.

`trace_id` is sourced from the active OTel context per ADR-0018 §6;
``rskill_id`` / ``rskill_revision`` are set per goal by the
``rskill_runner_node`` (an injected getter avoids tight-coupling).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import structlog
from openral_core.exceptions import ROSConfigError, ROSRuntimeError
from openral_core.schemas import Action, ControlMode, JointState, RobotDescription
from openral_observability import propagation

if TYPE_CHECKING:
    from rclpy.lifecycle import LifecycleNode

__all__ = ["ROSPublishingHAL"]

log = structlog.get_logger(__name__)


def _row_major_flatten(rows: list[list[float]] | None) -> list[float]:
    """Flatten a (H, N) action chunk into a row-major flat array.

    Returns an empty list when ``rows`` is ``None`` (the action carries no
    target in this control mode); callers gate on the per-control-mode
    field selection.
    """
    if not rows:
        return []
    out: list[float] = []
    for row in rows:
        out.extend(row)
    return out


def _flatten_rows(rows: list[list[float]] | None, horizon: int) -> tuple[list[float], int, int]:
    """Flatten ``list[list[float]]`` rows + return ``(flat, n_dof, horizon)``.

    ADR-0028b helper for ROSPublishingHAL._action_to_chunk dispatch on
    joint-mode payloads. Raises :class:`ROSConfigError` when the
    payload is empty (a joint action with no target is a programming
    error — the slot dispatcher / legacy path always populates it).
    """
    if not rows:
        raise ROSConfigError(
            "ROSPublishingHAL: joint-mode Action has empty payload (rows is None or [])"
        )
    flat = _row_major_flatten(rows)
    n_dof = len(rows[0]) if rows[0] else 0
    return flat, n_dof, int(horizon or len(rows))


def _flatten_tuple_rows(
    rows: Sequence[tuple[float, ...]] | None, horizon: int, *, expected_n: int
) -> tuple[list[float], int, int]:
    """Flatten a sequence of fixed-width float tuples.

    ADR-0028b helper for cartesian / twist Action payloads. The slot
    dispatcher upstream guarantees each row has exactly ``expected_n``
    components; this check is a runtime guard against future drift.
    """
    if not rows:
        raise ROSConfigError(
            "ROSPublishingHAL: cartesian/twist Action has empty payload (rows is None or [])"
        )
    bad = [(i, len(r)) for i, r in enumerate(rows) if len(r) != expected_n]
    if bad:
        raise ROSConfigError(
            f"ROSPublishingHAL: cartesian/twist rows must have width {expected_n}; "
            f"got rows with widths {bad!r}"
        )
    flat: list[float] = []
    for row in rows:
        flat.extend(float(v) for v in row)
    return flat, expected_n, int(horizon or len(rows))


# Wire-format encoding for `ActionChunk.control_mode` is owned by
# `openral_core.CONTROL_MODE_TO_UINT8` — single source of truth shared
# with the panda_mobile lifecycle node's `_on_safe_action` decoder.
from openral_core import CONTROL_MODE_TO_UINT8 as _CONTROL_MODE_TO_UINT8  # noqa: E402


class ROSPublishingHAL:
    """HAL adapter that publishes ``ActionChunk`` and caches ``/joint_states``.

    Implements the structural :class:`openral_hal.protocol.HAL` Protocol
    without inheriting from it (the Protocol is ``runtime_checkable`` —
    any class with the right shape passes).

    Args:
        node: The host ``rclpy.lifecycle.LifecycleNode`` that owns the
            adapter's publisher / subscription. ``connect`` /
            ``disconnect`` are no-ops outside this node's lifecycle.
        description: The :class:`RobotDescription` for the robot this
            adapter represents. Surfaced via the ``HAL.description``
            attribute consumed by `HardwareRunner` for span attributes
            and per-joint limit lookups.
        skill_id_getter: Zero-arg callable returning the in-flight
            skill id (filled into ``ActionChunk.rskill_id``). The
            ``rskill_runner_node`` swaps this per goal.
        skill_revision_getter: Zero-arg callable returning the
            in-flight skill revision (filled into
            ``ActionChunk.rskill_revision``).
        joint_state_topic: Topic to subscribe for cached
            ``read_state()``. Defaults to ``/joint_states``.
        candidate_action_topic: Topic to publish ``ActionChunk`` on.
            Defaults to ``/openral/candidate_action`` (ADR-0018 §F1).

    Example:
        >>> # Real usage exercised in
        >>> # tests/unit/test_ros_publishing_hal.py against a real
        >>> # rclpy LifecycleNode + the real openral_msgs IDL.
        >>> pass
    """

    description: RobotDescription

    def __init__(
        self,
        *,
        node: LifecycleNode,
        description: RobotDescription,
        skill_id_getter: Callable[[], str] = lambda: "",
        skill_revision_getter: Callable[[], str] = lambda: "",
        joint_state_topic: str = "/joint_states",
        candidate_action_topic: str = "/openral/candidate_action",
    ) -> None:
        """Store references; opens no ROS resources until :meth:`connect`."""
        self._node = node
        self.description = description
        self._skill_id_getter = skill_id_getter
        self._skill_revision_getter = skill_revision_getter
        self._joint_state_topic = joint_state_topic
        self._candidate_action_topic = candidate_action_topic
        self._publisher: Any = None
        self._subscription: Any = None
        # Cached JointState; populated on every /joint_states callback.
        # ``read_state`` raises ROSRuntimeError if this is still ``None``
        # when called (matches the HAL Protocol contract — connect-then-
        # read sequence is enforced).
        self._last_state: JointState | None = None
        self._connected: bool = False

    # ── HAL Protocol ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the publisher on candidate_action + subscriber on joint_states."""
        if self._connected:
            return
        from openral_msgs.msg import ActionChunk  # type: ignore[import-not-found,unused-ignore]
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import JointState as RosJointState

        # ADR-0028b — the slot dispatcher publishes N chunks per policy
        # tick (arm CARTESIAN_DELTA + gripper GRIPPER_POSITION, etc.).
        # KEEP_LAST=1 + back-to-back publishes => the kernel's subscriber
        # buffer holds only the most recent and silently drops the
        # earlier per-tick chunks. Symptom: in deploy_sim, ONLY the last
        # slot's chunks (gripper) reach the HAL, the arm freezes even
        # though policy_step keeps emitting big OSC deltas. Depth=10
        # matches the safety/sensor guidance in CLAUDE.md §3 and is the
        # minimum that survives slot fan-out at 30 Hz tick rate.
        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        try:
            self._publisher = self._node.create_publisher(
                ActionChunk, self._candidate_action_topic, chunk_qos
            )
            self._subscription = self._node.create_subscription(
                RosJointState,
                self._joint_state_topic,
                self._on_joint_state,
                state_qos,
            )
        except Exception as exc:  # reason: surface as typed error
            raise ROSConfigError(f"ROSPublishingHAL.connect failed: {exc!r}") from exc
        self._connected = True
        log.info(
            "ros_publishing_hal.connected",
            candidate_action_topic=self._candidate_action_topic,
            joint_state_topic=self._joint_state_topic,
        )

    def disconnect(self) -> None:
        """Close the publisher + subscription. Idempotent."""
        if not self._connected:
            return
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
        if self._publisher is not None:
            self._node.destroy_publisher(self._publisher)
            self._publisher = None
        self._connected = False

    def read_state(self) -> JointState:
        """Return the latest cached ``JointState`` from ``/joint_states``."""
        if not self._connected:
            raise ROSRuntimeError("ROSPublishingHAL.read_state called before connect")
        if self._last_state is None:
            raise ROSRuntimeError(
                f"no /joint_states message received yet on {self._joint_state_topic!r}"
            )
        return self._last_state

    def send_action(self, action: Action) -> None:
        """Publish ``ActionChunk`` on ``/openral/candidate_action``.

        Does NOT touch motors — the ``safety_node`` (F5) republishes the
        chunk on ``/openral/safe_action``, which the per-robot HAL
        lifecycle node consumes and forwards to the underlying
        controller.
        """
        if not self._connected:
            raise ROSRuntimeError("ROSPublishingHAL.send_action called before connect")
        chunk = self._action_to_chunk(action)
        self._publisher.publish(chunk)

    def estop(self) -> None:
        """Trigger an emergency stop.

        Per CLAUDE.md §10, this raises ``ROSEStopRequested`` so the
        safety supervisor boundary can record + brake. The owning
        lifecycle node also publishes ``std_msgs/Empty`` on
        ``/openral/estop`` (see ``rskill_runner_node.estop``) — defense
        in depth.
        """
        from openral_core.exceptions import ROSEStopRequested

        raise ROSEStopRequested("ROSPublishingHAL.estop()")

    # ── Internals ───────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: object) -> None:
        """Cache the latest ``sensor_msgs/JointState`` as a Pydantic snapshot."""
        # Use ``getattr`` to stay decoupled from the rosidl class shape
        # (which carries no py.typed marker).
        name = list(getattr(msg, "name", []) or [])
        position = list(getattr(msg, "position", []) or [])
        velocity = list(getattr(msg, "velocity", []) or [])
        effort = list(getattr(msg, "effort", []) or [])
        self._last_state = JointState(
            name=name,
            position=position,
            velocity=velocity,
            effort=effort,
            stamp_ns=time.time_ns(),
        )

    def _action_to_chunk(self, action: Action) -> object:
        """Serialise the typed ``Action`` into ``openral_msgs/ActionChunk``.

        ADR-0028b — the wire format is mode-agnostic (``flat`` +
        ``n_dof`` + ``control_mode``); this dispatcher chooses the
        per-mode source field on the Pydantic :class:`Action` and
        flattens it into the ActionChunk's ``flat`` array. The HAL
        decodes the flat array per its ``control_mode`` (the
        panda_mobile HAL already does this for JOINT_POSITION +
        BODY_TWIST; ADR-0028c extends to CARTESIAN_DELTA +
        GRIPPER_POSITION).

        ``n_dof`` carries the per-row width — for cartesian / twist
        modes that's 6, for gripper modes it's 1, for joint modes
        it's the joint count.
        """
        from openral_msgs.msg import ActionChunk

        flat, n_dof, horizon = self._flatten_action_payload(action)

        chunk = ActionChunk()
        # rosidl-generated message classes carry no py.typed marker, so
        # the per-field assignments are duck-typed.
        chunk.control_mode = _CONTROL_MODE_TO_UINT8[action.control_mode]
        chunk.horizon = int(horizon)
        chunk.flat = flat
        chunk.n_dof = int(n_dof) & 0xFF
        chunk.ee_name = action.ee_name or ""
        chunk.frame_id = action.frame_id or ""
        chunk.confidence = float(action.confidence)
        chunk.rskill_id = self._skill_id_getter()
        chunk.rskill_revision = self._skill_revision_getter()
        # ADR-0018 §6 — trace_id is the join key. Source it from the
        # active OTel span context via the existing W3C helper so the
        # field stays in lock-step with the OTel parent.
        chunk.trace_id = propagation.current_traceparent() or ""
        return chunk

    @staticmethod
    def _flatten_action_payload(  # noqa: PLR0911  # reason: one early return per control-mode payload shape
        action: Action,
    ) -> tuple[list[float], int, int]:
        """Dispatch on ``control_mode`` to extract (flat, n_dof, horizon).

        Centralises ADR-0028b's per-mode shape rules so the test
        surface lives next to the serialiser. Raises
        :class:`ROSConfigError` for modes without a defined
        serialisation (today: ``CARTESIAN_POSE`` carries a
        :class:`Pose6D` not a flat tuple — encode separately when the
        first consumer needs it; ``FOOT_PLACEMENT`` /
        ``DEX_HAND_JOINT`` deferred to humanoid work).
        """
        mode = action.control_mode
        # ── Joint modes (matrix [H, N]) ────────────────────────────
        if mode in (ControlMode.JOINT_POSITION, ControlMode.JOINT_TRAJECTORY):
            return _flatten_rows(action.joint_targets, action.horizon)
        if mode is ControlMode.JOINT_VELOCITY:
            return _flatten_rows(action.joint_velocities, action.horizon)
        if mode is ControlMode.JOINT_TORQUE:
            return _flatten_rows(action.joint_torques, action.horizon)
        # ── Cartesian delta / twist (list of 6-tuples) ─────────────
        if mode is ControlMode.CARTESIAN_DELTA:
            return _flatten_tuple_rows(action.cartesian_delta, action.horizon, expected_n=6)
        if mode is ControlMode.CARTESIAN_TWIST:
            return _flatten_tuple_rows(action.cartesian_twist, action.horizon, expected_n=6)
        # ── Body twist (list of 6-tuples) ──────────────────────────
        if mode is ControlMode.BODY_TWIST:
            return _flatten_tuple_rows(action.body_twist, action.horizon, expected_n=6)
        # ── Gripper (flat list of 1-D commands per horizon step) ───
        if mode in (ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION):
            gripper = list(action.gripper or [])
            if not gripper:
                raise ROSConfigError(
                    f"ROSPublishingHAL: {mode.value} action has empty gripper payload"
                )
            horizon = action.horizon or len(gripper)
            return [float(v) for v in gripper], 1, int(horizon)
        # ── ADR-0028d — sim-only composite mux flag, 1-D per step ──
        if mode is ControlMode.COMPOSITE_MODE:
            composite = list(action.composite_mode or [])
            if not composite:
                raise ROSConfigError(
                    "ROSPublishingHAL: composite_mode action has empty composite_mode payload"
                )
            horizon = action.horizon or len(composite)
            return [float(v) for v in composite], 1, int(horizon)
        # ── Modes deliberately unsupported on the F1 wire ──────────
        raise ROSConfigError(
            f"ROSPublishingHAL does not serialise {action.control_mode!r} actions yet. "
            "Supported: joint_position, joint_velocity, joint_torque, joint_trajectory, "
            "cartesian_delta, cartesian_twist, body_twist, gripper_binary, "
            "gripper_position, composite_mode. cartesian_pose / foot_placement / "
            "dex_hand_joint are tracked but not yet wired to the typed wire format."
        )
