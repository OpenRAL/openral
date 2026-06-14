#!/usr/bin/env python3
"""ADR-0018 F5 — Day-1 safety_node pass-through (Python).

Owns the chunk-rate safety boundary on the OpenRAL graph:

* Subscribes to ``/openral/candidate_action`` (``openral_msgs/ActionChunk``).
* Publishes ``/openral/safe_action`` (same type) when the envelope checks
  pass.
* Publishes ``/openral/estop`` (``std_msgs/Empty``) when they fail.
* Subscribes to ``/openral/estop`` (defense in depth, CLAUDE.md §1.5) —
  external estop latches the node so subsequent candidates are dropped
  until reset.
* Exposes ``/openral/estop_reset`` (``std_srvs/Trigger``) — explicit
  recovery only; ``ROSEStopRequested`` is never auto-cleared
  (CLAUDE.md §10).
* Publishes 1 Hz ``/diagnostics`` via
  :class:`openral_observability.DiagnosticsHeartbeat` (ADR-0018 F8).

This node is the topic-shape lock for ADR-0018 step 1: the topic
contract is real and tested; the envelope checks are deliberately
minimal so the C++ kernel that lands in ADR-0020 can replace internals
**behind the same topic surface** without renegotiating the graph. Per
CLAUDE.md §1.5 ("Python proposes, C++ disposes") and §7.7 (safety
working-group review), any addition of enforcement beyond what is in
this file requires safety-WG sign-off.

Day-1 envelope checks (stubbed but real):

* ``n_dof`` mismatch vs ``--ros-args -p n_dof:=N`` parameter (``n_dof``
  defaults to ``-1`` meaning "do not enforce" so a launch without an
  explicit value passes through).
* First-row joint targets vs configured per-joint position limits.

On envelope violation:

1. Drop the candidate (do not republish on ``/openral/safe_action``).
2. Publish ``std_msgs/Empty`` on ``/openral/estop``.
3. Latch internal estop state until ``/openral/estop_reset`` is called
   AND ≥``estop_reset_cooldown_s`` (default 500 ms) have passed since
   the last estop publish.

Stub for velocity / force / workspace lands when the C++ kernel does
(ADR-0020).
"""

from __future__ import annotations

import sys
import time
from typing import Any

import rclpy
from openral_observability import safety_span, semconv
from rclpy.executors import ExternalShutdownException
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

# Identity label surfaced as ``safety.kernel`` on every safety.check span.
# Matches the ``SAFETY_KERNEL_NULL`` semconv constant by spirit — the
# Day-1 passthrough is functionally a null kernel that only enforces
# n_dof + per-joint limits, leaving velocity/force/workspace to the C++
# kernel that lands in ADR-0020.
_KERNEL_LABEL_PASSTHROUGH = "passthrough"

__all__ = [
    "DEFAULT_ESTOP_RESET_COOLDOWN_S",
    "SafetyPassthroughNode",
    "SafetySupervisorNode",
    "main",
]

# Default cooldown between an estop publish and the first successful
# ``/openral/estop_reset`` call. Configurable per-launch via the
# ``estop_reset_cooldown_s`` parameter — tests use a larger value to
# avoid races, production keeps the 500 ms default.
DEFAULT_ESTOP_RESET_COOLDOWN_S = 0.5


class SafetyPassthroughNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy untyped
    """ADR-0018 F5 — Day-1 safety_node pass-through.

    Owns ``/openral/candidate_action → /openral/safe_action`` plus the
    ``/openral/estop`` / ``/openral/estop_reset`` pair. See module
    docstring for the full contract.

    Parameters (declared on ``__init__``):

    * ``robot_name`` (str): Short robot label for the ``hardware_id``
      diagnostic field. Default ``"robot"``.
    * ``n_dof`` (int): Expected joints in the ``ActionChunk.flat`` row.
      Default ``-1`` (no enforcement). Set this to the robot's DOF in
      production launches.
    * ``min_joint`` / ``max_joint`` (float[]): Per-joint position
      limits. Same length as ``n_dof`` when ``n_dof > 0``; empty list
      disables the limit check.
    * ``estop_reset_cooldown_s`` (float): Cooldown between an estop
      publish and a valid ``/openral/estop_reset``.
    """

    def __init__(self, node_name: str = "openral_safety") -> None:
        """Declare parameters; opens no resources until ``on_configure``."""
        from rcl_interfaces.msg import ParameterDescriptor

        super().__init__(node_name)
        self.declare_parameter("robot_name", "robot")
        # n_dof default -1 keeps the node usable in tests that do not
        # care about envelope enforcement (just verifying pass-through).
        self.declare_parameter("n_dof", -1)
        # ``declare_parameter(name, [])`` infers the parameter type as
        # PARAMETER_NOT_SET, which prevents ``set_parameters`` from
        # subsequently accepting a typed double array. Use dynamic_typing
        # so launches / tests can override with a real DOUBLE_ARRAY value.
        _dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("min_joint", None, _dyn)
        self.declare_parameter("max_joint", None, _dyn)
        self.declare_parameter("estop_reset_cooldown_s", DEFAULT_ESTOP_RESET_COOLDOWN_S)
        # ADR-0028b per-control-mode envelope bounds. Sentinel ``-1.0``
        # means "no enforcement declared, skip the check" — same
        # semantics as ``n_dof=-1`` above. Launches that route
        # cartesian / twist / gripper chunks (panda_mobile + RoboCasa
        # OSC rSkills) override these with the matching SafetyEnvelope
        # values declared on the robot.yaml. Strictly additive: a
        # legacy launch that sets none of these passes every non-joint
        # chunk through verbatim, exactly as before this commit.
        self.declare_parameter("max_cartesian_step_m", -1.0)
        self.declare_parameter("max_cartesian_step_rad", -1.0)
        self.declare_parameter("max_ee_speed_m_s", -1.0)
        self.declare_parameter("max_ee_angular_speed_rad_s", -1.0)
        self.declare_parameter("max_base_linear_speed_m_s", -1.0)
        self.declare_parameter("max_base_angular_speed_rad_s", -1.0)
        self.declare_parameter("gripper_min", -1.0)
        self.declare_parameter("gripper_max", -1.0)

        self._candidate_sub: Any = None
        self._safe_pub: Any = None
        self._estop_pub: Any = None
        self._estop_sub: Any = None
        self._reset_srv: Any = None
        self._heartbeat: Any = None

        # State.
        self._estopped: bool = False
        self._last_estop_ns: int = 0
        self._chunks_passed: int = 0
        self._chunks_dropped: int = 0
        self._last_drop_reason: str = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Open publishers, subscribers, service, and diagnostics heartbeat."""
        del state
        from diagnostic_msgs.msg import DiagnosticArray  # noqa: F401  # ensure available
        from openral_msgs.msg import ActionChunk
        from openral_observability import DiagnosticsHeartbeat, Level
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from std_msgs.msg import Empty
        from std_srvs.srv import Trigger

        chunk_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        estop_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self._safe_pub = self.create_publisher(ActionChunk, "/openral/safe_action", chunk_qos)
        self._estop_pub = self.create_publisher(Empty, "/openral/estop", estop_qos)
        self._candidate_sub = self.create_subscription(
            ActionChunk,
            "/openral/candidate_action",
            self._on_candidate_action,
            chunk_qos,
        )
        # Defense-in-depth: external estop publishers (deadman watchdog,
        # hardware estop pendant, human estop forwarder, the C++ kernel
        # itself) also latch us so we stop forwarding candidates without
        # waiting for the operator to reset (CLAUDE.md §1.5).
        self._estop_sub = self.create_subscription(
            Empty, "/openral/estop", self._on_external_estop, estop_qos
        )
        self._reset_srv = self.create_service(Trigger, "/openral/estop_reset", self._on_estop_reset)

        robot_name: str = self.get_parameter("robot_name").get_parameter_value().string_value

        def _status() -> tuple[int, str, dict[str, str]]:
            if self._estopped:
                return (
                    Level.ERROR,
                    "estop latched",
                    {
                        "robot": robot_name,
                        "passed": str(self._chunks_passed),
                        "dropped": str(self._chunks_dropped),
                        "last_drop_reason": self._last_drop_reason or "—",
                    },
                )
            return (
                Level.OK,
                "passthrough active",
                {
                    "robot": robot_name,
                    "passed": str(self._chunks_passed),
                    "dropped": str(self._chunks_dropped),
                },
            )

        self._heartbeat = DiagnosticsHeartbeat(
            self,
            hardware_id=f"openral_safety:{robot_name}",
            component_name="openral_safety",
            status_fn=_status,
        )
        self._heartbeat.create_publisher()
        self.get_logger().info(
            f"openral_safety configured (robot={robot_name}); "
            "candidate_action → safe_action passthrough armed."
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Start the diagnostics heartbeat."""
        del state
        if self._heartbeat is not None:
            self._heartbeat.start()
        self.get_logger().info("openral_safety activated.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Stop the heartbeat. Topic surface stays open for clean teardown."""
        del state
        if self._heartbeat is not None:
            self._heartbeat.stop()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Release topics, service, subscriptions, and heartbeat publisher."""
        del state
        if self._heartbeat is not None:
            self._heartbeat.destroy()
            self._heartbeat = None
        if self._reset_srv is not None:
            self.destroy_service(self._reset_srv)
            self._reset_srv = None
        if self._candidate_sub is not None:
            self.destroy_subscription(self._candidate_sub)
            self._candidate_sub = None
        if self._estop_sub is not None:
            self.destroy_subscription(self._estop_sub)
            self._estop_sub = None
        if self._safe_pub is not None:
            self.destroy_publisher(self._safe_pub)
            self._safe_pub = None
        if self._estop_pub is not None:
            self.destroy_publisher(self._estop_pub)
            self._estop_pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Force cleanup."""
        return self.on_cleanup(state)

    # ── Envelope check + publication ─────────────────────────────────────────

    def _on_candidate_action(self, msg: object) -> None:
        """Validate one ``ActionChunk`` and forward it on ``/openral/safe_action``.

        Per ADR-0018 §5:

        * Drop and estop on envelope violation.
        * Drop (no estop) when an estop is already latched — recovery must
          go through ``/openral/estop_reset``.

        Emits a ``safety.check`` OTel span per candidate so the dashboard's
        Safety card + Safety Check Ledger populate. ``safety.severity``
        rides ``"info"`` on pass-through, ``"warn"`` on a drop due to a
        latched estop (no fresh estop fired), and ``"violation"`` on an
        envelope failure (which also re-fires ``/openral/estop``).
        """
        with safety_span(
            check_name="envelope",
            kernel=_KERNEL_LABEL_PASSTHROUGH,
        ) as span:
            # If currently estopped, drop without republishing — the latch
            # only clears via /openral/estop_reset.
            if self._estopped:
                self._chunks_dropped += 1
                self._last_drop_reason = "estop_latched"
                span.set_attribute(semconv.SAFETY_SEVERITY, "warn")
                span.set_attribute(semconv.SAFETY_CLAMPED, False)
                span.set_attribute("safety.drop_reason", "estop_latched")
                self.get_logger().debug(
                    "dropping candidate_action while estop latched: "
                    f"skill={msg.rskill_id!r} trace={msg.trace_id!r}"
                )
                return

            kind, reason = self._envelope_violation(msg)
            if kind is not None:
                span.set_attribute(semconv.SAFETY_SEVERITY, "violation")
                span.set_attribute(semconv.SAFETY_CLAMPED, False)
                span.set_attribute("safety.drop_reason", kind)
                span.set_attribute("safety.violation_reason", reason)
                self._handle_violation(msg, kind=kind, reason=reason)
                return

            # Pass through — publish the same payload on /openral/safe_action.
            # We forward the exact message (no field rewrites) so trace_id /
            # rskill_id stay attached for the F7 correlator and `openral replay`.
            span.set_attribute(semconv.SAFETY_CLAMPED, False)
            assert self._safe_pub is not None  # invariant on active state
            self._safe_pub.publish(msg)
            self._chunks_passed += 1

    def _envelope_violation(  # noqa: PLR0911  # reason: dispatches over control-mode families; each mode is a single early return
        self, msg: object
    ) -> tuple[str | None, str]:
        """Return ``(violation_kind, reason)`` or ``(None, '')`` if OK.

        Dispatches on the chunk's ``control_mode`` (ADR-0028b). Joint
        chunks keep their Day-1 ``n_dof`` + per-joint position bounds
        verbatim; cartesian / twist / gripper chunks get their own
        per-mode bound checks. All new bounds default to ``-1.0``
        ("no enforcement declared, skip") so a legacy launch that
        doesn't override them keeps passing every non-joint chunk
        through, exactly as before this commit.
        """
        from openral_core import UINT8_TO_CONTROL_MODE
        from openral_core.schemas import ControlMode

        mode_uint8 = int(getattr(msg, "control_mode", 0) or 0)
        mode = UINT8_TO_CONTROL_MODE.get(mode_uint8)
        if mode is None:
            return (
                "control_mode",
                f"unknown control_mode uint8 {mode_uint8}; refusing to validate",
            )

        flat = list(getattr(msg, "flat", []) or [])
        n_dof = int(getattr(msg, "n_dof", 0) or 0)

        # ── Joint modes — Day-1 path (unchanged) ─────────────────────
        if mode in (
            ControlMode.JOINT_POSITION,
            ControlMode.JOINT_VELOCITY,
            ControlMode.JOINT_TORQUE,
            ControlMode.JOINT_TRAJECTORY,
        ):
            return self._envelope_violation_joint(flat=flat, n_dof=n_dof)

        # ── Cartesian / twist / gripper — ADR-0028b per-mode checks ──
        if mode is ControlMode.CARTESIAN_DELTA:
            return self._envelope_violation_cartesian_delta(flat=flat, n_dof=n_dof)
        if mode is ControlMode.CARTESIAN_TWIST:
            return self._envelope_violation_cartesian_twist(flat=flat, n_dof=n_dof)
        if mode is ControlMode.BODY_TWIST:
            return self._envelope_violation_body_twist(flat=flat, n_dof=n_dof)
        if mode in (ControlMode.GRIPPER_BINARY, ControlMode.GRIPPER_POSITION):
            return self._envelope_violation_gripper(flat=flat)

        # ── Modes the supervisor doesn't enforce yet (cartesian pose,
        # foot placement, dex hand joints) pass through. The HAL
        # whitelist downstream rejects what it can't consume.
        return (None, "")

    def _envelope_violation_joint(self, *, flat: list[float], n_dof: int) -> tuple[str | None, str]:
        """Day-1 joint envelope: n_dof + per-joint position bounds.

        Behaviour is byte-identical to the pre-ADR-0028b
        ``_envelope_violation`` body for any chunk whose control_mode
        is a JOINT_* mode. Tests in ``test_supervisor_node.py`` pin
        this — adding the per-mode dispatch must not regress the
        joint-only path.
        """
        # n_dof check (skipped when configured -1).
        expected_dof: int = self.get_parameter("n_dof").get_parameter_value().integer_value
        if expected_dof > 0 and n_dof != expected_dof:
            return ("n_dof", f"expected n_dof={expected_dof}, got {n_dof}")

        # Joint limit check (skipped when either array unset).
        min_param = self.get_parameter("min_joint").get_parameter_value()
        max_param = self.get_parameter("max_joint").get_parameter_value()
        min_joint = list(min_param.double_array_value)
        max_joint = list(max_param.double_array_value)
        if min_joint and max_joint:
            dof = n_dof if n_dof > 0 else len(min_joint)
            if len(min_joint) != dof or len(max_joint) != dof:
                return (
                    "envelope_config",
                    "min_joint / max_joint length mismatch with n_dof",
                )
            if len(flat) < dof:
                return ("flat", f"flat shorter than n_dof ({len(flat)} < {dof})")
            row0 = flat[:dof]
            for i, v in enumerate(row0):
                if v < min_joint[i] or v > max_joint[i]:
                    return (
                        "workspace",
                        f"joint[{i}]={v:.4f} out of [{min_joint[i]:.4f}, {max_joint[i]:.4f}]",
                    )
        return (None, "")

    def _envelope_violation_cartesian_delta(
        self, *, flat: list[float], n_dof: int
    ) -> tuple[str | None, str]:
        """CARTESIAN_DELTA: bound the Euclidean step over (xyz) and (rotvec).

        Chunk layout (per ROSPublishingHAL._flatten_action_payload):
        each row is ``[dx, dy, dz, rx, ry, rz]`` (n_dof=6). Bound the
        xyz triplet against ``max_cartesian_step_m`` and the rotvec
        triplet against ``max_cartesian_step_rad``. Each parameter
        ``-1.0`` skips its check.
        """
        cart_step_m = float(
            self.get_parameter("max_cartesian_step_m").get_parameter_value().double_value
        )
        cart_step_rad = float(
            self.get_parameter("max_cartesian_step_rad").get_parameter_value().double_value
        )
        if cart_step_m < 0.0 and cart_step_rad < 0.0:
            return (None, "")
        if n_dof != 6 or len(flat) < 6:
            return (
                "cartesian_shape",
                "cartesian_delta chunk must carry 6-vec row; "
                f"got n_dof={n_dof} flat_len={len(flat)}",
            )
        row0 = flat[:6]
        if cart_step_m >= 0.0:
            mag_xyz = (row0[0] ** 2 + row0[1] ** 2 + row0[2] ** 2) ** 0.5
            if mag_xyz > cart_step_m:
                return (
                    "cartesian_step",
                    f"|dxyz|={mag_xyz:.4f} > max_cartesian_step_m={cart_step_m:.4f}",
                )
        if cart_step_rad >= 0.0:
            mag_rot = (row0[3] ** 2 + row0[4] ** 2 + row0[5] ** 2) ** 0.5
            if mag_rot > cart_step_rad:
                return (
                    "cartesian_step_rot",
                    f"|drotvec|={mag_rot:.4f} > max_cartesian_step_rad={cart_step_rad:.4f}",
                )
        return (None, "")

    def _envelope_violation_cartesian_twist(
        self, *, flat: list[float], n_dof: int
    ) -> tuple[str | None, str]:
        """Bound a CARTESIAN_TWIST chunk by the EE speed limits.

        Linear bound from ``max_ee_speed_m_s``, angular from
        ``max_ee_angular_speed_rad_s``.

        Chunk layout: ``[vx, vy, vz, wx, wy, wz]`` (n_dof=6).
        """
        ee_lin = float(self.get_parameter("max_ee_speed_m_s").get_parameter_value().double_value)
        ee_ang = float(
            self.get_parameter("max_ee_angular_speed_rad_s").get_parameter_value().double_value
        )
        if ee_lin < 0.0 and ee_ang < 0.0:
            return (None, "")
        if n_dof != 6 or len(flat) < 6:
            return (
                "twist_shape",
                "cartesian_twist chunk must carry 6-vec row; "
                f"got n_dof={n_dof} flat_len={len(flat)}",
            )
        row0 = flat[:6]
        if ee_lin >= 0.0:
            v = (row0[0] ** 2 + row0[1] ** 2 + row0[2] ** 2) ** 0.5
            if v > ee_lin:
                return (
                    "ee_linear_speed",
                    f"|v_ee|={v:.4f} > max_ee_speed_m_s={ee_lin:.4f}",
                )
        if ee_ang >= 0.0:
            w = (row0[3] ** 2 + row0[4] ** 2 + row0[5] ** 2) ** 0.5
            if w > ee_ang:
                return (
                    "ee_angular_speed",
                    f"|w_ee|={w:.4f} > max_ee_angular_speed_rad_s={ee_ang:.4f}",
                )
        return (None, "")

    def _envelope_violation_body_twist(
        self, *, flat: list[float], n_dof: int
    ) -> tuple[str | None, str]:
        """Bound a BODY_TWIST chunk by the base speed limits.

        Linear from ``max_base_linear_speed_m_s``, angular from
        ``max_base_angular_speed_rad_s``.

        Chunk layout: ``[vx, vy, vz, wx, wy, wz]`` (n_dof=6). On planar
        bases only ``vx``, ``vy``, ``wz`` are typically non-zero — the
        bound is Euclidean over each triplet so it works on both
        planar and 6-DoF bases.
        """
        base_lin = float(
            self.get_parameter("max_base_linear_speed_m_s").get_parameter_value().double_value
        )
        base_ang = float(
            self.get_parameter("max_base_angular_speed_rad_s").get_parameter_value().double_value
        )
        if base_lin < 0.0 and base_ang < 0.0:
            return (None, "")
        if n_dof != 6 or len(flat) < 6:
            return (
                "body_twist_shape",
                f"body_twist chunk must carry 6-vec row; got n_dof={n_dof} flat_len={len(flat)}",
            )
        row0 = flat[:6]
        if base_lin >= 0.0:
            v = (row0[0] ** 2 + row0[1] ** 2 + row0[2] ** 2) ** 0.5
            if v > base_lin:
                return (
                    "base_linear_speed",
                    f"|v_base|={v:.4f} > max_base_linear_speed_m_s={base_lin:.4f}",
                )
        if base_ang >= 0.0:
            w = (row0[3] ** 2 + row0[4] ** 2 + row0[5] ** 2) ** 0.5
            if w > base_ang:
                return (
                    "base_angular_speed",
                    f"|w_base|={w:.4f} > max_base_angular_speed_rad_s={base_ang:.4f}",
                )
        return (None, "")

    def _envelope_violation_gripper(self, *, flat: list[float]) -> tuple[str | None, str]:
        """GRIPPER_*: clamp width to the ``[gripper_min, gripper_max]`` range.

        The launch sources these parameters from the robot.yaml's
        gripper joint's ``position_limits`` (when present); ``-1.0``
        on either end skips that side of the bound.
        """
        gmin = float(self.get_parameter("gripper_min").get_parameter_value().double_value)
        gmax = float(self.get_parameter("gripper_max").get_parameter_value().double_value)
        if gmin < 0.0 and gmax < 0.0:
            return (None, "")
        if not flat:
            return ("gripper_shape", "gripper chunk has empty flat array")
        w = float(flat[0])
        if gmin >= 0.0 and w < gmin:
            return ("gripper_range", f"width={w:.4f} < gripper_min={gmin:.4f}")
        if gmax >= 0.0 and w > gmax:
            return ("gripper_range", f"width={w:.4f} > gripper_max={gmax:.4f}")
        return (None, "")

    def _handle_violation(self, msg: object, *, kind: str, reason: str) -> None:
        """Drop the candidate, fire estop, log structured."""
        from std_msgs.msg import Empty

        self._chunks_dropped += 1
        self._last_drop_reason = kind
        self._estopped = True
        self._last_estop_ns = time.time_ns()
        assert self._estop_pub is not None  # invariant on active state
        self._estop_pub.publish(Empty())
        # ADR-0018 §5 — log structured for the F7 query-time correlator.
        self.get_logger().error(
            "safety.envelope_violation "
            f"kind={kind!r} reason={reason!r} "
            f"rskill_id={msg.rskill_id!r} trace_id={msg.trace_id!r}"
        )

    # ── External estop subscription (defense in depth) ───────────────────────

    def _on_external_estop(self, _msg: object) -> None:
        """Latch on any external ``/openral/estop`` publication.

        Note that the node publishes ``/openral/estop`` itself on internal
        envelope violation, so it will see its own message — but
        ``self._estopped`` is already True by the time this callback
        fires, so the latch is idempotent.
        """
        if not self._estopped:
            self._estopped = True
            self._last_estop_ns = time.time_ns()
            self._last_drop_reason = "external_estop"
            self.get_logger().warning("safety.external_estop_received: latching node")

    # ── /openral/estop_reset ─────────────────────────────────────────────────

    def _on_estop_reset(self, request: object, response: object) -> object:
        """Service callback: clear the estop latch after the cooldown."""
        del request
        if not self._estopped:
            response.success = True
            response.message = "no estop to reset"
            return response

        cooldown_s: float = (
            self.get_parameter("estop_reset_cooldown_s").get_parameter_value().double_value
        )
        elapsed_ns = time.time_ns() - self._last_estop_ns
        if elapsed_ns < int(cooldown_s * 1e9):
            response.success = False
            response.message = f"cooldown not elapsed ({elapsed_ns / 1e9:.3f}s < {cooldown_s}s)"
            return response

        self._estopped = False
        self._last_drop_reason = ""
        self.get_logger().info("safety.estop_reset succeeded")
        response.success = True
        response.message = "estop cleared"
        return response


# Back-compat alias — the original skeleton class name still imports.
SafetySupervisorNode = SafetyPassthroughNode


def main(args: list[str] | None = None) -> int:
    """Entry point for ``ros2 run openral_safety supervisor_node``."""
    from openral_observability import configure_observability

    # Idempotent + no-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset.
    # Drives the safety.check spans onto the same OTLP collector the
    # rest of the graph is publishing to.
    configure_observability(service_name="openral.safety")

    rclpy.init(args=args)
    try:
        node = SafetyPassthroughNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if already shut down
    return 0


if __name__ == "__main__":
    sys.exit(main())
