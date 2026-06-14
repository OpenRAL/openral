"""Generic mobile-base ROS wiring for the manifest-driven HAL lifecycle node.

Sibling of :class:`~openral_hal.sim_sensor_bridge.SimSensorBridge`. Where the
sensor bridge owns cameras / depth / scan / viewer, this bridge owns the
**planar mobile base** streams every wheeled/holonomic robot needs:

1. **``/odom``** (``nav_msgs/Odometry``) — pose from the HAL's ``base_pose``
   (or the richer ``base_pose_6dof`` when the HAL surfaces a robocasa-style
   6-DoF proprio), twist from the HAL's latched ``base_twist``.
2. **``odom -> base_link`` TF** — the same transform on ``/tf`` (slam_toolbox +
   Nav2 resolve frames through TF).
3. **``/cmd_vel`` → ``BODY_TWIST``** — Nav2 / teleop publish
   ``geometry_msgs/Twist``; the bridge maps each message to a 6-vec BODY_TWIST
   :class:`~openral_core.schemas.Action` and applies it via the node's
   ``_send_action_traced`` (ADR-0024 out-of-scope: this path intentionally
   bypasses the OpenRAL safety supervisor — Nav2's ``velocity_smoother`` caps
   velocity).

The manifest-driven :class:`~openral_hal.lifecycle.ManifestHALLifecycleNode`
attaches this bridge in ``on_activate_post_subs`` **iff the manifest declares
``base_joints``** — so adding a mobile robot needs only that manifest field, no
per-robot lifecycle subclass (issue #191 Phase 3). Frame ids come from the
robot's :class:`~openral_core.RobotDescription` (``odom_frame`` / ``base_frame``)
so nothing is hardcoded (CLAUDE.md §2 — TF2 is the only source of frames).

Lifted verbatim from the panda_mobile bespoke node's ``_publish_odom`` /
``_on_cmd_vel`` so nav-stack behaviour stays bit-identical.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openral_core import RobotDescription


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return the ``(x, y, z, w)`` quaternion for a yaw-only rotation.

    Used by ``MobileBaseBridge._publish_odom`` for both the Odometry orientation
    and the TF rotation. Pure function so it is hermetically testable without
    rclpy.

    Args:
        yaw: Rotation about z, in radians.

    Returns:
        Tuple ``(x, y, z, w)`` — the unit quaternion for ``Rz(yaw)``.

    Example:
        >>> quaternion_from_yaw(0.0)
        (0.0, 0.0, 0.0, 1.0)
    """
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class MobileBaseBridge:
    """Owns ``/odom`` + ``odom->base_link`` TF + ``/cmd_vel``→BODY_TWIST for a node.

    Attach from a lifecycle node's ``on_activate_post_subs`` when the robot has a
    planar base (``RobotDescription.base_joints``); call :meth:`teardown` from
    ``on_deactivate_pre_teardown``. The HAL must expose ``base_pose``
    (``(x, y, yaw)``); ``base_pose_6dof()`` and ``base_twist`` are used when
    present and degrade gracefully when absent.
    """

    def __init__(
        self,
        node: Any,
        hal: Any,
        description: RobotDescription,
        *,
        odom_rate_hz: float = 20.0,
        cmd_vel_topic: str = "/cmd_vel",
        proprio: Any = None,
    ) -> None:
        """Bind node + HAL + manifest; opens no publishers until :meth:`setup`.

        ``proprio`` (ADR-0049): when supplied (sim-attached HALs), odom is NOT
        published from a timer on the executor thread — that thread is busy
        stepping/rendering the sim, which starved ``odom->base_link`` to ~1.8 Hz.
        Instead the node's dedicated publisher thread calls
        :meth:`publish_from_snapshot`, which reads this plain-data snapshot rather
        than the simulator. ``None`` (real HALs) keeps the legacy odom timer
        reading ``hal.base_pose`` directly.
        """
        self._node = node
        self._hal = hal
        self._description = description
        self._odom_rate_hz = odom_rate_hz
        self._cmd_vel_topic = cmd_vel_topic
        self._proprio = proprio
        self._odom_frame = description.odom_frame
        self._base_frame = description.base_frame
        self._odom_pub: Any = None
        self._tf_broadcaster: Any = None
        self._odom_timer: Any = None
        self._cmd_vel_sub: Any = None

    def setup(self) -> None:
        """Create the ``/odom`` publisher + TF broadcaster + timer + ``/cmd_vel`` sub."""
        from nav_msgs.msg import Odometry
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
        from tf2_ros import TransformBroadcaster

        # /odom is RELIABLE — Nav2 wants every sample.
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._odom_pub = self._node.create_publisher(Odometry, "/odom", odom_qos)
        self._tf_broadcaster = TransformBroadcaster(self._node)
        # ADR-0049 — sim-attached HALs publish odom from the node's dedicated
        # thread (:meth:`publish_from_snapshot`); only the legacy (real-HAL /
        # in-process-twin) path drives it from a timer on the executor thread.
        if self._proprio is None:
            self._odom_timer = self._node.create_timer(
                1.0 / max(self._odom_rate_hz, 1.0), self._publish_odom
            )

        # /cmd_vel → BODY_TWIST bridge (ADR-0024 out-of-scope path). Empty topic
        # disables the subscription (purely Action-driven).
        if self._cmd_vel_topic:
            from geometry_msgs.msg import Twist

            cmd_vel_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._cmd_vel_sub = self._node.create_subscription(
                Twist, self._cmd_vel_topic, self._on_cmd_vel, cmd_vel_qos
            )
        self._node.get_logger().info(
            f"MobileBaseBridge: /odom @ {self._odom_rate_hz:.1f} Hz, "
            f"TF {self._odom_frame}->{self._base_frame}, "
            f"cmd_vel={self._cmd_vel_topic or '(disabled)'}"
        )

    def teardown(self) -> None:
        """Stop the timer + destroy the publishers/subscription. Idempotent."""
        if self._odom_timer is not None:
            self._odom_timer.cancel()
            self._odom_timer = None
        if self._odom_pub is not None:
            self._node.destroy_publisher(self._odom_pub)
            self._odom_pub = None
        self._tf_broadcaster = None
        if self._cmd_vel_sub is not None:
            self._node.destroy_subscription(self._cmd_vel_sub)
            self._cmd_vel_sub = None

    def publish_from_snapshot(self) -> None:
        """Publish one ``/odom`` + TF sample (ADR-0049 dedicated-thread entry).

        Called from the node's publisher thread for sim-attached HALs; reads the
        proprio snapshot, never the simulator. Thin alias over :meth:`_publish_odom`
        (which already branches on ``self._proprio``) so the threading contract is
        explicit at the call site.
        """
        self._publish_odom()

    def _publish_odom(self) -> None:
        """Construct + publish ``/odom`` and broadcast the ``odom->base_link`` TF."""
        if self._odom_pub is None or self._hal is None:
            return
        from nav_msgs.msg import Odometry

        # ADR-0049 — on the sim-attached path read the post-step snapshot (plain
        # data) so this control-group callback never touches the simulator off
        # the sim thread; the legacy path reads the HAL directly.
        if self._proprio is not None:
            frame = self._proprio.latest()
            if frame is None:
                return
            base_pose = frame.base_pose
            pose_6dof = frame.base_pose_6dof
            bt: tuple[float, ...] = frame.base_twist
        else:
            getter = getattr(self._hal, "base_pose_6dof", None)
            pose_6dof = getter() if getter is not None else None
            base_pose = self._hal.base_pose
            bt = getattr(self._hal, "base_twist", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Prefer a 6-DoF proprio (robocasa base pos+quat) so base_z carries the
        # platform height and any non-yaw rotation survives; fall back to the
        # planar (x, y, yaw) projection for the in-process digital twin / non-MJCF
        # backends. Without the 6-DoF path the ADR-0027 state assembler sees
        # base_z = 0 and the policy reaches at the wrong height.
        x, y, yaw = base_pose
        if pose_6dof is not None:
            (px, py, pz), (qx, qy, qz, qw) = pose_6dof
        else:
            px, py, pz = float(x), float(y), 0.0
            qx, qy, qz, qw = quaternion_from_yaw(yaw)

        now = self._node.get_clock().now().to_msg()
        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame
        msg.pose.pose.position.x = px
        msg.pose.pose.position.y = py
        msg.pose.pose.position.z = pz
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Twist in the child (base_link) frame per REP-105 (``bt`` resolved above
        # from the snapshot or the HAL). The HAL latches the last commanded base
        # body twist and advances by exact Euler integration of it, so it is the
        # base velocity Nav2's controller reads back; absent → zero (no regression).
        msg.twist.twist.linear.x = float(bt[0])
        msg.twist.twist.linear.y = float(bt[1])
        msg.twist.twist.linear.z = float(bt[2])
        msg.twist.twist.angular.x = float(bt[3])
        msg.twist.twist.angular.y = float(bt[4])
        msg.twist.twist.angular.z = float(bt[5])
        self._odom_pub.publish(msg)

        self._broadcast_odom_tf(now, (px, py, pz), (qx, qy, qz, qw))

    def _broadcast_odom_tf(
        self,
        stamp: object,
        translation: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ) -> None:
        """Broadcast the ``odom->base_link`` transform for the given pose."""
        if self._tf_broadcaster is None:
            return
        from geometry_msgs.msg import TransformStamped

        px, py, pz = translation
        qx, qy, qz, qw = rotation
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = px
        t.transform.translation.y = py
        t.transform.translation.z = pz
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(t)

    def _on_cmd_vel(self, msg: object) -> None:
        """Bridge ``geometry_msgs/Twist`` on ``/cmd_vel`` to a BODY_TWIST Action.

        Maps the message into the canonical 6-vec ``[vx, vy, vz, wx, wy, wz]`` —
        only ``linear.x`` / ``linear.y`` / ``angular.z`` carry signal on a planar
        holonomic base. Per ADR-0024 this bypasses the OpenRAL safety supervisor
        (Nav2's ``velocity_smoother`` enforces caps); run an external
        ``twist_to_action`` relay onto ``/openral/candidate_action`` for the
        supervised path.
        """
        from openral_core.schemas import Action, ControlMode

        if self._hal is None or getattr(self._node, "_estopped", False):
            return
        linear = getattr(msg, "linear", None)
        angular = getattr(msg, "angular", None)
        if linear is None or angular is None:
            return
        row = [
            float(getattr(linear, "x", 0.0) or 0.0),
            float(getattr(linear, "y", 0.0) or 0.0),
            0.0,
            0.0,
            0.0,
            float(getattr(angular, "z", 0.0) or 0.0),
        ]
        action = Action(
            control_mode=ControlMode.BODY_TWIST,
            horizon=1,
            body_twist=[tuple(row)],
            frame_id=self._base_frame,
        )
        self._node._send_action_traced(action, source="cmd_vel")
