"""A mobile base driven by its own platform stack, seen through odometry and a twist topic.

Some mobile bases ship their own drive stack. OpenAMRobot's platform runs a
micro-ROS motor board and a ``robot_localization`` EKF (``openamr-platform-sw``
``real_bringup.launch.py``). OpenRAL does not replace that stack; it talks to it:

* the base pose and velocity come from the platform's ``nav_msgs/Odometry``;
* ``BODY_TWIST`` is published as ``geometry_msgs/Twist`` on the platform's command topic.

In an OpenRAL deploy, ``MobileBaseBridge`` owns ``/cmd_vel`` (Nav2's output) and
``odom -> base_frame``. The platform's stack must therefore be remapped onto
``odom_topic`` / ``cmd_vel_topic`` with its own odom TF broadcast off, so neither
topic nor TF has two owners. The defaults (``/platform/odom``, ``/platform/cmd_vel``)
make that remapping explicit.

The part's joints are the robot's three planar base joints; their positions are the
odometry pose (x, y, yaw).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import structlog
from openral_core import BODY_TWIST_DIM, RobotDescription
from openral_core.exceptions import (
    ROSConfigError,
    ROSEStopRequested,
    ROSPerceptionStale,
    ROSRuntimeError,
)
from openral_core.schemas import Action, ControlMode, JointState

__all__ = ["PlatformBaseHAL"]

log = structlog.get_logger(__name__)

_UNACTUATED_TWIST_EPS = 1e-6
_PLANAR_BASE_DOF = 3


class PlatformBaseHAL:
    """Planar base part over a platform's odometry and twist topics (see module doc).

    Args:
        description: Description whose joints are the three ``base_joints``.
        odom_topic: The platform's ``nav_msgs/Odometry``.
        cmd_vel_topic: The platform's ``geometry_msgs/Twist`` input.
        staleness_limit_s: Odometry older than this fails ``read_state``.

    Raises:
        ROSConfigError: The description does not declare three base joints.
    """

    def __init__(
        self,
        *,
        description: RobotDescription,
        odom_topic: str = "/platform/odom",
        cmd_vel_topic: str = "/platform/cmd_vel",
        staleness_limit_s: float = 0.5,
    ) -> None:
        """Store the wiring; ROS resources are created at ``attach_ros_node``."""
        if not description.base_joints or len(description.base_joints) != _PLANAR_BASE_DOF:
            raise ROSConfigError(
                f"PlatformBaseHAL needs base_joints [x, y, yaw]; got {description.base_joints}"
            )
        self.description = description
        self._odom_topic = odom_topic
        self._cmd_vel_topic = cmd_vel_topic
        self._staleness_limit_s = staleness_limit_s
        self._differential = description.base_kinematics == "differential"
        self._node: Any = None
        self._pub: Any = None
        self._sub: Any = None
        self._lock = threading.Lock()
        self._pose: tuple[float, float, float] | None = None
        self._twist: tuple[float, float, float, float, float, float] = (0.0,) * 6
        self._arrival = 0.0
        self._connected = False

    def attach_ros_node(self, node: Any) -> None:  # noqa: ANN401  # reason: rclpy node
        """Create the odometry subscription and twist publisher on ``node``."""
        from geometry_msgs.msg import Twist  # noqa: PLC0415  # reason: ROS-only import
        from nav_msgs.msg import Odometry  # noqa: PLC0415
        from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: PLC0415

        self._node = node
        self._twist_type = Twist
        self._sub = node.create_subscription(
            Odometry,
            self._odom_topic,
            self._on_odom,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._pub = node.create_publisher(Twist, self._cmd_vel_topic, 1)

    def _on_odom(self, msg: Any) -> None:  # noqa: ANN401  # reason: nav_msgs/Odometry
        p, q, t = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._pose = (float(p.x), float(p.y), yaw)
            self._twist = (t.linear.x, t.linear.y, 0.0, 0.0, 0.0, t.angular.z)
            self._arrival = time.monotonic()

    def connect(self) -> None:
        """Require the node; odometry freshness is checked at ``read_state``."""
        if self._node is None:
            raise ROSRuntimeError(
                "PlatformBaseHAL needs attach_ros_node before connect "
                "(the HAL lifecycle node provides it)."
            )
        self._connected = True

    def disconnect(self) -> None:
        """Stop the base and release its ROS endpoints; idempotent.

        The endpoints live on the lifecycle node, which outlives this HAL across a
        cleanup/configure cycle, so a kept subscription would feed a dead instance
        beside the new one's.
        """
        try:
            if self._connected and self._pub is not None:
                self._pub.publish(self._twist_type())
        finally:  # a failed stop publish still releases this HAL's endpoints
            self._connected = False
            if self._node is not None:
                if self._sub is not None:
                    self._node.destroy_subscription(self._sub)
                if self._pub is not None:
                    self._node.destroy_publisher(self._pub)
            self._sub = self._pub = None

    @property
    def base_pose(self) -> tuple[float, float, float]:
        """The latest odometry ``(x, y, yaw)``."""
        return self._fresh()[0]

    @property
    def base_twist(self) -> tuple[float, float, float, float, float, float]:
        """The latest odometry body twist."""
        with self._lock:
            return self._twist

    def _fresh(self) -> tuple[tuple[float, float, float], float]:
        with self._lock:
            pose, arrival = self._pose, self._arrival
        age = time.monotonic() - arrival
        if pose is None or age > self._staleness_limit_s:
            raise ROSPerceptionStale(
                f"PlatformBaseHAL: no odometry on {self._odom_topic} within "
                f"{self._staleness_limit_s} s (age {age:.3f} s)."
            )
        return pose, arrival

    def read_state(self) -> JointState:
        """The base joints at the odometry pose."""
        if not self._connected:
            raise ROSRuntimeError("PlatformBaseHAL.read_state() while not connected.")
        (x, y, yaw), arrival = self._fresh()
        vx, vy, _, _, _, wz = self.base_twist
        # Odometry twist is in the body frame; base_x/base_y are odom-frame coordinates,
        # so their rates are the body velocity rotated by yaw.
        c, s = math.cos(yaw), math.sin(yaw)
        age_s = max(time.monotonic() - arrival, 0.0)
        return JointState(
            name=list(self.description.base_joints or []),
            position=[x, y, yaw],
            velocity=[c * vx - s * vy, s * vx + c * vy, wz],
            effort=[],  # odometry measures no effort; never zero-filled (JointState)
            # Wall clock, as every HAL stamps JointState; arrival stays monotonic for
            # the staleness check.
            stamp_ns=time.time_ns() - int(age_s * 1e9),
        )

    def send_action(self, action: Action) -> None:
        """Publish a ``BODY_TWIST``; refuse other modes and, on a diff base, any ``vy``."""
        if not self._connected or self._pub is None:
            raise ROSRuntimeError("PlatformBaseHAL.send_action() while not connected.")
        vx, vy, _vz, _wx, _wy, wz = self.check_action(action)
        msg = self._twist_type()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self._pub.publish(msg)

    def check_action(self, action: Action) -> tuple[float, float, float, float, float, float]:
        """Refuse anything but an actuable ``BODY_TWIST``; return its twist.

        ``ComposedHAL`` asks this before commanding any part, so a joint target for
        the base (a whole-robot row) is refused before the arm has moved.
        """
        if action.control_mode != ControlMode.BODY_TWIST.value:
            raise ROSConfigError(
                f"PlatformBaseHAL drives the base by BODY_TWIST only; got {action.control_mode}."
            )
        if not action.body_twist or len(action.body_twist[-1]) != BODY_TWIST_DIM:
            raise ROSConfigError("PlatformBaseHAL: BODY_TWIST needs a 6-float row.")
        vx, vy, vz, wx, wy, wz = (float(v) for v in action.body_twist[-1])
        unactuated = (vy, vz, wx, wy) if self._differential else (vz, wx, wy)
        if any(abs(v) > _UNACTUATED_TWIST_EPS for v in unactuated):
            raise ROSConfigError(
                f"PlatformBaseHAL ({self.description.base_kinematics} base) cannot actuate "
                f"{(vx, vy, vz, wx, wy, wz)}."
            )
        return (vx, vy, vz, wx, wy, wz)

    def estop(self) -> None:
        """Command zero twist, then raise.

        The platform's own E-stop chain is the authority over its motors. This
        only stops OpenRAL from commanding motion.
        """
        if self._pub is not None:
            self._pub.publish(self._twist_type())
        raise ROSEStopRequested(f"Emergency stop on the base of '{self.description.name}'.")
