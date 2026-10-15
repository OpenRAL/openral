#!/usr/bin/env python3
"""Relay Nav2's ``/cmd_vel`` onto ``/openral/candidate_action`` so the safety kernel checks it.

Opt-in (``nav2.launch.py cmd_vel_relay:=true``; default off). The supervised alternative to the
direct path ``openral_hal.mobile_base_bridge.MobileBaseBridge`` gives a manifest that declares
``base_joints`` — that bypass is a recorded decision (ADR-0040, re-affirmed by ADR-0099) and this
node does not touch it.

The gap it closes: a robot that advertises ``body_twist`` but declares no ``base_joints``
(today ``g1`` and ``r1pro``) gets no ``MobileBaseBridge``, so nothing in-tree consumes the
``/cmd_vel`` Nav2 publishes for it — the base either never moves or is driven by an external
base driver subscribing ``/cmd_vel`` directly, outside the kernel.

Each ``geometry_msgs/Twist`` becomes a horizon-1 ``BODY_TWIST`` ``Action``
(``[linear.x, linear.y, 0, 0, 0, angular.z]`` in the manifest's ``base_frame`` — the same mapping
as ``MobileBaseBridge._on_cmd_vel``), published through ``openral_runner.ROSPublishingHAL`` like
every other candidate-action producer. The kernel bounds ``‖v‖`` / ``‖ω‖`` against the manifest's
``max_base_linear_speed_m_s`` / ``max_base_angular_speed_rad_s``, vetoes or E-stops, and forwards
on ``/openral/safe_action`` to the HAL. Nav2's costmap is still the only base collision
avoidance — the kernel adds a speed bound and its E-stop latch, not path checking.

Staleness: if no ``/cmd_vel`` arrives for ``cmd_vel_timeout_s`` (wall clock — a transport
watchdog, not sim time) after a non-zero command, the relay publishes ONE zero twist and logs it,
so a dead Nav2 cannot leave a latching base HAL driving on its last command.

Refuses to start (``ROSConfigError``, process exits) when the manifest declares ``base_joints``
(``MobileBaseBridge`` already consumes ``/cmd_vel``; both would double-actuate), omits
``body_twist`` (the kernel would reject every chunk), or leaves either ``max_base_*`` bound
unset (the kernel skips an unset bound, so the relay would add no speed limit).

Nav2 Jazzy publishes ``geometry_msgs/Twist`` (``enable_stamped_cmd_vel`` defaults false and the
shipped configs leave it unset); Kilted flips that default to ``TwistStamped``.

Adapted from DsslRobot/openral 2c08c1185.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["main", "twist_to_body_twist_row"]


def twist_to_body_twist_row(msg: Any) -> tuple[float, float, float, float, float, float]:
    """Map a ``geometry_msgs/Twist`` to the planar 6-vec ``[vx, vy, 0, 0, 0, wz]``.

    Example:
        >>> from types import SimpleNamespace as NS
        >>> msg = NS(linear=NS(x=0.2, y=-0.1, z=9.0), angular=NS(x=1.0, y=1.0, z=0.3))
        >>> twist_to_body_twist_row(msg)
        (0.2, -0.1, 0.0, 0.0, 0.0, 0.3)
    """
    return (float(msg.linear.x), float(msg.linear.y), 0.0, 0.0, 0.0, float(msg.angular.z))


def main(args: Any = None) -> None:
    """Entry point for ``ros2 run openral_nav2_bringup twist_to_action_relay_node.py``."""
    import rclpy
    import structlog
    from geometry_msgs.msg import Twist
    from openral_core.exceptions import ROSConfigError
    from openral_core.schemas import Action, ControlMode, RobotDescription
    from openral_runner.ros_publishing_hal import ROSPublishingHAL
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

    log = structlog.get_logger("openral_nav2_bringup.twist_to_action_relay")

    class TwistToActionRelayNode(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
        """``/cmd_vel`` in, kernel-checked ``BODY_TWIST`` chunks out."""

        def __init__(self) -> None:
            super().__init__("openral_nav2_twist_to_action_relay")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("cmd_vel_topic", "/cmd_vel")
            self.declare_parameter("candidate_action_topic", "/openral/candidate_action")
            self.declare_parameter("cmd_vel_timeout_s", 0.5)

            gp = self.get_parameter
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            cmd_vel_topic = gp("cmd_vel_topic").get_parameter_value().string_value
            candidate_topic = gp("candidate_action_topic").get_parameter_value().string_value
            self._timeout_s = gp("cmd_vel_timeout_s").get_parameter_value().double_value
            if self._timeout_s <= 0.0:
                raise ROSConfigError("twist_to_action_relay: cmd_vel_timeout_s must be positive")
            if not robot_yaml:
                raise ROSConfigError("twist_to_action_relay: `robot_yaml` parameter is required")
            description = RobotDescription.from_yaml(robot_yaml)
            if description.base_joints:
                raise ROSConfigError(
                    f"twist_to_action_relay: {description.name!r} declares base_joints, so "
                    "MobileBaseBridge already consumes /cmd_vel (ADR-0040/0099); relaying too "
                    "would double-actuate every command"
                )
            if ControlMode.BODY_TWIST not in description.capabilities.supported_control_modes:
                raise ROSConfigError(
                    f"twist_to_action_relay: {description.name!r} does not support body_twist"
                )
            safety = description.safety
            if (
                safety.max_base_linear_speed_m_s is None
                or safety.max_base_angular_speed_rad_s is None
            ):
                # The kernel skips an unset bound, so the relay would add no speed limit at all
                # (hazard-log Entry 060 Amendment 1, ruling 1).
                raise ROSConfigError(
                    f"twist_to_action_relay: {description.name!r} does not declare both "
                    "safety.max_base_linear_speed_m_s and safety.max_base_angular_speed_rad_s; "
                    "the kernel would not bound the relayed twist"
                )

            self._base_frame = description.base_frame
            self._hal = ROSPublishingHAL(
                node=self, description=description, candidate_action_topic=candidate_topic
            )
            self._hal.connect()
            self._last_rx_s: float | None = None
            self._moving = False
            # Control QoS (CLAUDE.md §2): RELIABLE, VOLATILE, KEEP_LAST=1.
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, qos)
            # Check at timeout/5: a stale command is zeroed within 1.2x the timeout.
            self.create_timer(self._timeout_s / 5.0, self._check_stale)
            log.info(
                "twist_to_action_relay.started",
                robot=description.name,
                cmd_vel_topic=cmd_vel_topic,
                candidate_action_topic=candidate_topic,
                base_frame=self._base_frame,
                cmd_vel_timeout_s=self._timeout_s,
            )

        def _publish(self, row: tuple[float, float, float, float, float, float]) -> None:
            # tick_index stays 0 (no rSkill goal), so send_action never blocks on
            # /openral/action_applied.
            self._hal.send_action(
                Action(
                    control_mode=ControlMode.BODY_TWIST,
                    horizon=1,
                    body_twist=[row],
                    frame_id=self._base_frame,
                )
            )

        def _on_cmd_vel(self, msg: Any) -> None:
            row = twist_to_body_twist_row(msg)
            self._last_rx_s = time.monotonic()
            self._moving = any(row)
            self._publish(row)

        def _check_stale(self) -> None:
            if not self._moving or self._last_rx_s is None:
                return
            age_s = time.monotonic() - self._last_rx_s
            if age_s > self._timeout_s:
                log.warning(
                    "twist_to_action_relay.cmd_vel_stale_zeroed",
                    age_s=round(age_s, 3),
                    cmd_vel_timeout_s=self._timeout_s,
                )
                self._moving = False
                self._publish((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    rclpy.init(args=args)
    try:
        node = TwistToActionRelayNode()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
