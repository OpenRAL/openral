#!/usr/bin/env python3
"""Stand-in data source for smoke-testing openral_foxglove_bringup headless.

NOT part of the product and NOT the OpenRAL sim — a dev convenience that emits
REAL, continuously-moving ROS messages on two Bucket-1 (whitelisted) topics so
the Foxglove panels animate without bringing up the full stack:

  * /tf            (tf2_msgs/TFMessage)     map -> odom -> base_link, base orbiting
  * /joint_states  (sensor_msgs/JointState) 6 joints, sine motion

The render path it exercises (whitelisted topic + native schema) is identical
to what the real producers (HAL sim sensor bridge, slam_toolbox, octomap) emit.

Run under ROS's interpreter (NOT a conda/miniforge python):
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 packages/openral_foxglove_bringup/tools/demo_publisher.py
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

JOINTS = [f"joint_{i}" for i in range(6)]


class DemoPublisher(Node):
    def __init__(self) -> None:
        super().__init__("openral_foxglove_demo")
        self.br = TransformBroadcaster(self)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.t0 = self.get_clock().now()
        self.create_timer(0.05, self.tick)  # 20 Hz

    def tick(self) -> None:
        now = self.get_clock().now()
        s = (now - self.t0).nanoseconds * 1e-9
        stamp = now.to_msg()

        a = TransformStamped()
        a.header.stamp = stamp
        a.header.frame_id = "map"
        a.child_frame_id = "odom"
        a.transform.rotation.w = 1.0

        b = TransformStamped()
        b.header.stamp = stamp
        b.header.frame_id = "odom"
        b.child_frame_id = "base_link"
        b.transform.translation.x = 2.0 * math.cos(s * 0.5)
        b.transform.translation.y = 2.0 * math.sin(s * 0.5)
        yaw = s * 0.5 + math.pi / 2
        b.transform.rotation.z = math.sin(yaw / 2)
        b.transform.rotation.w = math.cos(yaw / 2)
        self.br.sendTransform([a, b])

        js = JointState()
        js.header.stamp = stamp
        js.name = JOINTS
        js.position = [0.8 * math.sin(s * 0.7 + i) for i in range(len(JOINTS))]
        js.velocity = [0.56 * math.cos(s * 0.7 + i) for i in range(len(JOINTS))]
        self.js_pub.publish(js)


def main() -> None:
    rclpy.init()
    node = DemoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
