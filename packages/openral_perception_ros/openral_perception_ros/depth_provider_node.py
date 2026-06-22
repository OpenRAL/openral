#!/usr/bin/env python3
"""ADR-0064 — monocular metric-depth provider node (RGB -> depth for nvblox).

Subscribes a mono RGB camera stream, forwards each frame to the DA3 depth
sidecar (`tools/_da3_depth_server.py`, default `depth-anything/DA3-SMALL` —
measured 0.27 GB / ~27 Hz on an 8 GB Ada), and republishes the returned metric
depth as a `32FC1` `sensor_msgs/Image` (+ `CameraInfo`) that **nvblox** fuses
with cuVSLAM's pose into the Nav2 cost map. This is what gives a lidar-less,
camera-only robot a navigable `/map`.

The model runs out-of-process (the DA3 package is not transformers-native and
wants its own venv) — this node is the thin ZMQ client + ROS bridge, reusing the
tested `depth_convert` encoders. The depth/camera_info topics it publishes are
the ones `nvblox.launch.py` remaps onto nvblox's `depth/image` + `depth/camera_info`.

Live bring-up is operator-run (needs the sidecar venv + a GPU); the pure
conversion half is covered by `tests/unit/test_depth_convert.py`.
"""

from __future__ import annotations

import io
from typing import Any


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the depth-provider node, shut down cleanly."""
    import numpy as np
    import rclpy
    from PIL import Image as PILImage
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    from openral_perception_ros.depth_convert import (
        camera_info_from_intrinsics,
        depth_array_to_image_msg,
    )
    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes

    class DepthProviderNode(Node):  # type: ignore[misc]
        """Subscribe RGB, call the DA3 depth sidecar, publish 32FC1 depth + CameraInfo."""

        def __init__(self) -> None:
            super().__init__("openral_depth_provider")
            self.declare_parameter("image_topic", "/openral/cameras/front/image")
            self.declare_parameter("depth_topic", "/openral/depth/image")
            self.declare_parameter("camera_info_topic", "/openral/depth/camera_info")
            self.declare_parameter("depth_frame_id", "camera_depth_optical_frame")
            self.declare_parameter("sidecar_host", "127.0.0.1")
            self.declare_parameter("sidecar_port", 5771)
            self.declare_parameter("process_res", 504)
            self.declare_parameter("request_timeout_ms", 2000)

            gp = self.get_parameter
            self._depth_frame = gp("depth_frame_id").get_parameter_value().string_value
            self._process_res = gp("process_res").get_parameter_value().integer_value

            import zmq

            self._zmq = zmq.Context.instance()
            self._sock = self._zmq.socket(zmq.REQ)
            timeout_ms = gp("request_timeout_ms").get_parameter_value().integer_value
            self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self._sock.setsockopt(zmq.LINGER, 0)
            host = gp("sidecar_host").get_parameter_value().string_value
            port = gp("sidecar_port").get_parameter_value().integer_value
            self._sock.connect(f"tcp://{host}:{port}")

            sensor_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            info_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._depth_pub = self.create_publisher(
                Image, gp("depth_topic").get_parameter_value().string_value, sensor_qos
            )
            self._info_pub = self.create_publisher(
                CameraInfo, gp("camera_info_topic").get_parameter_value().string_value, info_qos
            )
            self._sub = self.create_subscription(
                Image,
                gp("image_topic").get_parameter_value().string_value,
                self._on_image,
                sensor_qos,
            )
            self.get_logger().info(
                f"depth_provider: {gp('image_topic').get_parameter_value().string_value} "
                f"-> {gp('depth_topic').get_parameter_value().string_value} "
                f"(sidecar {host}:{port}, frame {self._depth_frame!r})"
            )

        def _on_image(self, msg: Any) -> None:
            import msgpack

            try:
                bgr, w, h = image_to_bgr_bytes(msg)
            except ImageConvertError as exc:
                self.get_logger().warning(f"skip frame: {exc}")
                return
            rgb = np.frombuffer(bgr, dtype=np.uint8).reshape(h, w, 3)[..., ::-1]
            buf = io.BytesIO()
            PILImage.fromarray(np.ascontiguousarray(rgb), "RGB").save(buf, format="PNG")

            try:
                self._sock.send(
                    msgpack.packb(
                        {"op": "depth", "image": buf.getvalue(), "process_res": self._process_res},
                        use_bin_type=True,
                    )
                )
                rep = msgpack.unpackb(self._sock.recv(), raw=False)
            except Exception as exc:  # a sidecar hiccup must not kill the node
                self.get_logger().warning(f"sidecar request failed: {exc}")
                return
            if not rep.get("ok"):
                self.get_logger().warning(f"sidecar error: {rep.get('error')}")
                return

            dh, dw = int(rep["h"]), int(rep["w"])
            depth = np.frombuffer(rep["depth"], dtype=np.float32).reshape(dh, dw)
            stamp = msg.header.stamp
            depth_msg = depth_array_to_image_msg(depth, frame_id=self._depth_frame, stamp=stamp)
            info_msg = camera_info_from_intrinsics(
                fx=rep["fx"],
                fy=rep["fy"],
                cx=rep["cx"],
                cy=rep["cy"],
                width=dw,
                height=dh,
                frame_id=self._depth_frame,
                stamp=stamp,
            )
            self._depth_pub.publish(depth_msg)
            self._info_pub.publish(info_msg)

    rclpy.init(args=args)
    node = DepthProviderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
