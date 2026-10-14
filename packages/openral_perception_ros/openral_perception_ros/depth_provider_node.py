#!/usr/bin/env python3
"""Monocular metric-depth provider node (RGB -> depth for nvblox).

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

**Metric mode** (``DeployRuntime.estimated_depth_camera``, world collision): with
``rgb_camera_info_topic`` set, the sidecar must serve ``DA3METRIC-*`` (checked on every
reply). Its canonical output becomes metres through the camera's own ``CameraInfo``
focal scaled to DA3's raster (``metric_depth``) — after the frame is rectified to an
exact pinhole with that ``CameraInfo``'s lens model (``lens_rectify``; or read through
``CameraInfo.p`` when ``rgb_image_rectified`` declares the stream already rectified) —
the depth + that scaled ``CameraInfo``
are published in the ``CameraInfo``'s optical frame, and, with ``points_topic`` set, the
depth back-projected (clipped at ``max_range_m``) as a ``PointCloud2`` for octomap. A
frame with no ``CameraInfo`` yet, or from a non-metric model, is dropped and logged —
never converted with a guessed focal.

Live bring-up is operator-run (needs the sidecar venv + a GPU); the pure
conversion half is covered by `tests/unit/test_depth_convert.py` and the
``metric_depth`` doctests.
"""

from __future__ import annotations

import io
from typing import Any


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the depth-provider node, shut down cleanly."""
    import numpy as np
    import rclpy
    from openral_observability.rclpy_spin import spin_node_until_shutdown
    from PIL import Image as PILImage
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    from openral_perception_ros.depth_convert import (
        camera_info_from_intrinsics,
        depth_array_to_image_msg,
    )
    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes
    from openral_perception_ros.lens_rectify import (
        PinholeRectifier,
        build_rectifier,
        distortion_problem,
        rectified_stream_k,
    )
    from openral_perception_ros.metric_depth import (
        camera_info_problem,
        da3_metric_depth,
        is_metric_model,
        points_from_depth,
        scale_intrinsics,
    )

    class DepthProviderNode(Node):  # type: ignore[misc]  # reason: rclpy.node.Node is untyped
        """Subscribe RGB, call the DA3 depth sidecar, publish 32FC1 depth + CameraInfo."""

        def __init__(self) -> None:
            super().__init__("openral_depth_provider")
            # No topic defaults (ADR-0108): deploy_e2e passes the mono camera's
            # ``camera_topic(<name>, IMAGE / DEPTH_IMAGE / DEPTH_CAMERA_INFO)``.
            self.declare_parameter("image_topic", "")
            self.declare_parameter("depth_topic", "")
            self.declare_parameter("camera_info_topic", "")
            # "" = metric mode stamps in the RGB CameraInfo's frame_id; the SLAM mode's
            # historic default otherwise.
            self.declare_parameter("depth_frame_id", "")
            self.declare_parameter("sidecar_host", "127.0.0.1")
            self.declare_parameter("sidecar_port", 5771)
            self.declare_parameter("process_res", 504)
            self.declare_parameter("request_timeout_ms", 2000)
            # Metric mode (world collision): the RGB camera's own CameraInfo, and where
            # the cloud goes. Empty = the SLAM/nvblox mode above (DA3's estimated K).
            self.declare_parameter("rgb_camera_info_topic", "")
            self.declare_parameter("points_topic", "")
            self.declare_parameter("max_range_m", 3.0)
            # The image is already rectified (image_pipeline's image_rect shares the raw
            # stream's CameraInfo): take its pinhole from CameraInfo.p, never undistort.
            self.declare_parameter("rgb_image_rectified", False)

            gp = self.get_parameter
            missing = [
                p
                for p in ("image_topic", "depth_topic", "camera_info_topic")
                if not gp(p).get_parameter_value().string_value
            ]
            if missing:
                from openral_core import ROSConfigError

                raise ROSConfigError(f"depth_provider: {missing} empty; pass camera_topic(...)")
            self._depth_frame = gp("depth_frame_id").get_parameter_value().string_value
            self._max_range_m = gp("max_range_m").get_parameter_value().double_value
            self._rgb_info: Any = None
            self._image_rectified = gp("rgb_image_rectified").get_parameter_value().bool_value
            self._rectifier: PinholeRectifier | None = None
            self._rectifier_key: tuple[Any, ...] = ()
            self._process_res = gp("process_res").get_parameter_value().integer_value

            import zmq

            self._zmq = zmq.Context.instance()
            self._timeout_ms = gp("request_timeout_ms").get_parameter_value().integer_value
            host = gp("sidecar_host").get_parameter_value().string_value
            port = gp("sidecar_port").get_parameter_value().integer_value
            self._endpoint = f"tcp://{host}:{port}"
            self._sock = self._make_socket()

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
            rgb_info_topic = gp("rgb_camera_info_topic").get_parameter_value().string_value
            points_topic = gp("points_topic").get_parameter_value().string_value
            self._metric = bool(rgb_info_topic)
            self._points_pub: Any = None
            if self._metric:
                from sensor_msgs.msg import PointCloud2

                # BEST_EFFORT matches every CameraInfo publisher: in-tree ones are RELIABLE,
                # but a driver on sensor-data QoS would never match a RELIABLE subscriber.
                rgb_info_qos = QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                )
                self._info_sub = self.create_subscription(
                    CameraInfo, rgb_info_topic, self._on_rgb_info, rgb_info_qos
                )
                if points_topic:
                    self._points_pub = self.create_publisher(PointCloud2, points_topic, sensor_qos)
                self.get_logger().warning(
                    f"depth_provider: ESTIMATED metric depth (DA3) from {rgb_info_topic}"
                    + (f" -> {points_topic} (world-collision cloud)" if points_topic else "")
                    + "; a monocular estimate, not a measurement"
                )
            self.get_logger().info(
                f"depth_provider: {gp('image_topic').get_parameter_value().string_value} "
                f"-> {gp('depth_topic').get_parameter_value().string_value} "
                f"(sidecar {host}:{port}, frame {self._depth_frame!r})"
            )

        def _on_rgb_info(self, msg: Any) -> None:
            self._rgb_info = msg

        def _metric_input(self, msg: Any, rgb: Any) -> tuple[Any, Any] | None:
            """Metric mode: the pinhole image DA3 must see, and its K; ``None`` drops it.

            The camera's ``CameraInfo`` is checked against the image (an unset focal,
            another aspect ratio), then the frame is rectified with that ``CameraInfo``'s
            lens model to an exact pinhole: DA3's metric scale and the back-projection both
            assume one. The remap is rebuilt only when the calibration or size changes.
            """
            info = self._rgb_info
            if info is None or not (self._depth_frame or info.header.frame_id) or not info.width:
                self.get_logger().warning(
                    "depth_provider: no CameraInfo (with a frame and size) for the RGB "
                    "camera yet; frame dropped",
                    throttle_duration_sec=5.0,
                )
                return None
            if self._image_rectified:
                k = rectified_stream_k(info.p)
                model, coeffs = "none", ()
            else:
                k = (float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]))
                model, coeffs = str(info.distortion_model), tuple(float(c) for c in info.d)
            info_wh = (int(info.width), int(info.height))
            image_wh = (int(msg.width), int(msg.height))
            problem = camera_info_problem(k, info_wh, image_wh) or distortion_problem(model, coeffs)
            if problem is not None:
                self.get_logger().warning(
                    f"depth_provider: {problem}; frame dropped", throttle_duration_sec=5.0
                )
                return None
            key = (k, info_wh, image_wh, model, coeffs)
            if self._rectifier is None or key != self._rectifier_key:
                # A uniform resize scales K and leaves the (normalised) coefficients alone.
                try:
                    self._rectifier = build_rectifier(
                        scale_intrinsics(k, info_wh, image_wh), image_wh, model, coeffs
                    )
                except ValueError as exc:
                    self.get_logger().warning(
                        f"depth_provider: {exc}; frame dropped", throttle_duration_sec=5.0
                    )
                    return None
                self._rectifier_key = key
                self.get_logger().info(
                    f"depth_provider: lens {model or 'none'} at {image_wh[0]}x{image_wh[1]} -> "
                    f"pinhole K {tuple(round(v, 2) for v in self._rectifier.k)}"
                )
            return self._rectifier.apply(rgb), self._rectifier

        def _publish_metric(
            self, msg: Any, rep: dict[str, Any], canonical: Any, rectifier: Any
        ) -> None:
            """Metric mode: canonical DA3 output -> metres, CameraInfo and the cloud."""
            model = str(rep.get("model", ""))
            if not is_metric_model(model):
                self.get_logger().error(
                    f"depth_provider: sidecar serves {model or 'an unnamed model'!r}, not "
                    "DA3METRIC-*; its depth is not metric, frame dropped",
                    throttle_duration_sec=5.0,
                )
                return
            info = self._rgb_info
            # DA3 saw the rectified image: its pinhole K, scaled to DA3's raster.
            dh, dw = canonical.shape
            fx, fy, cx, cy = scale_intrinsics(rectifier.k, rectifier.size, (dw, dh))
            depth = da3_metric_depth(canonical, fx=fx, fy=fy)
            # The launch's resolved optical frame wins over the CameraInfo's own frame_id.
            frame = self._depth_frame or str(info.header.frame_id)
            stamp = msg.header.stamp
            self._depth_pub.publish(depth_array_to_image_msg(depth, frame_id=frame, stamp=stamp))
            self._info_pub.publish(
                camera_info_from_intrinsics(
                    fx=fx, fy=fy, cx=cx, cy=cy, width=dw, height=dh, frame_id=frame, stamp=stamp
                )
            )
            if self._points_pub is not None:
                from sensor_msgs_py.point_cloud2 import create_cloud_xyz32

                pts = points_from_depth(
                    depth, fx=fx, fy=fy, cx=cx, cy=cy, max_range_m=self._max_range_m
                )
                header = msg.header.__class__()
                header.stamp = stamp
                header.frame_id = frame
                self._points_pub.publish(create_cloud_xyz32(header, pts))

        def _make_socket(self) -> Any:
            """Fresh REQ socket connected to the sidecar endpoint."""
            import zmq

            sock = self._zmq.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self._endpoint)
            return sock

        def _on_image(self, msg: Any) -> None:
            import msgpack

            try:
                bgr, w, h = image_to_bgr_bytes(msg)
            except ImageConvertError as exc:
                self.get_logger().warning(
                    f"dropping {msg.encoding!r} frame: {exc}", throttle_duration_sec=5.0
                )
                return
            rgb = np.frombuffer(bgr, dtype=np.uint8).reshape(h, w, 3)[..., ::-1]
            rectifier = None
            if self._metric:
                prepared = self._metric_input(msg, np.ascontiguousarray(rgb))
                if prepared is None:
                    return
                rgb, rectifier = prepared
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
                # A REQ socket is a strict send/recv state machine: after a
                # timed-out recv it refuses every further send (EFSM), so one
                # slow reply — e.g. the autostarted sidecar still provisioning
                # its venv — would otherwise wedge depth forever. Rebuild it.
                self._sock.close(linger=0)
                self._sock = self._make_socket()
                self.get_logger().warning(f"sidecar request failed (socket reset): {exc}")
                return
            if not rep.get("ok"):
                self.get_logger().warning(f"sidecar error: {rep.get('error')}")
                return

            dh, dw = int(rep["h"]), int(rep["w"])
            depth = np.frombuffer(rep["depth"], dtype=np.float32).reshape(dh, dw)
            if self._metric:
                self._publish_metric(msg, rep, depth, rectifier)
                return
            if not all(np.isfinite(float(rep.get(c, np.nan))) for c in ("fx", "fy", "cx", "cy")):
                # A DA3METRIC sidecar predicts no camera (NaN K): never hand nvblox NaN.
                self.get_logger().error(
                    f"depth_provider: sidecar {rep.get('model', '?')!r} returned no finite "
                    "intrinsics; SLAM mode needs DA3's own K, frame dropped",
                    throttle_duration_sec=5.0,
                )
                return
            stamp = msg.header.stamp
            slam_frame = self._depth_frame or "camera_depth_optical_frame"
            depth_msg = depth_array_to_image_msg(depth, frame_id=slam_frame, stamp=stamp)
            info_msg = camera_info_from_intrinsics(
                fx=rep["fx"],
                fy=rep["fy"],
                cx=rep["cx"],
                cy=rep["cy"],
                width=dw,
                height=dh,
                frame_id=slam_frame,
                stamp=stamp,
            )
            self._depth_pub.publish(depth_msg)
            self._info_pub.publish(info_msg)

    rclpy.init(args=args)
    node = DepthProviderNode()
    try:
        spin_node_until_shutdown(node)  # quiet on the SIGINT shutdown race
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
