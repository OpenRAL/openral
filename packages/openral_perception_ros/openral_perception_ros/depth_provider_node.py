#!/usr/bin/env python3
"""Monocular depth provider node (RGB -> depth for nvblox or the world-collision map).

Subscribes a mono RGB camera stream, forwards each frame to the sidecar of a
``kind: depth_estimator`` rSkill (``manifest_path``; booted by
``tools/depth_estimator_sidecar.py``: Depth Anything 3 or Depth Pro 2 in their own venvs,
or an OpenRAL Pro TensorRT backend), and republishes the returned depth as a ``32FC1``
``sensor_msgs/Image`` (+ ``CameraInfo``). The manifest's ``DepthEstimatorContract`` says
how the raw output becomes metres; every reply must come from the manifest's model repo.

**SLAM mode** (``rgb_camera_info_topic`` empty, ``DeployRuntime.slam_mono_camera``): the
skill must predict its own camera (``predicts_intrinsics``, DA3-SMALL measured
0.27 GB / ~27 Hz on an 8 GB Ada); depth + that K feed **nvblox** and cuVSLAM, which is
what gives a lidar-less, camera-only robot a navigable ``/map``. DA3-SMALL's output is
``relative`` (not documented as metric): this mode publishes it unconverted, as
*approximately* metres, so the ``/map`` and cuVSLAM's scale carry that scale error. It
never reaches the safety kernel's world map (metric mode refuses a ``relative`` skill).

**Metric mode** (``DeployRuntime.estimated_depth_camera``, world collision): with
``rgb_camera_info_topic`` set, the skill's ``output`` must be ``metric`` or
``canonical_focal`` (a ``relative`` skill is refused at startup). The output becomes
metres through the camera's own ``CameraInfo`` focal (``metric_depth``): scaled to the
model's raster for ``canonical_focal`` (DA3METRIC), sent as ``f_px`` with the frame for
``metric`` (Depth Pro 2) — after the frame is rectified to an
exact pinhole with that ``CameraInfo``'s lens model (``lens_rectify``; or read through
``CameraInfo.p`` when ``rgb_image_rectified`` declares the stream already rectified) —
the depth + that scaled ``CameraInfo``
are published in the ``CameraInfo``'s optical frame, and, with ``points_topic`` set, the
depth back-projected (clipped at ``max_range_m``) as a ``PointCloud2`` for octomap. A
frame with no ``CameraInfo`` yet, or a reply from another model, is dropped and logged —
never converted with a guessed focal.

**Lifecycle.** A managed lifecycle node (``openral_depth_provider`` /
``openral_estimated_depth_provider`` in ``deploy_e2e``, autostarted ACTIVE): configure
reads the parameters, refuses an unusable rSkill
(``openral_rskill.depth_estimator.depth_skill``, license gate included) and opens the
sidecar socket and topics; only ACTIVE forwards frames and publishes; deactivate stops
both (the sidecar is its own process and keeps its weights, so no VRAM is freed);
cleanup / shutdown close the socket. The estimated-depth provider is a reasoner
lifecycle peer: while it is inactive octomap gets no cloud and the kernel's world-voxel
deadline stops motion.

Live bring-up is operator-run (needs the sidecar venv + a GPU); the pure
conversion half is covered by `tests/unit/test_depth_convert.py` and the
``metric_depth`` doctests.
"""

from __future__ import annotations

import io
from typing import Any


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the depth-provider lifecycle node, shut down cleanly.

    The node starts UNCONFIGURED; ``deploy_e2e.launch.py`` drives it to ACTIVE
    (``tools/lifecycle_autostart.py``).
    """
    import numpy as np
    import rclpy
    from openral_observability.rclpy_spin import spin_node_until_shutdown
    from PIL import Image as PILImage
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
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
        metres_from_output,
        points_from_depth,
        scale_intrinsics,
    )

    class DepthProviderNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy is untyped
        """Subscribe RGB, call the depth_estimator sidecar, publish 32FC1 depth + CameraInfo.

        A managed lifecycle node: ``configure`` reads the parameters, checks the manifest
        and opens the sidecar socket, publishers and subscriptions; only an ACTIVE node
        calls the sidecar and publishes; ``deactivate`` stops both (the sidecar keeps its
        weights: it is its own process); ``cleanup`` / ``shutdown`` close the socket.
        """

        def __init__(self, node_name: str = "openral_depth_provider") -> None:
            super().__init__(node_name)
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
            # The kind: depth_estimator rSkill the sidecar serves (its contract and model).
            self.declare_parameter("manifest_path", "")
            self.declare_parameter("request_timeout_ms", 2000)
            # Metric mode (world collision): the RGB camera's own CameraInfo, and where
            # the cloud goes. Empty = the SLAM/nvblox mode above (DA3's estimated K).
            self.declare_parameter("rgb_camera_info_topic", "")
            self.declare_parameter("points_topic", "")
            self.declare_parameter("max_range_m", 3.0)
            # The image is already rectified (image_pipeline's image_rect shares the raw
            # stream's CameraInfo): take its pinhole from CameraInfo.p, never undistort.
            self.declare_parameter("rgb_image_rectified", False)
            self._active = False
            self._sock: Any = None
            self._entities: list[tuple[str, Any]] = []
            self._points_pub: Any = None
            # rclpy (Jazzy) raises out of the executor on a change_state request whose
            # transition id the current state lacks (a second ``deactivate``: the reasoner
            # sends ids, not labels), killing the node. Answer it ``success: false``.
            change_state = self._service_change_state
            serve = change_state.callback

            def guarded(req: Any, resp: Any) -> Any:
                try:
                    return serve(req, resp)
                except RuntimeError as exc:  # rclpy's RCLError; its type is private
                    self.get_logger().warning(f"change_state refused: {exc}")
                    resp.success = False
                    return resp

            change_state.callback = guarded

        # ── lifecycle ────────────────────────────────────────────────────────

        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Read parameters, check the rSkill, open the socket, topics (idle until ACTIVE)."""
            del state
            from openral_core import ROSConfigError
            from openral_rskill.depth_estimator import depth_skill

            gp = self.get_parameter
            missing = [
                p
                for p in ("image_topic", "depth_topic", "camera_info_topic", "manifest_path")
                if not gp(p).get_parameter_value().string_value
            ]
            rgb_info_topic = gp("rgb_camera_info_topic").get_parameter_value().string_value
            self._metric = bool(rgb_info_topic)
            try:
                if missing:
                    raise ROSConfigError(f"depth_provider: {missing} empty")
                self._contract, self._model = depth_skill(
                    gp("manifest_path").get_parameter_value().string_value, metric=self._metric
                )
            except ROSConfigError as exc:
                self.get_logger().error(f"configure refused: {exc}")
                return TransitionCallbackReturn.FAILURE
            self._depth_frame = gp("depth_frame_id").get_parameter_value().string_value
            self._max_range_m = gp("max_range_m").get_parameter_value().double_value
            self._rgb_info: Any = None
            self._image_rectified = gp("rgb_image_rectified").get_parameter_value().bool_value
            self._rectifier: PinholeRectifier | None = None
            self._rectifier_key: tuple[Any, ...] = ()

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
            self._entities = [("pub", self._depth_pub), ("pub", self._info_pub)]
            sub = self.create_subscription(
                Image,
                gp("image_topic").get_parameter_value().string_value,
                self._on_frame,
                sensor_qos,
            )
            self._entities.append(("sub", sub))
            points_topic = gp("points_topic").get_parameter_value().string_value
            self._points_pub = None
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
                info_sub = self.create_subscription(
                    CameraInfo, rgb_info_topic, self._on_rgb_info, rgb_info_qos
                )
                self._entities.append(("sub", info_sub))
                if points_topic:
                    self._points_pub = self.create_publisher(PointCloud2, points_topic, sensor_qos)
                    self._entities.append(("pub", self._points_pub))
                self.get_logger().warning(
                    f"depth_provider: ESTIMATED metric depth ({self._model}) from "
                    f"{rgb_info_topic}"
                    + (f" -> {points_topic} (world-collision cloud)" if points_topic else "")
                    + "; a monocular estimate, not a measurement"
                )
            self.get_logger().info(
                f"depth_provider: {gp('image_topic').get_parameter_value().string_value} "
                f"-> {gp('depth_topic').get_parameter_value().string_value} "
                f"(sidecar {host}:{port} serving {self._model}, frame {self._depth_frame!r})"
            )
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Start calling the sidecar and publishing."""
            self._active = True
            self.get_logger().info("depth_provider: active (estimating + publishing)")
            return super().on_activate(state)

        def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Stop calling the sidecar and publishing (its weights stay in the sidecar)."""
            self._active = False
            self.get_logger().warning(
                "depth_provider: deactivated; no depth"
                + (" or cloud" if self._points_pub is not None else "")
                + " is published until it is activated again"
            )
            return super().on_deactivate(state)

        def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Close the socket; destroy the publishers and subscriptions."""
            del state
            self._active = False
            if self._sock is not None:
                self._sock.close(linger=0)
                self._sock = None
            for kind, entity in self._entities:
                if kind == "pub":
                    self.destroy_publisher(entity)
                else:
                    self.destroy_subscription(entity)
            self._entities = []
            self._points_pub = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Same teardown as cleanup, from any state."""
            return self.on_cleanup(state)

        def _on_rgb_info(self, msg: Any) -> None:
            self._rgb_info = msg

        def _metric_input(self, msg: Any, rgb: Any) -> tuple[Any, Any] | None:
            """Metric mode: the pinhole image the model must see, and its K; ``None`` drops it.

            The camera's ``CameraInfo`` is checked against the image (an unset focal,
            another aspect ratio), then the frame is rectified with that ``CameraInfo``'s
            lens model to an exact pinhole: the metric scale and the back-projection both
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
            coeffs: tuple[float, ...]
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
            """Metric mode: raw model output -> metres, CameraInfo and the cloud."""
            info = self._rgb_info
            # The model saw the rectified image: its pinhole K, scaled to the output raster.
            dh, dw = canonical.shape
            fx, fy, cx, cy = scale_intrinsics(rectifier.k, rectifier.size, (dw, dh))
            depth = metres_from_output(canonical, self._contract, fx=fx, fy=fy)
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

        def _on_frame(self, msg: Any) -> None:
            if self._active:  # configured or deactivated: no sidecar call, nothing out
                self._on_image(msg)

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
            req: dict[str, Any] = {"op": "depth", "image": buf.getvalue()}
            if self._contract.process_res is not None:
                req["process_res"] = self._contract.process_res
            if rectifier is not None and self._contract.output.value == "metric":
                # Depth Pro 2 scales by the focal of the image it is sent: the rectified fx.
                req["f_px"] = float(rectifier.k[0])

            try:
                self._sock.send(msgpack.packb(req, use_bin_type=True))
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
                dropped = f"sidecar error {rep.get('error')!r}"
            elif rep.get("model") != self._model:
                dropped = (
                    f"the sidecar serves {rep.get('model')!r}, not the rSkill's {self._model!r}"
                )
            else:
                dropped = ""
            if dropped:
                self.get_logger().warning(
                    f"depth_provider: frame dropped: {dropped}", throttle_duration_sec=5.0
                )
                return

            dh, dw = int(rep["h"]), int(rep["w"])
            depth = np.frombuffer(rep["depth"], dtype=np.float32).reshape(dh, dw)
            if self._metric:
                self._publish_metric(msg, rep, depth, rectifier)
                return
            if not all(np.isfinite(float(rep.get(c, np.nan))) for c in ("fx", "fy", "cx", "cy")):
                # A model that predicts no camera replies NaN K: never hand nvblox NaN.
                self.get_logger().error(
                    f"depth_provider: sidecar {rep.get('model', '?')!r} returned no finite "
                    "intrinsics; SLAM mode needs the model's own K, frame dropped",
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
