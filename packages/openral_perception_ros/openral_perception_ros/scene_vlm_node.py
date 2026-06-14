#!/usr/bin/env python3
"""Scene-VLM query service node (ADR-0047).

Subscribes one or more camera ``sensor_msgs/Image`` streams, caches each
camera's latest frame, and serves ``/openral/perception/query_scene``
(``openral_msgs/srv/QueryScene``): a read-only, on-demand "answer this question
about camera Y's current view" backed by a ``kind: "vlm"`` rSkill (Qwen3.5-4B
NF4) running in the out-of-process sidecar (:mod:`tools.qwen_vlm_sidecar`).

Driven by the reasoner's ``query_scene`` tool (ADR-0047): the reasoner asks
open-ended scene-state questions for its replanning ladder — task progress and
success/failure verification ("has the robot grasped the mug?", "is the bowl on
the shelf?", "did we drop the object?").

This is the scene-reasoning counterpart of the object-localization detector node
(:mod:`openral_perception_ros.ros_image_detector_node`, which serves
``locate_in_view``). They are separate nodes because a scene VLM is a reasoning
aid, not a detector: it returns text, publishes nothing continuously, and runs
on-demand only.

**Camera-agnostic.** The ``cameras`` param maps logical camera ids to image
topics; with none given it falls back to the single ``image_topic`` under id
``primary_camera``. Every camera's latest frame is cached so ``query_scene`` can
answer about any of them; the reasoner picks a viewpoint by camera id.

Parameters:
    cameras (str[]): logical cameras as ``"id=topic"`` entries. Empty = a single
        camera ``primary_camera`` on ``image_topic``.
    primary_camera (str): id of the default camera (used when a request leaves
        ``camera`` empty).
    image_topic (str): single-camera fallback topic.
    manifest_path (str): rSkill manifest path (``kind: "vlm"``). Required.
    sidecar_host (str): ZMQ host of the scene-VLM sidecar. Default 127.0.0.1.
    sidecar_port (int): ZMQ port of the scene-VLM sidecar. Default 5759.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the scene-VLM node, shut down cleanly."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes

    class SceneVlmNode(Node):  # type: ignore[misc]
        """Subscribe camera Image(s), cache frames, serve query_scene."""

        def __init__(self) -> None:
            super().__init__("openral_scene_vlm")
            self.declare_parameter("cameras", [""])
            self.declare_parameter("primary_camera", "default")
            self.declare_parameter("image_topic", "/openral/cameras/agentview_left/image")
            self.declare_parameter("manifest_path", "")
            self.declare_parameter("sidecar_host", "127.0.0.1")
            self.declare_parameter("sidecar_port", 5759)

            gp = self.get_parameter
            manifest_path = gp("manifest_path").get_parameter_value().string_value
            if not manifest_path:
                raise ValueError("scene_vlm_node requires a manifest_path (kind: 'vlm')")

            # Latest BGR frame per camera id, for the on-demand query_scene service.
            self._frames: dict[str, tuple[bytes, int, int]] = {}
            self._cameras = self._resolve_cameras()
            self._primary_id = next(iter(self._cameras))
            self._vlm = self._build_vlm(manifest_path)

            img_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._subs = [
                self.create_subscription(Image, topic, self._make_cache_cb(cid), img_qos)
                for cid, topic in self._cameras.items()
            ]

            # query_scene service (ADR-0047) — only if the IDL is built.
            self._srv = None
            try:
                from openral_msgs.srv import QueryScene

                self._srv = self.create_service(
                    QueryScene,
                    "/openral/perception/query_scene",
                    self._on_query_scene,
                )
            except ImportError:
                self.get_logger().warning(
                    "openral_msgs/srv/QueryScene not built; query_scene service disabled"
                )

            self.get_logger().info(
                f"scene_vlm: cameras={self._cameras} primary={self._primary_id!r} "
                f"manifest={manifest_path}, query_scene={'on' if self._srv else 'off'}"
            )

        def _resolve_cameras(self) -> dict[str, str]:
            """Resolve the camera-id -> topic map (camera-agnostic)."""
            gp = self.get_parameter
            entries = [s for s in gp("cameras").get_parameter_value().string_array_value if s]
            cameras: dict[str, str] = {}
            for entry in entries:
                cid, _, topic = entry.partition("=")
                if cid and topic:
                    cameras[cid] = topic
            if not cameras:
                primary = gp("primary_camera").get_parameter_value().string_value or "default"
                cameras[primary] = gp("image_topic").get_parameter_value().string_value
            return cameras

        def _build_vlm(self, manifest_path: str) -> Any:
            """Build the scene-VLM backend from the rSkill manifest."""
            from openral_core.schemas import RSkillManifest
            from openral_runner.backends.gstreamer.qwen_scene_vlm import build_scene_vlm

            gp = self.get_parameter
            manifest = RSkillManifest.from_yaml(manifest_path)
            vlm = build_scene_vlm(
                manifest,
                host=gp("sidecar_host").get_parameter_value().string_value,
                port=gp("sidecar_port").get_parameter_value().integer_value,
            )
            self.get_logger().info(f"scene_vlm backend model={manifest.name}")
            return vlm

        def _make_cache_cb(self, cid: str) -> Callable[[Any], None]:
            def _cb(msg: Any) -> None:
                try:
                    bgr, w, h = image_to_bgr_bytes(msg)
                except ImageConvertError as exc:
                    self.get_logger().debug(f"cache_frame({cid}): convert failed: {exc}")
                    return
                self._frames[cid] = (bgr, w, h)

            return _cb

        def _on_query_scene(self, request: Any, response: Any) -> Any:
            """Service (ADR-0047): answer a question about camera Y's current frame."""
            question = request.question.strip()
            camera = request.camera.strip() or self._primary_id
            response.camera = camera
            frame = self._frames.get(camera)
            if frame is None:
                response.ok = False
                response.answer = ""
                self.get_logger().warning(
                    f"query_scene: no frame for camera {camera!r} (known: {sorted(self._frames)})"
                )
                return response
            bgr, w, h = frame
            try:
                answer = self._vlm.query(bgr, w, h, question)
            except Exception as exc:  # best-effort; never crash the service
                self.get_logger().warning(f"query_scene failed: {exc}")
                response.ok = False
                response.answer = ""
                return response
            response.ok = True
            response.answer = answer
            self.get_logger().info(
                f"query_scene: question={question!r} camera={camera!r} answer=({len(answer)} chars)"
            )
            return response

    rclpy.init(args=args)
    node = SceneVlmNode()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            node.destroy_node()
    finally:
        # Idempotent — no-op when the SIGINT handler (or whoever fired
        # ExternalShutdownException) already shut down the context.
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
