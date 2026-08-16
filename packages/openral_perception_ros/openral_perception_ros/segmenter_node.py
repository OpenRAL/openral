#!/usr/bin/env python3
"""Promptable-segmenter service node (``segmenter`` rSkill kind).

Subscribes one or more camera ``sensor_msgs/Image`` streams, caches each
camera's latest frame, and serves ``/openral/perception/segment_in_view``
(``openral_msgs/srv/SegmentInView``): a read-only, on-demand "which pixels of
camera Y's current view belong to the thing under *this* 3-D point?", backed by
a ``kind: "segmenter"`` rSkill (SAM 2.1 hiera-small) running **in process**
under the workspace's own ``transformers``
(:class:`~openral_runner.backends.gstreamer.sam2_segmenter.Sam2Segmenter`).

Driven by the HAL's vision attachment-evidence producer at attach / detach /
regrasp events — one shot per event, never per frame. It runs *here*, beside the
detector rSkills on the runner/GStreamer graph side, because the HAL is
deliberately kept torch-free: the HAL asks over the service instead of importing
a model.

This is the **geometric** counterpart of the object-localization detector node
(:mod:`openral_perception_ros.ros_image_detector_node`, which serves
``locate_in_view`` for a free-text query). Separate nodes because a segmenter
answers a different question with a different contract: no label vocabulary, no
score threshold, a point in and masks out.

**Where the intrinsics live.** The request carries a 3-D point, not a pixel,
precisely so camera intrinsics stay on this side of the boundary. This node
transforms that point from the request's ``frame_id`` into the camera's optical
frame through TF2 (the only source of coordinate frames — CLAUDE.md §2), then
projects it with the **manifest's** ``SensorSpec.intrinsics``, rescaled to the
frame actually received. Nothing here hardcodes a pixel or a frame id.

**Why the reply is plural.** With ``segmenter.multimask`` the model emits three
nested hypotheses per point prompt (roughly subpart / part / whole). Only
geometry can say which is the payload, and this node has no depth — so it
returns every candidate that cleared ``min_mask_area_px``, area ascending, and
the HAL picks. It must not, and does not, pick by the model's own score: a
mis-aimed prompt was measured returning a mask covering 59.8% of a real frame at
this model's **top** score of 0.977.

**Lifecycle.** A *managed* lifecycle node, mirroring
``ros_image_detector_node``: cameras, subscriptions and the service live for the
configured→cleanup span; the model is built **and warmed** on ``on_activate``
and released on ``on_deactivate``, so its VRAM can be freed like any other
perception model. The warm-up is not optional — the first forward pass was
measured at ~742 ms against ~53 ms warmed, and only the warmed figure fits
inside the ~100 ms deferred-ack barrier the HAL holds while it calls this.

Parameters:
    cameras (str[]): logical cameras as ``"id=topic"`` entries. Each id MUST be
        a ``SensorSpec`` name in ``robot_yaml`` — that is where its intrinsics
        and optical ``frame_id`` come from. Empty = a single camera
        ``primary_camera`` on ``image_topic``.
    primary_camera (str): id of the default camera (used when a request leaves
        ``camera`` empty).
    image_topic (str): single-camera fallback topic.
    robot_yaml (str): RobotDescription path, for each camera's intrinsics +
        optical frame. Required.
    manifest_path (str): rSkill manifest path (``kind: "segmenter"``). Required.
    segment_in_view_service (str): service name. Default
        ``/openral/perception/segment_in_view``.
    publish_debug_masks (bool): **Default false.** Publish each successful reply
        again on ``debug_masks_topic`` as an ``openral_msgs/SegmentMasks``, so
        the dashboard's camera tiles can draw what was masked. Strictly
        diagnostic (see below). Off costs nothing: no publisher is created and
        no message is packed.
    debug_masks_topic (str): topic for the above. Default
        ``/openral/perception/masks``.
    device (str): ``auto`` (CUDA when available), ``cpu``, ``cuda``, ``cuda:N``.
        ``cpu`` is the honest setting on a pre-sm_70 dev GPU, where the CUDA
        wheels have no kernels for the card.

**The debug mask topic is not a second contract.** The HAL's producer takes its
masks from *its own* service reply and gates them on geometry; this topic exists
only so an operator can see the same masks on the dashboard, which has rendered
``topics.perception.overlays[camera].masks`` since PR #122 and had no producer
to render. Publishing happens *after* the response is fully built, and a failure
to publish is logged and swallowed — a display path must never be able to
degrade a grasp. Sensor-class QoS (``BEST_EFFORT`` / ``VOLATILE`` /
``KEEP_LAST=1``): only the newest mask set is worth drawing, and a dropped one
changes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from openral_perception_ros.camera_topics import resolve_camera_topics

if TYPE_CHECKING:  # pragma: no cover — typing only, keeps numpy off the import path
    import numpy as np
    from numpy.typing import NDArray
    from openral_core import RobotDescription, SensorSpec

__all__ = [
    "main",
    "make_segmenter_node",
    "mono8_bytes_from_mask",
    "sensor_spec_by_name",
]

#: Service name the HAL's vision attachment bridge calls by default.
DEFAULT_SEGMENT_SERVICE = "/openral/perception/segment_in_view"

#: Diagnostic topic the latest mask set is re-published on when
#: ``publish_debug_masks`` is enabled. Mirrors the detector leg's
#: ``/openral/perception/objects``, and is the topic the dashboard's
#: ``PerceptionOverlaySubscriber`` reads to paint mask overlays.
DEFAULT_DEBUG_MASKS_TOPIC = "/openral/perception/masks"


def sensor_spec_by_name(description: RobotDescription, name: str) -> SensorSpec | None:
    """Find a ``SensorSpec`` by name across ``sensors`` + ``sensor_bundles``.

    Args:
        description: The robot manifest.
        name: Sensor name, which is also the logical camera id in ``cameras``.

    Returns:
        The spec, or ``None`` when the manifest declares no such sensor.
    """
    for spec in description.sensors:
        if spec.name == name:
            return spec
    for bundle in description.sensor_bundles:
        for spec in bundle.sensors:
            if spec.name == name:
                return spec
    return None


def mono8_bytes_from_mask(mask: NDArray[np.bool_]) -> bytes:
    """Pack a boolean mask into ``mono8`` image bytes: 255 set, 0 clear.

    Raw ``mono8``, deliberately not run-length encoded: this is a one-shot event
    call, so a 640x480 mask is one ~307 kB image and at most three of them —
    well inside the barrier the caller already holds. An RLE codec would save a
    few hundred bytes in exchange for a bespoke encoder/decoder pair that the
    perception node and the safety-adjacent HAL producer would have to agree on
    exactly, forever. ``sensor_msgs/Image`` is already normative and already
    introspectable with ``ros2 topic echo``.

    Args:
        mask: ``(H, W)`` boolean array.

    Returns:
        ``H * W`` tightly-packed bytes, row-major.

    Example:
        >>> import numpy as np
        >>> list(mono8_bytes_from_mask(np.array([[True, False]])))
        [255, 0]
    """
    import numpy as np

    return bytes(np.asarray(mask, dtype=bool).astype(np.uint8) * np.uint8(255))


def _node_class() -> type:
    """Define the lifecycle node class behind lazy ROS imports, and return it.

    Hoisted out of :func:`main` — unlike the sibling perception nodes, which
    define their class inside ``main()`` — so the live integration test can
    stand up the **real** node in-process rather than substitute a double
    (CLAUDE.md §1.11). The lazy-import property is unchanged: nothing ROS is
    imported until this is called, so the module still imports cleanly on a host
    without rclpy.
    """
    import numpy as np
    from openral_core import RobotDescription, scale_intrinsics_to
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes

    class SegmenterNode(LifecycleNode):  # type: ignore[misc]  # reason: rclpy is untyped at runtime
        """Cache camera frames; answer geometric point prompts with masks."""

        def __init__(self, node_name: str = "openral_segmenter") -> None:
            """Declare parameters; no ROS entities and no model yet."""
            super().__init__(node_name)
            self.declare_parameter("cameras", [""])
            self.declare_parameter("primary_camera", "default")
            self.declare_parameter("image_topic", "/openral/cameras/wrist/image")
            self.declare_parameter("robot_yaml", "")
            self.declare_parameter("manifest_path", "")
            self.declare_parameter("segment_in_view_service", DEFAULT_SEGMENT_SERVICE)
            # Diagnostic only, and OFF by default: enabling it puts a few
            # hundred kB of mask on the wire per attach event purely so a human
            # can look at it (see the module docstring).
            self.declare_parameter("publish_debug_masks", False)
            self.declare_parameter("debug_masks_topic", DEFAULT_DEBUG_MASKS_TOPIC)
            self.declare_parameter("device", "auto")

            self._frames: dict[str, tuple[bytes, int, int]] = {}
            self._cameras: dict[str, str] = {}
            self._primary_id = ""
            self._description: Any = None
            self._segmenter: Any = None
            self._model_name = ""
            self._subs: list[Any] = []
            self._srv: Any = None
            self._masks_pub: Any = None
            self._tf_buffer: Any = None
            self._tf_listener: Any = None

        # ── lifecycle ────────────────────────────────────────────────────────

        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Wire cameras, TF, subscriptions and the service (no model yet)."""
            del state
            gp = self.get_parameter
            robot_yaml = gp("robot_yaml").get_parameter_value().string_value
            if not robot_yaml:
                raise ValueError("segmenter_node requires a robot_yaml (camera intrinsics)")
            if not gp("manifest_path").get_parameter_value().string_value:
                raise ValueError("segmenter_node requires a manifest_path (kind: 'segmenter')")
            self._description = RobotDescription.from_yaml(robot_yaml)
            self._cameras = resolve_camera_topics(
                list(gp("cameras").get_parameter_value().string_array_value),
                primary_camera=gp("primary_camera").get_parameter_value().string_value,
                image_topic=gp("image_topic").get_parameter_value().string_value,
            )
            self._primary_id = next(iter(self._cameras))

            import tf2_ros

            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

            # Sensor data class per CLAUDE.md §2: BEST_EFFORT / VOLATILE, depth 1
            # — only the newest frame can answer "what is in the jaws *now*".
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

            # An instantaneous query => a service, not a topic (CLAUDE.md §2).
            # Left on rmw's services-default QoS (RELIABLE / VOLATILE): a lost
            # request must not silently become a stalled attach barrier.
            self._srv = None
            try:
                from openral_msgs.srv import SegmentInView

                self._srv = self.create_service(
                    SegmentInView,
                    gp("segment_in_view_service").get_parameter_value().string_value,
                    self._on_segment_in_view,
                )
            except ImportError:
                self.get_logger().warning(
                    "openral_msgs/srv/SegmentInView not built; segment_in_view service disabled"
                )
            self._open_debug_masks_publisher()
            self.get_logger().info(
                f"segmenter configured: cameras={self._cameras} primary={self._primary_id!r} "
                f"segment_in_view={'on' if self._srv else 'off'} "
                f"debug_masks={'on' if self._masks_pub else 'off'}"
            )
            return TransitionCallbackReturn.SUCCESS

        def _open_debug_masks_publisher(self) -> None:
            """Open the diagnostic mask publisher, but only when asked to.

            Default-off is the whole point: an operator who has not enabled it
            pays for no publisher, no serialization and no DDS traffic. The
            profile is sensor-class (``BEST_EFFORT`` / ``VOLATILE`` /
            ``KEEP_LAST=1``) and must MATCH the dashboard's subscriber — a
            RELIABLE publisher here would still match, but the reverse would
            not, and the overlay would sit blank with nothing to explain it.
            """
            gp = self.get_parameter
            if not gp("publish_debug_masks").get_parameter_value().bool_value:
                return
            try:
                from openral_msgs.msg import SegmentMasks
            except ImportError:
                self.get_logger().warning(
                    "openral_msgs/msg/SegmentMasks not built; debug mask topic disabled"
                )
                return
            self._masks_pub = self.create_publisher(
                SegmentMasks,
                gp("debug_masks_topic").get_parameter_value().string_value,
                QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    durability=QoSDurabilityPolicy.VOLATILE,
                ),
            )

        def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Build the segmenter backend and burn the cold forward pass."""
            if self._segmenter is None:
                self._segmenter = self._build_segmenter()
                # Not optional: the first call is ~742 ms vs ~53 ms warmed, and
                # only the warmed figure fits inside the HAL's deferred-ack
                # barrier. Paying it here moves the cost off the first grasp.
                self._segmenter.warm_up()
                self.get_logger().info("segmenter warmed (first forward pass burned at activate)")
            return super().on_activate(state)

        def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Release the model — frees its (GPU) memory."""
            self._release_segmenter()
            self.get_logger().info("segmenter deactivated (model released).")
            return super().on_deactivate(state)

        def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Tear down the model, subscriptions and the service."""
            del state
            self._release_segmenter()
            for sub in self._subs:
                self.destroy_subscription(sub)
            self._subs = []
            if self._srv is not None:
                self.destroy_service(self._srv)
                self._srv = None
            if self._masks_pub is not None:
                self.destroy_publisher(self._masks_pub)
                self._masks_pub = None
            self._tf_listener = None
            self._tf_buffer = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Force cleanup."""
            return self.on_cleanup(state)

        def _release_segmenter(self) -> None:
            """Release the backend, idempotently; teardown must never raise."""
            segmenter = self._segmenter
            self._segmenter = None
            if segmenter is None:
                return
            try:
                segmenter.close()
            except Exception as exc:  # reason: teardown must not raise from a lifecycle cb
                self.get_logger().warning(f"segmenter close failed: {exc}")

        def _build_segmenter(self) -> Any:
            """Build the backend from the ``kind: segmenter`` manifest."""
            from openral_core.schemas import RSkillManifest
            from openral_rskill._diagnostics import phase_timer
            from openral_runner.backends.gstreamer.segmenter_factory import (
                build_manifest_segmenter,
            )

            gp = self.get_parameter
            manifest = RSkillManifest.from_yaml(
                gp("manifest_path").get_parameter_value().string_value
            )
            device = gp("device").get_parameter_value().string_value or "auto"
            with phase_timer("build", prefix="segmenter", gpu_mb=True, model=str(manifest.name)):
                segmenter = build_manifest_segmenter(manifest, device=device)
            # Named on every diagnostic mask set, so "whose masks are these" is
            # answerable from the topic alone.
            self._model_name = str(manifest.name)
            self.get_logger().info(f"segmenter model={manifest.name} device={device}")
            return segmenter

        # ── frame cache ──────────────────────────────────────────────────────

        def _make_cache_cb(self, cid: str) -> Callable[[Any], None]:
            def _cb(msg: Any) -> None:
                try:
                    bgr, w, h = image_to_bgr_bytes(msg)
                except ImageConvertError as exc:
                    self.get_logger().debug(f"cache_frame({cid}): convert failed: {exc}")
                    return
                self._frames[cid] = (bgr, w, h)

            return _cb

        # ── service ──────────────────────────────────────────────────────────

        def _on_segment_in_view(self, request: Any, response: Any) -> Any:  # noqa: PLR0911 — reason: one named early return per fail-closed precondition; collapsing them would hide which one fired
            """Service: mask the object under a 3-D prompt point in one frame.

            Every failure path is *typed and named* in ``failure_reason`` and
            leaves ``masks`` empty. It never raises: the caller is holding an
            action-acknowledgement barrier open, and a service exception would
            turn a degraded grasp into a stalled robot. An empty reply is the
            caller's cue to fall back conservatively, which it does.
            """
            camera = request.camera.strip() or self._primary_id
            response.camera = camera
            response.ok = False
            response.masks = []
            response.mask_scores_advisory = []
            response.failure_reason = ""

            if self._segmenter is None:  # inactive (model released)
                response.failure_reason = "ROSRuntimeError: segmenter node is not active"
                return response
            frame = self._frames.get(camera)
            if frame is None:
                response.failure_reason = (
                    f"ROSPerceptionStale: no frame cached for camera {camera!r} "
                    f"(known: {sorted(self._frames)})"
                )
                self.get_logger().warning(response.failure_reason)
                return response
            spec = sensor_spec_by_name(self._description, camera)
            if spec is None or spec.intrinsics is None:
                response.failure_reason = (
                    f"ROSConfigError: camera {camera!r} has no SensorSpec intrinsics "
                    "in the robot manifest"
                )
                self.get_logger().warning(response.failure_reason)
                return response

            bgr, width, height = frame
            intrinsics = scale_intrinsics_to(spec.intrinsics, width, height)
            transform = self._prompt_transform(request, spec.frame_id)
            if transform is None:
                response.failure_reason = (
                    f"ROSPerceptionStale: no tf2 {spec.frame_id} <- "
                    f"{request.frame_id or spec.frame_id} at the requested stamp"
                )
                self.get_logger().warning(response.failure_reason)
                return response

            from openral_runner.backends.gstreamer.sam2_segmenter import project_point_to_pixel

            positive = project_point_to_pixel(
                _apply_transform(transform, request.tcp_point), intrinsics
            )
            if positive is None:
                response.failure_reason = (
                    f"ROSConfigError: the TCP point does not project into camera {camera!r} "
                    "(behind the optical plane or outside the image)"
                )
                self.get_logger().warning(response.failure_reason)
                return response
            # Negative prompts are hints (jaw tips). One that misses the image is
            # dropped rather than failing the whole call — the positive point is
            # what makes the prompt meaningful.
            negatives = [
                pixel
                for point in request.negative_points
                if (pixel := project_point_to_pixel(_apply_transform(transform, point), intrinsics))
                is not None
            ]
            try:
                candidates = self._segmenter.segment(
                    bgr,
                    width,
                    height,
                    positive_points=[positive],
                    negative_points=negatives,
                )
            except Exception as exc:  # never crash the service; the caller is waiting
                response.failure_reason = f"{type(exc).__name__}: {exc}"
                self.get_logger().warning(f"segment_in_view failed: {exc}")
                return response
            if not candidates:
                response.failure_reason = (
                    "ROSPerceptionStale: no mask cleared the manifest's min_mask_area_px"
                )
                return response

            response.masks = [
                self._mask_image(candidate.mask, request.stamp, spec.frame_id)
                for candidate in candidates
            ]
            response.mask_scores_advisory = [
                float(candidate.score_advisory) for candidate in candidates
            ]
            response.ok = True
            self.get_logger().info(
                f"segment_in_view: camera={camera!r} candidates={len(candidates)} "
                f"areas={[c.area_px for c in candidates]}"
            )
            # LAST, and only after the reply is complete: the caller is holding
            # an action-acknowledgement barrier open on this call.
            self._publish_debug_masks(response, spec.frame_id)
            return response

        def _publish_debug_masks(self, response: Any, frame_id: str) -> None:
            """Re-publish a successful reply on the diagnostic mask topic.

            No-op unless ``publish_debug_masks`` opened the publisher. The
            message reuses the reply's own ``sensor_msgs/Image`` objects rather
            than re-encoding them, so enabling this costs one serialization and
            no second mask representation.

            Never raises: this is a display path, and a grasp must not degrade
            because a dashboard could not be fed (CLAUDE.md §1.1).
            """
            if self._masks_pub is None:
                return
            try:
                from openral_msgs.msg import SegmentMasks

                msg = SegmentMasks()
                # The CALLER's attach stamp, copied through — a mask that
                # describes a frame that is gone must be detectable as stale.
                msg.header.stamp = response.masks[0].header.stamp
                msg.header.frame_id = frame_id
                msg.camera = response.camera
                msg.rskill_id = self._model_name
                msg.masks = list(response.masks)
                msg.mask_scores_advisory = list(response.mask_scores_advisory)
                self._masks_pub.publish(msg)
            except Exception as exc:  # reason: a diagnostic topic must never degrade a grasp
                self.get_logger().warning(f"debug mask publish failed: {exc}")

        def _prompt_transform(self, request: Any, camera_frame: str) -> Any:
            """``camera_optical <- request.frame_id`` as a 4x4, or ``None``.

            An empty ``frame_id`` means the caller already expressed the prompt
            in the camera's optical frame, which is the identity — stated
            explicitly rather than silently assumed for a populated frame.
            """
            source = request.frame_id.strip()
            if not source or source == camera_frame:
                return np.eye(4, dtype=np.float64)

            import rclpy
            from openral_core.geometry import homogeneous_from_quat_xyz

            for stamp, label in ((request.stamp, "requested stamp"), (rclpy.time.Time(), "latest")):
                try:
                    tf = self._tf_buffer.lookup_transform(camera_frame, source, stamp)
                except Exception as exc:  # tf2 raises several distinct lookup errors
                    self.get_logger().debug(
                        f"tf {camera_frame} <- {source} at {label} unavailable: {exc}"
                    )
                    continue
                if label != "requested stamp":
                    # Say so: the mask is then computed against a pose that is
                    # not the one the caller asked about (CLAUDE.md §1.4).
                    self.get_logger().warning(
                        f"tf {camera_frame} <- {source} unavailable at the requested stamp; "
                        "used the latest transform instead"
                    )
                t, q = tf.transform.translation, tf.transform.rotation
                return homogeneous_from_quat_xyz((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            return None

        def _mask_image(self, mask: NDArray[np.bool_], stamp: Any, frame_id: str) -> Any:
            """Wrap one boolean mask as a ``mono8`` ``sensor_msgs/Image``."""
            height, width = mask.shape
            msg = Image()
            # The CALLER's stamp: this mask describes the attach instant it
            # asked about, and the trace has to replay against that.
            msg.header.stamp = stamp
            msg.header.frame_id = frame_id
            msg.height = int(height)
            msg.width = int(width)
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = int(width)
            msg.data = mono8_bytes_from_mask(mask)
            return msg

    def _apply_transform(transform: Any, point: Any) -> tuple[float, float, float]:
        """Map a ``geometry_msgs/Point`` through a 4x4 homogeneous transform."""
        vec = np.asarray([point.x, point.y, point.z, 1.0], dtype=np.float64)
        out = transform @ vec
        return (float(out[0]), float(out[1]), float(out[2]))

    return SegmenterNode


def make_segmenter_node(node_name: str = "openral_segmenter") -> Any:
    """Construct the real segmenter lifecycle node (ROS imports happen here).

    The seam the live integration test uses to stand up the node in-process.

    Args:
        node_name: ROS node name.

    Returns:
        An un-configured ``SegmenterNode``; drive it with the usual
        ``trigger_configure()`` / ``trigger_activate()`` transitions.
    """
    return _node_class()(node_name)


def main(args: Sequence[str] | None = None) -> None:
    """Entry point: init ROS, spin the segmenter node, shut down cleanly."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = make_segmenter_node()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass  # context already shut down by the SIGINT handler
        finally:
            node.destroy_node()
    finally:
        rclpy.try_shutdown()  # idempotent — no-op if context already shut down


if __name__ == "__main__":
    main()
