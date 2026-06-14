#!/usr/bin/env python3
"""ROS-Image object-detection producer (ADR-0037/0035/0043, no GStreamer).

Subscribes one or more camera ``sensor_msgs/Image`` streams, runs a
GStreamer-free ``openral_runner`` detector backend, and publishes the detector's
``ObjectsMetadata`` as ``openral_msgs/PromptStamped`` on
``/openral/perception/objects``.

Backends, selected by ``manifest_path`` (ADR-0037 2026-06-09 amendment):

* **legacy / RT-DETR** — with no ``manifest_path``, builds an ``ObjectsDetector``
  (RT-DETR ONNX) from ``onnx_path`` + ``labels`` (unchanged behaviour).
* **manifest-driven** — with a ``manifest_path``, builds via
  ``build_manifest_detector``: ONNX for ``runtime: onnx``, or the open-vocabulary
  ``LocateAnythingDetector`` (``VLM_SIDECAR``) for ``runtime: pytorch``.

**Detector mode (ADR-0051).** The manifest's ``detector.mode`` selects how the
node wires the detector (via ``detector_node_wiring``):

* ``continuous`` (default; RT-DETR, ``omdet-turbo-indoor``) — the **primary**
  camera runs the continuous detect+publish leg (streams ``ObjectsMetadata`` into
  world state). The node does **not** expose ``locate_in_view`` or subscribe the
  ``detector_query`` topic — a background producer is not reasoner-prompted.
* ``on_demand`` (``locateanything-3b-nf4``, ``omdet-turbo-locator``) — the node
  exposes the ``locate_in_view`` service + ``detector_query`` topic and does
  **not** publish continuously; every camera's latest frame is still cached so
  the service can answer about the current view.

**Camera-agnostic (ADR-0043).** The node does not bake in a camera name. The
``cameras`` param maps logical camera ids to image topics; with none given it
falls back to the single ``image_topic`` under id ``primary_camera``. The
reasoner picks a viewpoint by camera id.

**locate_in_view service (ADR-0043, on_demand only).** Offers
``/openral/perception/locate_in_view`` (``openral_msgs/srv/LocateInView``): a
read-only "is object X visible in camera Y right now?" — runs a one-shot
detection (``detect_with_query``, without disturbing the persistent query) on the
requested camera's latest cached frame. Driven by the reasoner's
``locate_in_view`` tool. The dynamic ``query_topic`` (std_msgs/String) retargets
the persistent query via ``set_query``.

Parameters:
    cameras (str[]): logical cameras as ``"id=topic"`` entries. Empty = a single
        camera ``primary_camera`` on ``image_topic``.
    primary_camera (str): id of the primary (continuously-detected) camera.
    image_topic (str): single-camera fallback topic.
    output_topic (str): perception topic. Default /openral/perception/objects
    sensor_id (str): sensor name stamped on the metadata. Default "front_depth"
    onnx_path (str): RT-DETR ONNX path (legacy / onnx path).
    manifest_path (str): rSkill manifest path. When set, backend is manifest-driven.
    model_id, score_threshold, input_size, labels: legacy ONNX knobs.
    max_rate_hz (float): continuous publish rate cap. Default 5.0
    query (str): initial open-vocab query override (VLM only).
    query_topic (str): std_msgs/String topic to retarget the continuous VLM query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Operators set this to ``debug`` to surface the continuous leg's per-publish
# DEBUG line (and any other detector DEBUG logs), which the default INFO console
# level hides. Read in ``on_configure``; ``openral deploy sim`` propagates the
# caller's environment to every launched node, so
# ``OPENRAL_DETECTOR_LOG_LEVEL=debug openral deploy sim …`` turns it on without a
# code change or a per-node ``--ros-args --log-level`` the wrapped launch can't
# easily inject.
DETECTOR_LOG_LEVEL_ENV = "OPENRAL_DETECTOR_LOG_LEVEL"

# Accepted spellings → the rclpy ``LoggingSeverity`` member name. ``warning`` is
# the common spelling; rclpy names it ``WARN``.
_LOG_LEVEL_ALIASES = {
    "debug": "DEBUG",
    "info": "INFO",
    "warn": "WARN",
    "warning": "WARN",
    "error": "ERROR",
    "fatal": "FATAL",
}


def normalize_log_level(value: str) -> str | None:
    """Normalise an operator-supplied log level to an rclpy severity name.

    Case-insensitive and whitespace-trimmed; ``warning`` aliases ``WARN``.
    Returns ``None`` for an empty / whitespace-only / unrecognised value so the
    caller leaves the logger at its default level (fail-safe — a typo never
    silences the node).

    Args:
        value: Operator input, e.g. from ``OPENRAL_DETECTOR_LOG_LEVEL``.

    Returns:
        One of ``"DEBUG"`` / ``"INFO"`` / ``"WARN"`` / ``"ERROR"`` / ``"FATAL"``,
        or ``None`` when unset or unrecognised.

    Example:
        >>> normalize_log_level("debug")
        'DEBUG'
        >>> normalize_log_level("  Warning ")
        'WARN'
        >>> normalize_log_level("") is None
        True
    """
    return _LOG_LEVEL_ALIASES.get(value.strip().lower())


def classify_continuous_tick(
    *, error: BaseException | None, detection_count: int | None
) -> tuple[str, str]:
    """Map one continuous detect+publish tick to its ``(log_level, message)``.

    The continuous leg is a best-effort background producer, so a single bad
    frame must not kill it — but it must never fail *silently* (CLAUDE.md §1.4).
    The trap is that the real detector publishes nothing when it sees nothing
    (the ADR-0035 contract the world-state eviction relies on), so on
    ``/openral/perception/objects`` a *crashing* detector (e.g. a CUDA OOM under
    VLA co-residency on a small GPU) is indistinguishable from one watching a
    quiet scene — both leave the topic empty. This pure decision surfaces each
    outcome at the level that makes the leg observable without changing what
    lands on the bus:

    * a detect **exception** → ``warning`` — a crashing/OOM detector must be
      visible, not hidden at DEBUG;
    * an **empty** result (``detection_count`` ``0`` or ``None``) → ``info`` —
      a liveness signal proving the leg is alive and merely sees nothing, vs
      dead/stuck;
    * a **non-empty** result → ``debug`` — the normal, quiet path; the published
      metadata is itself the signal.

    Pure (no rclpy / clock): the caller applies its own throttling
    (``throttle_duration_sec``) and performs the publish, so this is unit-testable
    without a LifecycleNode or executor.

    Args:
        error: The exception ``detect()`` raised this tick, or ``None``.
        detection_count: Number of detections in the metadata, ``0`` for an empty
            result, or ``None`` when ``detect()`` returned no metadata.

    Returns:
        ``(level, message)`` where ``level`` is one of ``"warning"`` / ``"info"``
        / ``"debug"``.

    Example:
        >>> classify_continuous_tick(error=RuntimeError("CUDA OOM"), detection_count=None)[0]
        'warning'
        >>> classify_continuous_tick(error=None, detection_count=0)[0]
        'info'
        >>> classify_continuous_tick(error=None, detection_count=2)[0]
        'debug'
    """
    if error is not None:
        return (
            "warning",
            f"continuous detect raised {error!r}; nothing published this frame "
            "(a crashing detector looks identical to a quiet scene on the bus)",
        )
    if not detection_count:  # 0 or None — alive but saw nothing this frame
        return ("info", "continuous leg alive: 0 detections this frame (quiet scene)")
    return ("debug", f"continuous leg published {detection_count} detection(s)")


def main(args: Any = None) -> None:
    """Entry point: init ROS, spin the detector node, shut down cleanly."""
    import rclpy
    from openral_msgs.msg import PromptStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from openral_perception_ros.image_convert import ImageConvertError, image_to_bgr_bytes

    class RosImageObjectDetectorNode(LifecycleNode):  # type: ignore[misc]
        """Subscribe camera Image(s), detect objects, publish + serve queries.

        ADR-0050 — a *managed* lifecycle node. The (GPU-heavy) detector backend
        is built on ``on_activate`` and released on ``on_deactivate``, so the
        reasoner can free the detector's VRAM (via ``LifecycleTransitionTool``)
        before a co-resident grab policy loads on an 8 GB GPU. Cameras, the
        publisher, subscriptions and the ``locate_in_view`` service live for the
        configured→cleanup span; only the detector model tracks active→inactive.
        """

        def __init__(self) -> None:
            super().__init__("openral_ros_image_detector")
            self.declare_parameter("cameras", [""])
            self.declare_parameter("primary_camera", "default")
            self.declare_parameter("image_topic", "/openral/cameras/agentview_left/image")
            self.declare_parameter("output_topic", "/openral/perception/objects")
            self.declare_parameter("sensor_id", "front_depth")
            self.declare_parameter("onnx_path", "")
            self.declare_parameter("manifest_path", "")
            self.declare_parameter("model_id", "rtdetr-coco-r18")
            self.declare_parameter("score_threshold", 0.5)
            self.declare_parameter("input_size", 640)
            self.declare_parameter("max_rate_hz", 5.0)
            self.declare_parameter("labels", [""])
            self.declare_parameter("query", "")
            self.declare_parameter("query_topic", "/openral/perception/detector_query")

            self._last_pub_ns = 0
            # Latest BGR frame per camera id, for the on-demand locate_in_view service.
            self._frames: dict[str, tuple[bytes, int, int]] = {}
            # Built across lifecycle transitions (ADR-0050). The detector model
            # is the only GPU-resident piece; it tracks active→inactive.
            self._detector: Any = None
            self._cameras: dict[str, str] = {}
            self._primary_id = ""
            self._sensor_id = ""
            self._min_period_ns = int(1e9 / 5.0)
            self._subs: list[Any] = []
            self._query_sub: Any = None
            self._pub: Any = None
            self._srv: Any = None
            # ADR-0051 detector-mode wiring (continuous publish leg vs on-demand
            # locate_in_view service); resolved from the manifest at on_configure.
            self._wiring: Any = None

        def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Wire cameras, publisher, subscriptions and the service (no GPU yet)."""
            del state
            self._apply_env_log_level()
            gp = self.get_parameter
            self._sensor_id = gp("sensor_id").get_parameter_value().string_value
            self._min_period_ns = int(
                1e9 / max(gp("max_rate_hz").get_parameter_value().double_value, 0.1)
            )
            self._cameras = self._resolve_cameras()
            self._primary_id = next(iter(self._cameras))
            self._wiring = self._resolve_wiring()

            img_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            out_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._pub = self.create_publisher(
                PromptStamped, gp("output_topic").get_parameter_value().string_value, out_qos
            )
            # One subscription per camera. ADR-0051: the primary camera runs the
            # continuous detect+publish leg ONLY for a `continuous` detector; an
            # `on_demand` locator caches frames but does not publish. Bind cid via
            # a default arg.
            self._subs = []
            for cid, topic in self._cameras.items():
                cb = (
                    self._make_primary_cb(cid)
                    if cid == self._primary_id and self._wiring.run_continuous_leg
                    else self._make_cache_cb(cid)
                )
                self._subs.append(self.create_subscription(Image, topic, cb, img_qos))

            # locate_in_view service (ADR-0043) — only for `on_demand` detectors
            # (ADR-0051) and only if the IDL is built.
            self._srv = None
            if self._wiring.serve_on_demand:
                try:
                    from openral_msgs.srv import LocateInView

                    self._srv = self.create_service(
                        LocateInView,
                        "/openral/perception/locate_in_view",
                        self._on_locate_in_view,
                    )
                except ImportError:
                    self.get_logger().warning(
                        "openral_msgs/srv/LocateInView not built; locate_in_view service disabled"
                    )

            self.get_logger().info(
                f"ros_image_detector configured: cameras={self._cameras} "
                f"primary={self._primary_id!r} sensor_id={self._sensor_id}, "
                f"mode={'on_demand' if self._wiring.serve_on_demand else 'continuous'}, "
                f"continuous_leg={'on' if self._wiring.run_continuous_leg else 'off'}, "
                f"locate_in_view={'on' if self._srv else 'off'}"
            )
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Build (load) the detector backend — acquires GPU VRAM (ADR-0050)."""
            gp = self.get_parameter
            if self._detector is None:
                onnx_path = gp("onnx_path").get_parameter_value().string_value
                manifest_path = gp("manifest_path").get_parameter_value().string_value
                self._detector = self._build_detector(onnx_path, manifest_path)
                initial_query = gp("query").get_parameter_value().string_value
                if initial_query and hasattr(self._detector, "set_query"):
                    self._detector.set_query(initial_query)
                # ADR-0051: the detector_query retarget topic is wired only for an
                # `on_demand` locator — a `continuous` background producer is not
                # reasoner-retargetable even if its backend exposes set_query.
                if (
                    self._wiring.serve_on_demand
                    and hasattr(self._detector, "set_query")
                    and self._query_sub is None
                ):
                    self._query_sub = self.create_subscription(
                        String,
                        gp("query_topic").get_parameter_value().string_value,
                        self._on_query,
                        1,
                    )
            self.get_logger().info("ros_image_detector activated (detector loaded).")
            return super().on_activate(state)

        def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Release the detector backend — frees its (GPU) VRAM (ADR-0050)."""
            self._release_detector()
            self.get_logger().info("ros_image_detector deactivated (detector VRAM released).")
            return super().on_deactivate(state)

        def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Tear down the detector, publisher, subscriptions, and service."""
            del state
            self._release_detector()
            for sub in self._subs:
                self.destroy_subscription(sub)
            self._subs = []
            if self._query_sub is not None:
                self.destroy_subscription(self._query_sub)
                self._query_sub = None
            if self._srv is not None:
                self.destroy_service(self._srv)
                self._srv = None
            if self._pub is not None:
                self.destroy_publisher(self._pub)
                self._pub = None
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
            """Force cleanup."""
            return self.on_cleanup(state)

        def _release_detector(self) -> None:
            """ADR-0050 — release the detector backend, freeing its VRAM.

            Best-effort + idempotent: a backend without ``close()`` (the ONNX
            path) or one whose teardown raises must not break the transition.
            """
            detector = self._detector
            self._detector = None
            if detector is None:
                return
            close = getattr(detector, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # reason: teardown must not raise from a lifecycle cb
                    self.get_logger().warning(f"detector close failed: {exc}")

        def _resolve_wiring(self) -> Any:
            """Resolve the ADR-0051 detector-mode node wiring from the manifest.

            With a ``manifest_path``, the detector's ``mode`` selects continuous
            (publish leg) vs on-demand (locate_in_view service). The legacy ONNX
            path (no manifest) is a continuous background producer.
            """
            from openral_core.schemas import DetectorMode, RSkillManifest
            from openral_runner.backends.gstreamer.detector_factory import detector_node_wiring

            manifest_path = self.get_parameter("manifest_path").get_parameter_value().string_value
            if manifest_path:
                manifest = RSkillManifest.from_yaml(manifest_path)
                if manifest.detector is not None:
                    return detector_node_wiring(manifest.detector.mode)
            return detector_node_wiring(DetectorMode.CONTINUOUS)

        def _resolve_cameras(self) -> dict[str, str]:
            """Resolve the camera-id -> topic map (camera-agnostic, ADR-0043)."""
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

        def _build_detector(self, onnx_path: str, manifest_path: str) -> Any:
            """Build the detector backend (manifest-driven, or legacy ONNX)."""
            if manifest_path:
                from openral_core.schemas import RSkillManifest
                from openral_runner.backends.gstreamer.detector_factory import (
                    build_manifest_detector,
                )

                manifest = RSkillManifest.from_yaml(manifest_path)
                detector, tier = build_manifest_detector(manifest, onnx_path=onnx_path or None)
                self.get_logger().info(f"detector tier={tier.value} model={manifest.name}")
                return detector

            from openral_runner.backends.gstreamer.objects_detector import ObjectsDetector

            gp = self.get_parameter
            labels = [s for s in gp("labels").get_parameter_value().string_array_value if s]
            input_size = gp("input_size").get_parameter_value().integer_value
            return ObjectsDetector(
                onnx_path,
                labels=labels,
                model_id=gp("model_id").get_parameter_value().string_value,
                input_size=(input_size, input_size),
                score_threshold=gp("score_threshold").get_parameter_value().double_value,
            )

        def _make_cache_cb(self, cid: str) -> Callable[[Any], None]:
            def _cb(msg: Any) -> None:
                self._cache_frame(cid, msg)

            return _cb

        def _make_primary_cb(self, cid: str) -> Callable[[Any], None]:
            def _cb(msg: Any) -> None:
                self._cache_frame(cid, msg)
                self._detect_and_publish(msg)

            return _cb

        def _cache_frame(self, cid: str, msg: Any) -> None:
            try:
                bgr, w, h = image_to_bgr_bytes(msg)
            except ImageConvertError as exc:
                self.get_logger().debug(f"cache_frame({cid}): convert failed: {exc}")
                return
            self._frames[cid] = (bgr, w, h)

        def _on_query(self, msg: Any) -> None:
            """Retarget the continuous open-vocab leg from a std_msgs/String."""
            if self._detector is None:
                return
            query = msg.data.strip()
            if not query:
                return
            try:
                self._detector.set_query(query)
                self.get_logger().info(f"detector query set to {query!r}")
            except Exception as exc:  # best-effort; a bad query must not crash perception
                self.get_logger().warning(f"set_query({query!r}) failed: {exc}")

        def _apply_env_log_level(self) -> None:
            """Honour ``OPENRAL_DETECTOR_LOG_LEVEL`` to make DEBUG visible on demand.

            The default ROS console level (INFO) hides the continuous leg's
            per-publish DEBUG line. Setting the env var to ``debug`` before
            ``openral deploy sim`` (whose launch propagates the environment to
            every node) raises this node's logger so those lines show, without a
            ``--ros-args --log-level`` the wrapped launch can't easily inject.
            """
            import os

            from rclpy.logging import LoggingSeverity, set_logger_level

            level_name = normalize_log_level(os.environ.get(DETECTOR_LOG_LEVEL_ENV, ""))
            if level_name is None:
                return
            set_logger_level(self.get_logger().name, LoggingSeverity[level_name])
            self.get_logger().info(
                f"detector logger level set to {level_name} via {DETECTOR_LOG_LEVEL_ENV}"
            )

        def _log_throttled(self, level: str, message: str) -> None:
            """Emit ``message`` at ``level``, throttled to once every few seconds.

            A persistent state — a per-frame detect crash, or a long quiet
            stretch — would otherwise log at the camera rate; the throttle keeps
            it to one line every ``throttle_duration_sec``.
            """
            log = self.get_logger()
            {"warning": log.warning, "info": log.info, "debug": log.debug}[level](
                message, throttle_duration_sec=5.0
            )

        def _detect_and_publish(self, msg: Any) -> None:
            if self._detector is None:  # inactive (ADR-0050 — VRAM released)
                return
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_pub_ns < self._min_period_ns:
                return
            frame = self._frames.get(self._primary_id)
            if frame is None:
                return
            bgr, w, h = frame
            # Best-effort producer: one bad frame must not kill the leg — but it
            # must never fail SILENTLY. The real detector publishes nothing when it
            # sees nothing (ADR-0035), so a swallowed detect() crash (e.g. a CUDA
            # OOM under VLA co-residency) is invisible on /openral/perception/objects
            # — identical to a quiet scene. classify_continuous_tick raises that
            # crash to WARNING and logs a quiet-scene liveness heartbeat at INFO so
            # the leg is observable without changing what lands on the bus (issue
            # #12 / CLAUDE.md §1.4).
            try:
                md = self._detector.detect(bgr, w, h, self._sensor_id)
            except Exception as exc:  # surfaced at WARNING below, never swallowed
                level, message = classify_continuous_tick(error=exc, detection_count=None)
                self._log_throttled(level, message)
                return
            level, message = classify_continuous_tick(
                error=None, detection_count=None if md is None else len(md.detections)
            )
            self._log_throttled(level, message)
            if md is None:
                return
            self._last_pub_ns = now_ns
            out = PromptStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self._sensor_id
            out.text = f"{len(md.detections)} objects"
            out.metadata_json = md.model_dump_json()
            self._pub.publish(out)

        def _on_locate_in_view(self, request: Any, response: Any) -> Any:
            """Service (ADR-0043): one-shot 'is X in camera Y right now?'."""
            query = request.query.strip()
            camera = request.camera.strip() or self._primary_id
            response.camera = camera
            if self._detector is None:  # inactive (ADR-0050 — VRAM released)
                response.found = False
                response.metadata_json = ""
                return response
            frame = self._frames.get(camera)
            if frame is None:
                response.found = False
                response.metadata_json = ""
                self.get_logger().warning(
                    f"locate_in_view: no frame for camera {camera!r} "
                    f"(known: {sorted(self._frames)})"
                )
                return response
            bgr, w, h = frame
            try:
                if query and hasattr(self._detector, "detect_with_query"):
                    md = self._detector.detect_with_query(bgr, w, h, self._sensor_id, query)
                else:
                    md = self._detector.detect(bgr, w, h, self._sensor_id)
            except Exception as exc:  # best-effort; never crash the service
                self.get_logger().warning(f"locate_in_view detect failed: {exc}")
                response.found = False
                response.metadata_json = ""
                return response
            if md is None or not md.detections:
                response.found = False
                response.metadata_json = ""
            else:
                response.found = True
                response.metadata_json = md.model_dump_json()
            self.get_logger().info(
                f"locate_in_view: query={query!r} camera={camera!r} found={response.found}"
            )
            return response

    rclpy.init(args=args)
    node = RosImageObjectDetectorNode()
    try:
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            # Normal teardown path. rclpy installs a SIGINT handler at
            # `rclpy.init()` that shuts down the context AND raises
            # KeyboardInterrupt out of `rclpy.spin()` on Jazzy. On
            # ROS 2 Rolling / a manual `rclpy.shutdown()` from another
            # thread, spin instead raises ExternalShutdownException.
            # Either way the context is already shut down by the time we
            # reach the `finally` below, so the bare `rclpy.shutdown()`
            # we used to call there raised
            # `RCLError: rcl_shutdown already called` — the
            # `try_shutdown()` switch below is the corresponding fix.
            pass
        finally:
            node.destroy_node()
    finally:
        # Idempotent — no-op when the SIGINT handler (or whoever fired
        # ExternalShutdownException) already shut down the context.
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
