"""Real-mode camera leg: open every deploy-bound sensor and publish to ROS.

Real hardware has no camera publisher (unlike the sim HAL, which publishes frames itself).
Physical ``/dev/video*`` devices are described by ``deploy_binding``
— on the robot manifest for robot-mounted cameras (wrist/head), on
``sensors`` for workcell-mounted ones (overhead/front).

``open_deploy_sensor_readers`` opens one reader per bound spec and publishes to
``<topic_prefix>/<name>/image`` (BEST_EFFORT QoS, matching WorldState's subscription):

* ``gstreamer`` — native in-pipeline ROS tee.
* ``opencv_thread`` (or any tee-less backend) — wrapped in a polling
  ``SensorRosPublisher``. Calibrated ``intrinsics`` also
  publish ``CameraInfo`` on ``<topic_prefix>/<name>/camera_info`` (sim HAL's layout), enabling
  mono visual SLAM on real hardware.

**Direct aggregator path (zero-copy vision path).** When ``aggregator`` is passed (reader,
aggregator, and skill runner share one process), an ``_AggregatorPump`` per reader writes
``read_latest()`` frames straight into ``WorldStateAggregator.update_image_frame`` — no ROS
serialize round trip, NVMM ``SensorFrame.handle`` stays intact — and emits the dashboard span
at full reader cadence. List these sensors in WorldState's ``direct_image_frame_sensors`` so
``_on_image`` skips them; the ROS tee keeps serving its other subscribers (detector, reward
monitor, reasoner).

Call ``SensorLeg.start`` only after the composed ROS lifecycle nodes are configured; own
teardown via ``SensorLeg.close``. Wired in by ``runtime_node`` when ``deploy_config`` is
set (real deploys only).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from openral_core import SensorSpec

__all__ = [
    "SensorLeg",
    "merge_deploy_sensors",
    "open_deploy_sensor_readers",
    "slam_camera_names",
    "topic_frame_size",
]

log = structlog.get_logger(__name__)

#: WorldState's camera subscription prefix (`<prefix>/<name>/image`).
DEFAULT_TOPIC_PREFIX = "/openral/cameras"

#: Publish cadence when the binding's backend_params carry no fps.
#: Matches the WorldStateAggregator staleness-gate expectation (10 Hz cameras).
_DEFAULT_PUBLISH_RATE_HZ = 10.0

#: Channel count of an interleaved colour frame — the only layout the 180°
#: dashboard flip knows how to reshape.
_RGB_CHANNELS: Final[int] = 3

# Cap on the ROS topic cadence for the PYTHON FALLBACK publisher only (the native GStreamer
# tee publishes from inside the pipeline with no Python conversion, so it is NOT capped).
#
# Each publish hands a full-res `sensor_msgs/Image` to rclpy; the Python->C conversion holds
# the GIL for the ~900 KiB copy. SO-101 bench profile (reward monitor co-resident, so Cyclone
# must actually serialise): `py-spy record --gil` found 89.5% of GIL-holding samples in that
# one call vs 0.3% for the VLA-load thread — a 13 s SmolVLA load stretched to 86 s on a
# 22-core host.
#
# The policy bypasses this topic entirely (reads the shared aggregator directly, via
# `direct_image_frame_sensors`); the dashboard uses the same pump, via
# `_emit_frame_observability`, at full cadence. Remaining subscribers: reward monitor (~1 Hz),
# object detector, reasoner completion camera (0.2 Hz) — nothing needs 30 Hz.
#
# 3 Hz not 5: re-profiling after the first cap still showed 52.75% of GIL samples in
# `_publish_frame` (22.7 s / 43 s GIL-held over 75 s wall, ~30 ms per 640x480 publish). Floor
# is `WorldStateAggregator.staleness_limit_s` (0.5 s) — 2 Hz sits exactly on it and would flap
# per-sensor diagnostics.
_MAX_FALLBACK_TOPIC_RATE_HZ = 3.0

#: Resolution ceiling for the Python fallback topic when no launch-time
#: consumer needs native pixels. 320x240 clears every remaining subscriber —
#: Robometer and TOPReward declare a 224x224 minimum and the reasoner's
#: completion VLM resizes internally — while cutting the per-publish GIL cost
#: ~4x with the payload. The policy is untouched: it reads the aggregator
#: in-process at full capture resolution, which is what ACT (no resize at all,
#: 640x480 exact) and SmolVLA (its own 512x512 pad-resize) actually consume.
_DEFAULT_TOPIC_MAX_SIZE: Final[tuple[int, int]] = (320, 240)


class _Closeable(Protocol):
    """Structural type for anything with a no-arg close/stop."""

    def close(self) -> None: ...  # pragma: no cover — Protocol


def _emit_frame_observability(sensor_name: str, frame: Any, flip_180: bool) -> None:
    """Emit the dashboard's ``sensors.read_latest`` span for a pump-fed frame.

    WorldState's ``_on_image`` normally produces this span, but the ROS tee caps at
    ``_MAX_FALLBACK_TOPIC_RATE_HZ`` (3 Hz) — pump-fed cameras emit here instead, at full
    reader cadence, and ``_on_image`` skips them.

    Affordable: Pillow drops the GIL for resize/encode, measured 2.42 ms/frame at 320x240 q60
    (60 thumbnails/s costs 4.5% of a competing thread's GIL time, vs 89.5% for the uncapped
    full-res topic). Display-only. Shares
    ``openral_observability.producer.emit_sensor_frame_span`` with ``_on_image`` so
    pump-fed and tee-fed cameras render identically.
    """
    from openral_observability import producer as ral_producer

    ral_producer.emit_sensor_frame_span(
        frame,
        sensor_name=sensor_name,
        age_ms=max(0.0, (time.monotonic_ns() - frame.stamp_monotonic_ns) / 1e6),
        flip_180=flip_180,
        tracer_name="openral_rskill_ros.sensor_leg",
    )


class _AggregatorPump:
    """Poll a reader's latest frame straight into the shared aggregator.

    The in-process sibling of ``SensorRosPublisher``: same start/stop shape,
    but the destination is ``WorldStateAggregator.update_image_frame`` — the
    frame object (including a zero-copy NVMM ``handle``) reaches the skill
    runner without a ROS serialize/deserialize (the zero-copy vision path).

    A frame is written only when its monotonic stamp changed, so re-polling
    the same latched frame never refreshes the aggregator's staleness stamp.
    """

    # reader/aggregator duck-typed (SensorReader / WorldStateAggregator): imports stay deferred.
    def __init__(self, reader: Any, sensor_name: str, aggregator: Any, rate_hz: float) -> None:
        """Stash config; the polling thread starts in ``start``."""
        self._reader = reader
        self._sensor_name = sensor_name
        self._aggregator = aggregator
        self._period_s = 1.0 / max(rate_hz, 1.0)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_stamp_ns: int | None = None
        # Read once, same env var WorldState's ``_on_image`` and the sim
        # sensor bridge honour — display orientation, never the policy frame.
        self._flip_180 = os.environ.get("OPENRAL_DASHBOARD_FLIP_180", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def start(self) -> None:
        """Spawn the polling daemon thread. Idempotent."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"agg-pump-{self._sensor_name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal and join the polling thread. Idempotent."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        """Poll ``read_latest`` at the configured rate; write new frames only."""
        next_deadline = time.monotonic() + self._period_s
        while not self._stop_event.is_set():
            try:
                frame = self._reader.read_latest(max_age_ms=None)
            except Exception:  # reason: no frame yet / transient staleness — keep polling
                frame = None
            if frame is not None and frame.stamp_monotonic_ns != self._last_stamp_ns:
                self._last_stamp_ns = frame.stamp_monotonic_ns
                self._aggregator.update_image_frame(self._sensor_name, frame)
                # The policy already has the frame; everything below is display.
                # A broken thumbnail must never stop the aggregator being fed,
                # so this is best-effort — logged, never swallowed silently.
                try:
                    _emit_frame_observability(self._sensor_name, frame, self._flip_180)
                except Exception:
                    log.warning(
                        "sensor_leg.thumbnail_failed", sensor_id=self._sensor_name, exc_info=True
                    )
            remaining = next_deadline - time.monotonic()
            if remaining > 0 and self._stop_event.wait(timeout=remaining):
                return
            next_deadline += self._period_s
            if time.monotonic() > next_deadline + self._period_s:
                next_deadline = time.monotonic() + self._period_s


@dataclass
class SensorLeg:
    """Open readers + prepared publishers for one deploy session.

    Attributes:
        readers: Open ``SensorReader`` instances, one per deploy-bound
            ``SensorSpec`` (gstreamer readers publish via their
            internal ROS tee).
        publishers: Prepared ``SensorRosPublisher`` pumps for the
            readers without a native ROS tee. Parallel list, NOT
            index-aligned with ``readers``.
    """

    readers: list[object] = field(default_factory=list)
    publishers: list[object] = field(default_factory=list)
    #: Sensors written straight into the shared aggregator (zero-copy vision path).
    #: WorldState's ``direct_image_frame_sensors`` parameter must list these so
    #: ``_on_image`` skips them: the pump already owns both their aggregator
    #: write and their dashboard span.
    direct_sensors: list[str] = field(default_factory=list)

    def start(self) -> None:
        """Start every prepared publisher after ROS lifecycle configuration.

        Entity creation and lifecycle transitions remain single-threaded.
        Only after those complete may camera pumps publish concurrently.
        """
        try:
            for publisher in self.publishers:
                publisher.start()  # type: ignore[attr-defined]  # reason: duck-typed pump
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Stop publishers first (they poll the readers), then close readers.

        Idempotent and exception-safe: a failing stop/close never blocks
        the remaining teardown — deploy shutdown must always reach the
        HAL/lifecycle teardown behind it.
        """
        for publisher in self.publishers:
            try:
                publisher.stop()  # type: ignore[attr-defined]  # reason: duck-typed pump
            except Exception as exc:  # reason: teardown must not raise
                log.warning("sensor_leg.publisher_stop_failed", error=str(exc))
        self.publishers.clear()
        for reader in self.readers:
            try:
                reader.close()  # type: ignore[attr-defined]  # reason: SensorReader protocol
            except Exception as exc:  # reason: teardown must not raise
                log.warning("sensor_leg.reader_close_failed", error=str(exc))
        self.readers.clear()


def merge_deploy_sensors(
    manifest_sensors: Iterable[SensorSpec],
    scene_sensors: Iterable[SensorSpec],
) -> list[SensorSpec]:
    """Robot-manifest sensors ∪ ``DeployScene.sensors``, merged field-wise.

    On a name collision the scene's explicitly-set fields win (via ``model_fields_set``, so an
    unmentioned field falls through to the manifest) and the manifest fills the rest — exactly
    one spec survives per name, else the device would be opened and its topic published twice.

    Manifest owns robot-side geometry (``parent_frame`` / ``static_transform_xyz_rpy`` /
    ``intrinsics``); scene owns the host-side binding (device, topic, fps).
    """
    scene = list(scene_sensors)
    by_name = {s.name: s for s in scene}
    merged: list[SensorSpec] = []
    for spec in manifest_sensors:
        override = by_name.pop(spec.name, None)
        if override is None:
            merged.append(spec)
            continue
        merged.append(
            spec.model_copy(update={f: getattr(override, f) for f in override.model_fields_set})
        )
    # Scene-only sensors (workcell-mounted) keep their declaration order.
    merged.extend(s for s in scene if s.name in by_name)
    return merged


def _publish_rate_hz(spec: SensorSpec) -> float:
    """The ROS publish cadence for ``spec`` — binding fps, else spec rate, else 10 Hz."""
    assert spec.deploy_binding is not None  # reason: caller filters on binding
    fps = spec.deploy_binding.backend_params.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return float(fps)
    if spec.rate_hz > 0:
        return float(spec.rate_hz)
    return _DEFAULT_PUBLISH_RATE_HZ


#: The camera names visual SLAM tracks when the scene does not name them.
#: ``DeployRuntime.slam_stereo_cameras=None`` means "the impl's built-in
#: left/right default", so an implicit stereo rig must be exempted too — a
#: scene that never mentions camera names would otherwise be silently capped.
_DEFAULT_SLAM_STEREO_CAMERAS: Final[tuple[str, str]] = ("left", "right")


def slam_camera_names(runtime: object | None) -> frozenset[str]:
    """Camera names feeding visual SLAM, which must never be rate-capped.

    cuVSLAM / PyCuVSLAM lose the track on a starved stream, so these cameras keep full cadence
    regardless of ``_MAX_FALLBACK_TOPIC_RATE_HZ``. Derived from the scene's
    ``DeployRuntime`` rather than a hand-written per-binding override, so a stereo deploy can't
    silently degrade a tracking input by forgetting the flag.

    Covers both spellings: explicit ``slam_stereo_cameras`` / ``slam_mono_camera``, and the
    implicit ``left``/``right`` pair a ``None`` stereo field resolves to downstream.

    Args:
        runtime: A ``DeployRuntime`` (or ``None`` when the scene pins no runtime block).
            Duck-typed so this module keeps no schema import.

    Returns:
        Camera names exempt from the cap. Empty when SLAM is off — the cap then applies to
        every camera.

    Example:
        >>> slam_camera_names(None)
        frozenset()
    """
    if runtime is None or not getattr(runtime, "enable_slam", False):
        return frozenset()
    names: set[str] = set()
    stereo = getattr(runtime, "slam_stereo_cameras", None)
    names.update(stereo if stereo else _DEFAULT_SLAM_STEREO_CAMERAS)
    mono = getattr(runtime, "slam_mono_camera", None)
    if mono:
        names.add(str(mono))
    return frozenset(names)


def apply_launch_overrides(
    runtime: object | None,
    *,
    enable_object_detector: bool | None = None,
    enable_slam: bool | None = None,
    slam_stereo_cameras: tuple[str, ...] | None = None,
    slam_mono_camera: str | None = None,
) -> object | None:
    """Fold the launch's RESOLVED consumer flags over the scene's runtime block.

    Scene YAML's ``enable_object_detector`` / ``enable_slam`` are tri-state (``None`` = auto),
    resolved by the deploy CLI at launch time (detector ON when a backend exists; SLAM
    auto-enables from the robot manifest). The runtime node re-reads the original YAML, so
    ``slam_camera_names`` / ``topic_frame_size`` consuming the raw block would treat an
    auto-enabled leg as OFF and silently starve the cameras the detector/SLAM nodes subscribe to.

    Each ``None`` override keeps the scene's value (bare ``runtime_node`` runs unchanged); a
    launch that resolves its flags always passes them — the ground truth of which subscribers
    exist in the graph.

    Args:
        runtime: The scene's ``DeployRuntime`` (or ``None``). Duck-typed so
            this module keeps no schema import.
        enable_object_detector: Launch-resolved detector flag, or ``None``.
        enable_slam: Launch-resolved SLAM flag, or ``None``.
        slam_stereo_cameras: Launch-resolved stereo camera names, or ``None``.
        slam_mono_camera: Launch-resolved mono camera name, or ``None``.

    Returns:
        A duck-typed object carrying exactly the four attributes the two
        consumers read, with overrides applied.

    Example:
        >>> merged = apply_launch_overrides(None, enable_slam=True)
        >>> sorted(slam_camera_names(merged))
        ['left', 'right']
    """
    if runtime is None and all(
        v is None
        for v in (enable_object_detector, enable_slam, slam_stereo_cameras, slam_mono_camera)
    ):
        # No runtime block AND no launch opinion: preserve the callers'
        # conservative ``None`` contract (native pixels, cap-only rates).
        return None

    def _pick(override: object | None, attr: str) -> object | None:
        return override if override is not None else getattr(runtime, attr, None)

    return SimpleNamespace(
        enable_object_detector=_pick(enable_object_detector, "enable_object_detector"),
        enable_slam=_pick(enable_slam, "enable_slam"),
        slam_stereo_cameras=_pick(slam_stereo_cameras, "slam_stereo_cameras"),
        slam_mono_camera=_pick(slam_mono_camera, "slam_mono_camera"),
    )


def topic_frame_size(runtime: object | None) -> tuple[int, int] | None:
    """Resolution ceiling for the fallback topic, or ``None`` for native pixels.

    Derived from the scene's ``DeployRuntime`` at launch, deliberately NOT per-rSkill: the
    reasoner picks skills at runtime (a size chosen for the active skill would go stale on
    switch), and the topic's subscribers are fixed by launch flags, not by which policy is
    loaded. The policy never reads this topic.

    Returns ``None`` whenever a subscriber needs full pixels:

    * **object detector** — declares ``input_size`` 640; which camera it watches is a runtime
      detail, so the whole topic stays native.
    * **visual SLAM** — cuVSLAM triangulates against calibrated intrinsics; per-camera exemption
      is handled via ``slam_camera_names``, but a SLAM scene keeps every camera native
      rather than betting the name list is complete.

    Args:
        runtime: A ``DeployRuntime`` (or ``None`` when the scene pins no runtime block).
            Duck-typed so this module keeps no schema import.

    Returns:
        ``(width, height)`` ceiling, or ``None`` to publish at capture size.
    """
    if runtime is None:
        return None
    if getattr(runtime, "enable_object_detector", False):
        return None
    if getattr(runtime, "enable_slam", False):
        return None
    return _DEFAULT_TOPIC_MAX_SIZE


def _fallback_topic_rate_hz(spec: SensorSpec, uncapped: Collection[str] = ()) -> float:
    """ROS cadence for the Python fallback publisher — capped, with an opt-out.

    Capture rate and topic rate differ: readers keep running at full fps (the policy reads the
    freshest frame in-process), but republishing each frame through rclpy costs a GIL-held 900
    KiB conversion per tick — see ``_MAX_FALLBACK_TOPIC_RATE_HZ``. A scene asking for a
    *slower* rate than the cap is honoured as-is; the cap only ever lowers.

    Two ways out of the cap, in precedence order:

    1. ``uncapped`` — camera names kept at full cadence. Filled from ``slam_camera_names``,
       so visual SLAM is exempt automatically.
    2. ``backend_params["topic_rate_hz"]`` — an exact cadence for any other rate-sensitive
       out-of-process consumer.

    Args:
        spec: A sensor spec carrying a ``deploy_binding``.
        uncapped: Camera names exempt from the cap (see ``slam_camera_names``).

    Returns:
        The full configured rate when exempt, else ``backend_params["topic_rate_hz"]`` when set
        and positive, else ``min(configured rate, _MAX_FALLBACK_TOPIC_RATE_HZ)``.
    """
    assert spec.deploy_binding is not None  # reason: caller filters on binding
    if spec.name in uncapped:
        return _publish_rate_hz(spec)
    explicit = spec.deploy_binding.backend_params.get("topic_rate_hz")
    # bool is an int subclass: YAML `topic_rate_hz: true` would otherwise
    # parse as 1.0 Hz — below the staleness floor — instead of being ignored.
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit > 0:
        return float(explicit)
    return min(_publish_rate_hz(spec), _MAX_FALLBACK_TOPIC_RATE_HZ)


def _await_first_frame(
    reader: Any, sensor_id: str, *, attempts: int = 3, timeout_s: float = 6.0
) -> None:
    """Block until ``reader`` streams its first frame, reopening on a bus error.

    Serialises camera cold-start (call between consecutive ``reader.open()``s): two USB MJPG
    cameras whose ``nvjpegdec`` pipelines negotiate PLAYING simultaneously can race, and the
    loser latches a v4l2 ``streaming stopped (-5)`` bus error with no self-retry. Waiting for a
    frame before opening the next camera staggers negotiation; a bounded close+reopen recovers a
    lost race once the other camera already streams. Fast no-op once frames flow (tens of ms).

    Args:
        reader: An open ``SensorReader`` (``read_latest`` / ``open`` / ``close``).
        sensor_id: Sensor name, for diagnostics.
        attempts: Max open attempts before giving up.
        timeout_s: Per-attempt wait for the first frame.

    Raises:
        ROSRuntimeError: No frame after ``attempts`` open attempts.
    """
    from openral_core.exceptions import ROSPerceptionStale, ROSRuntimeError

    for attempt in range(1, attempts + 1):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                reader.read_latest(max_age_ms=None)
                return  # streaming — first fresh frame arrived
            except ROSPerceptionStale:
                time.sleep(0.05)  # no frame yet — keep polling
            except ROSRuntimeError as exc:  # bus error (e.g. v4l2 -5) — reopen
                log.warning(
                    "sensor_leg.camera_reopen",
                    sensor_id=sensor_id,
                    attempt=attempt,
                    error=str(exc),
                )
                reader.close()
                reader.open()
                break  # re-enter the wait loop against the reopened pipeline
    raise ROSRuntimeError(
        f"sensor_leg: camera {sensor_id!r} produced no frame after {attempts} "
        f"open attempts ({timeout_s:.0f}s each)"
    )


def open_deploy_sensor_readers(
    sensors: Iterable[SensorSpec],
    *,
    topic_prefix: str = DEFAULT_TOPIC_PREFIX,
    aggregator: Any | None = None,  # reason: WorldStateAggregator — deferred import
    ros_node: Any | None = None,  # reason: composed rclpy node — deferred import
    uncapped_sensors: Collection[str] = (),
    topic_max_size: tuple[int, int] | None = None,
) -> SensorLeg:
    """Open deploy-bound sensors and prepare their ROS publishing resources.

    Args:
        sensors: Robot-manifest sensors plus ``DeployScene.sensors`` (caller concatenates).
            Specs without a ``deploy_binding`` are skipped.
        topic_prefix: WorldState's ``camera_topic_prefix``. Final topic is
            ``<topic_prefix>/<spec.name>/image``.
        aggregator: The composed runtime's shared ``WorldStateAggregator``. When set, each
            opened reader also gets an in-process ``_AggregatorPump`` (zero-copy NVMM
            handles intact), and the sensor is recorded in ``SensorLeg.direct_sensors`` —
            forward that list to WorldState's ``direct_image_frame_sensors`` parameter.
        ros_node: Existing composed ROS node that owns image publishers. When omitted, each
            fallback publisher owns its own private node.
        uncapped_sensors: Camera names exempt from the fallback publisher's rate cap
            (``_MAX_FALLBACK_TOPIC_RATE_HZ``). ``runtime_node`` fills this from
            ``slam_camera_names``.
        topic_max_size: Optional ``(width, height)`` ceiling for the fallback topic;
            ``CameraInfo`` intrinsics are rescaled to match. Cameras in ``uncapped_sensors`` are
            exempt. ``runtime_node`` fills this from ``topic_frame_size``. ``None`` publishes
            at capture size.

    Returns:
        A ``SensorLeg`` holding open readers and prepared publishers. Call
        ``SensorLeg.start`` after the owning ROS lifecycle nodes are configured, then
        ``SensorLeg.close`` on shutdown.

    Raises:
        ROSConfigError: A binding names an unknown backend, or a backend's optional dependency
            (PyGObject / opencv-python) is missing.

    Example:
        >>> from openral_core import DeployScene, RobotDescription
        >>> desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")  # doctest: +SKIP
        >>> scene = DeployScene.from_yaml("scenes/deploy/so101_bench.yaml")  # doctest: +SKIP
        >>> leg = open_deploy_sensor_readers([*desc.sensors, *scene.sensors])  # doctest: +SKIP
        >>> leg.start()  # doctest: +SKIP
        >>> try:  # doctest: +SKIP
        ...     ...  # spin the graph
        ... finally:
        ...     leg.close()
    """
    # Deferred imports — openral_runner pulls torch-adjacent modules; keep
    # this module importable for AST/shape tests on minimal hosts.
    from openral_core import SensorReaderBackend, SensorReaderConfig
    from openral_runner.factory import make_sensor_readers

    leg = SensorLeg()
    prepared: list[tuple[SensorSpec, str, SensorReaderConfig, bool]] = []
    for spec in sensors:
        binding = spec.deploy_binding
        if binding is None:
            continue
        topic = f"{topic_prefix}/{spec.name}/image"
        native_tee = binding.backend == SensorReaderBackend.GSTREAMER
        prepared.append(
            (
                spec,
                topic,
                SensorReaderConfig(
                    sensor_id=spec.name,
                    backend=binding.backend,
                    backend_params=binding.backend_params,
                    max_age_ms=binding.max_age_ms,
                    publish_to_ros=native_tee,
                    publish_topic=topic if native_tee else None,
                    publish_rate_hz=_publish_rate_hz(spec) if native_tee else None,
                ),
                native_tee,
            )
        )
    readers = make_sensor_readers([cfg for _, _, cfg, _ in prepared])
    try:
        for (spec, topic, _cfg, native_tee), reader in zip(prepared, readers, strict=True):
            binding = spec.deploy_binding
            assert binding is not None  # reason: prepared filters unbound specs
            if native_tee:
                # Native in-pipeline tee: force it on so the frames reach ROS.
                reader.open()
                # Serialise cold-start: block until this camera streams a frame
                # before opening the next one (reopening on a transient bus
                # error). Two USB MJPG cameras whose ``nvjpegdec`` pipelines
                # negotiate PLAYING concurrently race — one loses with v4l2
                # ``streaming stopped (-5)`` and the reader latches it without
                # self-retry, so a co-mounted wrist+top pair came up one-camera-short.
                _await_first_frame(reader, spec.name)
                leg.readers.append(reader)
            else:
                # No native tee (opencv_thread): open the reader bare and
                # attach the polling ROS publisher pump.
                from openral_sensors.ros_publisher import SensorRosPublisher

                reader.open()
                leg.readers.append(reader)
                # Manifest-calibrated intrinsics → companion CameraInfo on the
                # same sibling topic layout the sim HAL uses, stamped with the
                # manifest's TF frame — this is what lets mono visual SLAM
                # (cuVSLAM rig build + nvblox depth framing) run on real
                # hardware. Specs without ``intrinsics`` publish images only.
                publisher = SensorRosPublisher(
                    reader=reader,
                    topic=topic,
                    rate_hz=_fallback_topic_rate_hz(spec, uncapped_sensors),
                    max_size=None if spec.name in uncapped_sensors else topic_max_size,
                    frame_id=spec.frame_id,
                    camera_info=spec.intrinsics,
                    info_topic=f"{topic_prefix}/{spec.name}/camera_info",
                    node=ros_node,
                )
                leg.publishers.append(publisher)
            if aggregator is not None:
                # Zero-copy vision path: in-process reader → aggregator, no ROS hop
                # for the policy leg (NVMM handles survive). The ROS tee above
                # keeps serving observability consumers.
                pump = _AggregatorPump(reader, spec.name, aggregator, _publish_rate_hz(spec))
                leg.publishers.append(pump)
                leg.direct_sensors.append(spec.name)
            log.info(
                "sensor_leg.camera_open",
                sensor_id=spec.name,
                backend=binding.backend.value,
                topic=topic,
                direct_to_aggregator=aggregator is not None,
            )
        # Prepare every ROS entity while execution is still single-threaded.
        # SensorLeg.start() runs only after the composed lifecycle nodes have
        # also finished creating their publishers, subscriptions, and action
        # servers.
        for publisher in leg.publishers:
            prepare = getattr(publisher, "prepare", None)
            if prepare is not None:
                prepare()
    except Exception:
        # Half-open leg → close what we already opened before re-raising;
        # a failed camera must not leak a v4l2 handle past the error.
        leg.close()
        raise
    return leg
