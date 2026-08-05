"""Real-mode camera leg: open every deploy-bound sensor and publish to ROS.

``openral deploy run`` composes the same launch graph as ``deploy sim``,
but where the sim HAL publishes camera frames itself (``SimSensorBridge``
renders MuJoCo cameras onto ``/openral/cameras/<name>/image``), real
hardware has no camera publisher. The physical ``/dev/video*`` devices are
described by :attr:`~openral_core.SensorSpec.deploy_binding` — on the robot
manifest for robot-mounted cameras (wrist / head) and on
:attr:`~openral_core.DeployScene.sensors` for workcell-mounted ones
(overhead / front).

This module is that leg. :func:`open_deploy_sensor_readers` builds one
reader per bound spec via the runner's ``SENSOR_BACKEND_REGISTRY`` and
prepares every camera for the WorldState subscription topic
``<topic_prefix>/<name>/image``:

* ``gstreamer`` backend — the reader's built-in ROS tee publishes
  directly from the pipeline (``pipeline._build_ros_tee_branch``).
* ``opencv_thread`` (and any backend without a native tee) — the open
  reader is wrapped in a polling
  :class:`~openral_sensors.ros_publisher.SensorRosPublisher`. When the
  spec carries calibrated ``intrinsics``, a companion ``CameraInfo`` is
  published on ``<topic_prefix>/<name>/camera_info`` (the same layout the
  sim HAL uses) so mono visual SLAM works on real hardware.

Publishers use the CLAUDE.md §2 sensor-stream QoS (BEST_EFFORT); the
WorldState image subscription requests BEST_EFFORT so both match.

**Direct aggregator path (zero-copy vision path).** The reader, WorldState
aggregator, and skill runner share one OS process (``compose_runtime``), yet
frames historically took an intra-process ROS round trip (reader tee →
``sensor_msgs/Image`` → ``_on_image`` rebuilds a data-only ``SensorFrame``) —
which both re-serializes every pixel and destroys the zero-copy NVMM
``SensorFrame.handle``. When the caller passes the shared ``aggregator``,
an :class:`_AggregatorPump` per reader writes ``read_latest()`` frames
straight into ``WorldStateAggregator.update_image_frame`` — handles intact.
The pump also emits each frame's dashboard span, so pump-fed cameras own
their display path end to end at full reader cadence; WorldState's
``direct_image_frame_sensors`` parameter makes ``_on_image`` skip them
entirely rather than re-doing both at the tee's capped rate. The ROS tee
stays on for its remaining subscribers (detector, reward monitor, reasoner).

The caller starts the prepared pumps with :meth:`SensorLeg.start` only after
the composed ROS lifecycle nodes are configured, then owns teardown through
:meth:`SensorLeg.close`. ``runtime_node`` wires this in when its
``deploy_config`` parameter is set (real deploys only — sim keeps the HAL
bridge as its single camera source).
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

# Cap on the ROS topic cadence for the PYTHON FALLBACK publisher only.
#
# Each tick of that publisher hands a full-resolution `sensor_msgs/Image` to
# rclpy, whose Python->C message conversion holds the GIL for the whole
# 900 KiB copy. Profiled on the SO-101 bench with the reward monitor
# co-resident (so the topic has a real remote subscriber and Cyclone must
# actually serialise): `py-spy record --gil` put **89.5% of all GIL-holding
# samples in that one publish call**, against 0.3% for the thread loading the
# VLA — which is why a 13 s SmolVLA load stretched to 86 s while the process
# used a quarter of one core on a 22-core host.
#
# Nothing needs 30 Hz on this topic. The policy does NOT read it at all: on a
# real deploy `open_deploy_sensor_readers` pumps frames straight into the
# shared aggregator and registers them in WorldState's
# `direct_image_frame_sensors`, so `_on_image` skips them. Nor does the
# dashboard — `_emit_frame_observability` produces its thumbnails from the same
# pump, at full cadence. What is left on this topic is the reward monitor
# (~1 Hz), the object detector, and the reasoner's completion camera (0.2 Hz).
#
# The NATIVE GStreamer tee is deliberately NOT capped — it publishes from
# inside the pipeline without the Python conversion, so it does not pay this
# cost and downsampling it would only lose frames.
#
# 3 Hz, not 5: re-profiling after the first cap still put 52.75 % of GIL
# samples in `_publish_frame` (22.7 s of the 43 s the GIL was held over 75 s
# ≈ 30 % of wall time — ~30 ms per 640x480 publish). Once the dashboard moved
# to `_emit_frame_observability`, nothing left on this topic runs faster than
# the reward monitor's ~1 Hz. The floor is `WorldStateAggregator`'s
# `staleness_limit_s` (0.5 s): 2 Hz sits exactly on it and would flap the
# per-sensor diagnostics, so 3 Hz (0.33 s) is as low as this can safely go.
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

    WorldState's ``_on_image`` normally produces this span, but it is driven by
    the ROS tee — which :data:`_MAX_FALLBACK_TOPIC_RATE_HZ` caps at 3 Hz, so the
    dashboard's camera tiles would stutter at 3 fps. Pump-fed cameras emit here
    instead, at the reader's full cadence, and ``_on_image`` skips them.

    Affordable because the thumbnail is small and Pillow drops the GIL for the
    resize/encode: measured 2.42 ms/frame at 320x240 q60, and 60 thumbnails/s
    costs **4.5 %** of a competing thread's GIL time — against the **89.5 %**
    that the uncapped full-resolution image topic held. Display-only; this
    never touches the frame the policy reads. The flip/span/thumbnail
    pipeline itself is the shared
    :func:`openral_observability.producer.emit_sensor_frame_span`, so
    pump-fed and tee-fed (``_on_image``) cameras can never render
    differently.
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
        """Stash config; the polling thread starts in :meth:`start`."""
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
        """Poll ``read_latest`` at the configured rate; write new frames only.

        Deadline-paced (same shape as ``openral_sensors.ros_publisher``):
        the old ``stop_event.wait(period)``-then-work loop ran at
        work + period, so the thumbnail-encode cost pushed the effective
        pump rate below the camera rate and every read sampled a slightly
        staler frame.
        """
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
            # Way behind (reader/thumbnail blocked for periods) — re-anchor
            # instead of firing a catch-up burst.
            if time.monotonic() > next_deadline + self._period_s:
                next_deadline = time.monotonic() + self._period_s


@dataclass
class SensorLeg:
    """Open readers + prepared publishers for one deploy session.

    Attributes:
        readers: Open :class:`SensorReader` instances, one per deploy-bound
            :class:`SensorSpec` (gstreamer readers publish via their
            internal ROS tee).
        publishers: Prepared :class:`SensorRosPublisher` pumps for the
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
    """Robot-manifest sensors ∪ ``DeployScene.sensors``, scene wins on name collision.

    A scene entry named like a manifest sensor is that sensor's deploy-time
    binding — keeping both would double-open the device
    and publish the same topic twice.
    """
    scene = list(scene_sensors)
    scene_names = {s.name for s in scene}
    return [s for s in manifest_sensors if s.name not in scene_names] + scene


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

    cuVSLAM / PyCuVSLAM track frame-to-frame motion and lose the track on a
    starved stream, so these cameras keep their full cadence regardless of
    :data:`_MAX_FALLBACK_TOPIC_RATE_HZ`. Derived from the scene's
    ``DeployRuntime`` rather than left to a hand-written per-binding
    override: a stereo deploy that forgot the override would degrade
    silently, and silent degradation of a tracking input is exactly the
    failure this cap must not cause.

    Both spellings are covered: the explicit ``slam_stereo_cameras`` /
    ``slam_mono_camera`` names, and the implicit ``left``/``right`` pair
    that a ``None`` stereo field resolves to downstream.

    Args:
        runtime: A ``DeployRuntime`` (or ``None`` when the scene pins no
            runtime block). Duck-typed so this module keeps no schema import.

    Returns:
        Lower-cost-to-be-wrong-in-this-direction set of camera names. Empty
        when SLAM is off — the cap then applies to every camera.

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

    :func:`slam_camera_names` and :func:`topic_frame_size` decide the fallback
    topic's rate/resolution from ``DeployRuntime`` — but the scene YAML's
    ``enable_object_detector`` / ``enable_slam`` are tri-state (``None`` =
    "auto"), and the deploy CLI resolves the auto at launch time (detector
    defaults ON when a backend exists; SLAM auto-enables from the robot
    manifest). The runtime node re-reads the *original* YAML, so consuming the
    raw block would treat an auto-enabled leg as OFF and silently starve the
    exact cameras the detector/SLAM nodes subscribe to.

    Each ``None`` override keeps the scene's value (bare ``runtime_node`` runs
    without the launch parameters behave as before); a launch that performs its
    flags always passes them, and those values gate the actual detector/SLAM
    nodes — the ground truth of which subscribers exist in the graph.

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

    Derived from the scene's ``DeployRuntime`` at launch — deliberately NOT
    per-rSkill. The reasoner picks skills at runtime, so a size chosen for the
    active skill would be wrong the moment it switched; and the topic's
    subscribers (detector, reward monitor, reasoner camera) are fixed by launch
    flags, not by which policy is loaded. The policy never reads this topic.

    Returns ``None`` whenever a subscriber needs full pixels:

    * **object detector** — its node declares ``input_size`` 640, and which
      camera it watches is a runtime detail, so the whole topic stays native.
    * **visual SLAM** — cuVSLAM triangulates against calibrated intrinsics;
      per-camera exemption is handled alongside the rate cap via
      :func:`slam_camera_names`, but a SLAM scene keeps every camera native
      rather than betting on the name list being complete.

    Args:
        runtime: A ``DeployRuntime`` (or ``None`` when the scene pins no
            runtime block). Duck-typed so this module keeps no schema import.

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

    The capture rate and the topic rate are different things. Readers keep
    running at the camera's full fps (the policy reads the freshest frame
    in-process), but republishing every one of those frames through rclpy
    costs a GIL-held 900 KiB conversion per camera per tick — see
    :data:`_MAX_FALLBACK_TOPIC_RATE_HZ`. A scene asking for a *slower* rate
    than the cap is honoured as-is; the cap only ever lowers.

    Two ways out of the cap, in precedence order:

    1. ``uncapped`` — camera names that must keep full cadence. The runtime
       fills this from :func:`slam_camera_names`, so visual SLAM is exempt
       **automatically**; nobody has to remember a per-binding flag.
    2. ``backend_params["topic_rate_hz"]`` — a binding demanding an exact
       cadence, for any other rate-sensitive out-of-process consumer.

    Args:
        spec: A sensor spec carrying a ``deploy_binding``.
        uncapped: Camera names exempt from the cap (see
            :func:`slam_camera_names`).

    Returns:
        The full configured rate when the sensor is exempt, else
        ``backend_params["topic_rate_hz"]`` when set and positive, else
        ``min(configured rate, _MAX_FALLBACK_TOPIC_RATE_HZ)``.
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

    Serialises camera cold-start (call between consecutive ``reader.open()``s):
    two USB MJPG cameras whose ``nvjpegdec`` pipelines negotiate PLAYING at the
    same instant can race, and the loser latches a v4l2 ``streaming stopped
    (-5)`` bus error with no self-retry. Waiting for a frame before opening the
    next camera staggers negotiation; a bounded close+reopen recovers a camera
    that lost a transient race (by then the other camera already streams, so the
    reopen runs unopposed). Fast no-op once frames flow (tens of ms).

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
        sensors: Robot-manifest sensors plus :attr:`DeployScene.sensors`
            (the caller concatenates). Specs without a
            :attr:`~openral_core.SensorSpec.deploy_binding` are skipped —
            committed reference manifests leave the binding unset.
        topic_prefix: WorldState's ``camera_topic_prefix``. The final
            topic is ``<topic_prefix>/<spec.name>/image``.
        aggregator: The composed runtime's shared ``WorldStateAggregator``.
            When set, every opened reader also gets an in-process
            :class:`_AggregatorPump` writing frames (zero-copy NVMM handles
            intact) straight into it, and the sensor is recorded in
            :attr:`SensorLeg.direct_sensors` — forward that list to
            WorldState's ``direct_image_frame_sensors`` parameter.
        ros_node: Existing composed ROS node that owns image publishers. When
            omitted, each fallback publisher owns its historical private node.
        uncapped_sensors: Camera names exempt from the fallback publisher's
            rate cap (:data:`_MAX_FALLBACK_TOPIC_RATE_HZ`). ``runtime_node``
            fills this from :func:`slam_camera_names` so visual SLAM keeps
            full cadence without anyone hand-setting a per-binding flag.
        topic_max_size: Optional ``(width, height)`` ceiling for the fallback
            topic; ``CameraInfo`` intrinsics are rescaled to match. Cameras in
            ``uncapped_sensors`` are exempt — visual SLAM needs native pixels
            as well as native cadence. ``runtime_node`` fills this from
            :func:`topic_frame_size`. ``None`` publishes at capture size.

    Returns:
        A :class:`SensorLeg` holding open readers and prepared publishers.
        Call :meth:`SensorLeg.start` after the owning ROS lifecycle nodes are
        configured, then :meth:`SensorLeg.close` on shutdown.

    Raises:
        ROSConfigError: A binding names an unknown backend, or a backend's
            optional dependency (PyGObject / opencv-python) is missing.

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
