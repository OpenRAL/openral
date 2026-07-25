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
The ROS tee stays on for observability (dashboard thumbnails, detectors);
WorldState's ``direct_image_frame_sensors`` parameter stops ``_on_image``
from double-writing those sensors into the aggregator.

The caller starts the prepared pumps with :meth:`SensorLeg.start` only after
the composed ROS lifecycle nodes are configured, then owns teardown through
:meth:`SensorLeg.close`. ``runtime_node`` wires this in when its
``deploy_config`` parameter is set (real deploys only — sim keeps the HAL
bridge as its single camera source).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openral_core import SensorSpec

__all__ = ["SensorLeg", "merge_deploy_sensors", "open_deploy_sensor_readers"]

log = structlog.get_logger(__name__)

#: WorldState's camera subscription prefix (`<prefix>/<name>/image`).
DEFAULT_TOPIC_PREFIX = "/openral/cameras"

#: Publish cadence when the binding's backend_params carry no fps.
#: Matches the WorldStateAggregator staleness-gate expectation (10 Hz cameras).
_DEFAULT_PUBLISH_RATE_HZ = 10.0


class _Closeable(Protocol):
    """Structural type for anything with a no-arg close/stop."""

    def close(self) -> None: ...  # pragma: no cover — Protocol


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
        while not self._stop_event.wait(self._period_s):
            try:
                frame = self._reader.read_latest(max_age_ms=None)
            except Exception:  # reason: no frame yet / transient staleness — keep polling
                continue
            if frame.stamp_monotonic_ns == self._last_stamp_ns:
                continue
            self._last_stamp_ns = frame.stamp_monotonic_ns
            self._aggregator.update_image_frame(self._sensor_name, frame)


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
    #: WorldState's ``direct_image_frame_sensors`` parameter must list these
    #: so ``_on_image`` doesn't double-write them from the ROS tee.
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
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY

    leg = SensorLeg()
    try:
        for spec in sensors:
            binding = spec.deploy_binding
            if binding is None:
                continue
            topic = f"{topic_prefix}/{spec.name}/image"
            if binding.backend == SensorReaderBackend.GSTREAMER:
                # Native in-pipeline tee: force it on so the frames reach ROS.
                cfg = SensorReaderConfig(
                    sensor_id=spec.name,
                    backend=binding.backend,
                    backend_params=binding.backend_params,
                    max_age_ms=binding.max_age_ms,
                    publish_to_ros=True,
                    publish_topic=topic,
                    publish_rate_hz=_publish_rate_hz(spec),
                )
                reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
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

                cfg = SensorReaderConfig(
                    sensor_id=spec.name,
                    backend=binding.backend,
                    backend_params=binding.backend_params,
                    max_age_ms=binding.max_age_ms,
                )
                reader = SENSOR_BACKEND_REGISTRY[cfg.backend.value](cfg)
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
                    rate_hz=_publish_rate_hz(spec),
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
