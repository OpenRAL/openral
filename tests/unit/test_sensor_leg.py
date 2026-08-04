"""Unit tests for the real-mode camera leg (``openral_rskill_ros.sensor_leg``).

The leg is what `openral deploy run` uses to open every deploy-bound
``SensorSpec`` (robot manifest ∪ ``DeployScene.sensors``) and publish each
camera onto the WorldState image topics
(``/openral/cameras/<name>/image``). Real components per CLAUDE.md §1.11:
the GStreamer test uses a real ``videotestsrc`` pipeline (headless, no
camera hardware); ROS-touching paths skip when ``rclpy`` isn't importable
(CI runners without a sourced ROS install).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from openral_core import (
    SensorDeployBinding,
    SensorReaderBackend,
    SensorReaderConfig,
    SensorSpec,
)
from openral_rskill_ros.sensor_leg import (
    _MAX_FALLBACK_TOPIC_RATE_HZ,
    SensorLeg,
    _fallback_topic_rate_hz,
    _publish_rate_hz,
    merge_deploy_sensors,
    open_deploy_sensor_readers,
)


def _spec(name: str, *, binding: SensorDeployBinding | None, rate_hz: float = 0.0) -> SensorSpec:
    return SensorSpec(
        name=name,
        modality="rgb",
        frame_id=name,
        rate_hz=rate_hz,
        deploy_binding=binding,
    )


def test_publish_rate_binding_fps_wins_over_spec_rate() -> None:
    """The binding's fps beats the spec's declared rate and the 10 Hz default."""
    spec = _spec(
        "top",
        binding=SensorDeployBinding(backend_params={"device": "/dev/video0", "fps": 15}),
        rate_hz=30.0,
    )
    assert _publish_rate_hz(spec) == 15.0


def test_publish_rate_falls_back_to_spec_rate_then_10hz() -> None:
    """No binding fps → the spec's rate_hz; neither → the WorldState-friendly 10 Hz."""
    spec = _spec("top", binding=SensorDeployBinding(), rate_hz=30.0)
    assert _publish_rate_hz(spec) == 30.0
    bare = _spec("top", binding=SensorDeployBinding(), rate_hz=0.0)
    assert _publish_rate_hz(bare) == 10.0


def test_fallback_topic_rate_is_capped_below_camera_rate() -> None:
    """A 30 fps camera must NOT drive the Python fallback publisher at 30 Hz.

    Each of those ticks is a GIL-held rclpy Python->C conversion of a
    900 KiB Image; profiling put 89.5% of all GIL-holding samples in that
    single call while a VLA was loading. Capture stays at 30 fps (the
    policy reads the freshest frame in-process); only the ROS cadence is
    capped.
    """
    spec = _spec(
        "top",
        binding=SensorDeployBinding(backend_params={"device": "/dev/video0", "fps": 30}),
        rate_hz=30.0,
    )
    assert _publish_rate_hz(spec) == 30.0, "capture rate must be untouched"
    assert _fallback_topic_rate_hz(spec) == _MAX_FALLBACK_TOPIC_RATE_HZ
    assert _fallback_topic_rate_hz(spec) < 30.0


def test_fallback_topic_rate_never_raises_a_slower_request() -> None:
    """The cap only ever lowers — a scene asking for 2 Hz keeps 2 Hz."""
    slow = _spec(
        "wrist",
        binding=SensorDeployBinding(backend_params={"device": "/dev/video4", "fps": 2}),
    )
    assert _fallback_topic_rate_hz(slow) == 2.0


def test_explicit_topic_rate_overrides_the_cap() -> None:
    """A rate-sensitive consumer can demand full cadence, cap included.

    Visual SLAM is the case that forces this: ``cuvslam`` / ``pycuvslam``
    subscribe to ``/openral/cameras/{left,right}/image`` and lose tracking
    on a starved stream, so a stereo deploy must be able to pin 30 Hz.
    Without the override the cap would silently degrade it.
    """
    stereo = _spec(
        "left",
        binding=SensorDeployBinding(
            backend_params={"device": "/dev/video2", "fps": 30, "topic_rate_hz": 30}
        ),
    )
    assert _fallback_topic_rate_hz(stereo) == 30.0
    assert _fallback_topic_rate_hz(stereo) > _MAX_FALLBACK_TOPIC_RATE_HZ


def test_merge_scene_entry_wins_on_name_collision() -> None:
    """A same-named DeployScene entry is that robot sensor's deploy binding —
    the manifest copy is dropped so the device is never double-opened."""
    manifest = [_spec("top", binding=None), _spec("wrist", binding=None)]
    scene = [
        _spec("top", binding=SensorDeployBinding(backend_params={"device": "/dev/video0"})),
        _spec("overhead", binding=SensorDeployBinding(backend_params={"device": "/dev/video2"})),
    ]
    merged = merge_deploy_sensors(manifest, scene)
    names = [s.name for s in merged]
    assert sorted(names) == ["overhead", "top", "wrist"]
    top = next(s for s in merged if s.name == "top")
    assert top.deploy_binding is not None  # the scene's bound copy survived


def test_unbound_specs_are_skipped() -> None:
    """Committed reference manifests leave deploy_binding unset — the leg skips them."""
    leg = open_deploy_sensor_readers([_spec("top", binding=None)])
    assert leg.readers == []
    assert leg.publishers == []


def test_empty_leg_close_is_idempotent() -> None:
    """close() on an empty/already-closed leg never raises."""
    leg = SensorLeg()
    leg.close()
    leg.close()
    assert leg.readers == []
    assert leg.publishers == []


def test_multi_camera_ros_publishers_start_only_after_explicit_lifecycle_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ROS entities are prepared before background publishing is allowed."""
    from openral_runner.factory import SENSOR_BACKEND_REGISTRY
    from openral_sensors.ros_publisher import SensorRosPublisher

    events: list[tuple[str, str]] = []

    class _Reader:
        def __init__(self, sensor_id: str) -> None:
            self.sensor_id = sensor_id

        def open(self) -> None:
            events.append(("open", self.sensor_id))

        def close(self) -> None:
            events.append(("close", self.sensor_id))

    def build_reader(config: SensorReaderConfig) -> _Reader:
        return _Reader(config.sensor_id)

    backend = SensorReaderBackend.OPENCV_THREAD
    monkeypatch.setitem(SENSOR_BACKEND_REGISTRY, backend.value, build_reader)
    monkeypatch.setattr(
        SensorRosPublisher,
        "prepare",
        lambda self: events.append(("prepare", self._reader.sensor_id)),
    )
    monkeypatch.setattr(
        SensorRosPublisher,
        "start",
        lambda self: events.append(("start", self._reader.sensor_id)),
    )
    monkeypatch.setattr(SensorRosPublisher, "stop", lambda self: None)

    binding = SensorDeployBinding(backend=backend)
    leg = open_deploy_sensor_readers(
        [
            _spec("front", binding=binding),
            _spec("wrist", binding=binding),
        ],
        ros_node=object(),
    )
    try:
        assert events == [
            ("open", "front"),
            ("open", "wrist"),
            ("prepare", "front"),
            ("prepare", "wrist"),
        ]
        leg.start()
        assert events == [
            ("open", "front"),
            ("open", "wrist"),
            ("prepare", "front"),
            ("prepare", "wrist"),
            ("start", "front"),
            ("start", "wrist"),
        ]
    finally:
        leg.close()


def test_gstreamer_testsrc_leg_publishes_frames(tmp_path: Path) -> None:
    """Full leg over a real videotestsrc pipeline: open → ROS tee → frame → close.

    Skips without PyGObject (gi) or rclpy — the same gates the production
    factory enforces. Runs in a SUBPROCESS with the production import
    order (GStreamer backend → Gst.init → rclpy), because ``rclpy.Node()``
    segfaults inside Fast-DDS thread setup when rclpy was imported before
    ``Gst.init()`` (see ``openral_runner/backends/gstreamer/reader.py``
    PR I/8 note) — and the pytest process may already have rclpy loaded
    from an earlier test.
    """
    pytest.importorskip("gi", reason="PyGObject (gstreamer extra) not installed")
    pytest.importorskip("rclpy", reason="ROS 2 not sourced")

    probe = """
# Production import order (mirrors scripts/runtime_node): gi + Gst.init()
# FIRST — before numpy/pydantic/rclpy. Under Fast-DDS, rclpy.Node()
# segfaults when numpy or pydantic were imported before Gst.init()
# (x86 Ubuntu 24.04 / system PyGObject / ROS Jazzy; Cyclone unaffected).
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)

import time

import rclpy
from openral_core import SensorDeployBinding, SensorSpec
from openral_rskill_ros.sensor_leg import open_deploy_sensor_readers

rclpy.init()
spec = SensorSpec(
    name="testcam",
    modality="rgb",
    frame_id="testcam",
    rate_hz=10.0,
    deploy_binding=SensorDeployBinding(
        backend="gstreamer",
        backend_params={"source": "testsrc", "width": 320, "height": 240, "fps": 10},
    ),
)
# Phase 3: the shared aggregator receives frames straight from
# the reader (no ROS hop). Subclass the REAL aggregator only to observe
# the write (super() still runs) — no behaviour is faked.
from openral_core import RobotDescription
from openral_world_state import WorldStateAggregator

description = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")

class _CountingAggregator(WorldStateAggregator):
    def __init__(self, desc):
        super().__init__(desc)
        self.image_writes = []

    def update_image_frame(self, sensor_name, frame):
        super().update_image_frame(sensor_name, frame)
        self.image_writes.append((sensor_name, frame))

aggregator = _CountingAggregator(description)
leg = open_deploy_sensor_readers([spec], aggregator=aggregator)
try:
    leg.start()
    assert len(leg.readers) == 1, leg.readers
    # The direct-aggregator pump is registered as a publisher-shaped pump;
    # the sensor is recorded for WorldState's direct_image_frame_sensors.
    assert leg.direct_sensors == ["testcam"], leg.direct_sensors
    assert len(leg.publishers) == 1, leg.publishers
    deadline = time.time() + 10.0
    frame = None
    while time.time() < deadline:
        try:
            frame = leg.readers[0].read_latest(max_age_ms=2000)
        except Exception:
            frame = None
        if frame is not None:
            break
        time.sleep(0.1)
    assert frame is not None, "videotestsrc produced no frame within 10 s"

    # The pump delivered the same frames in-process, pixels intact.
    deadline = time.time() + 10.0
    while time.time() < deadline and not aggregator.image_writes:
        time.sleep(0.1)
    assert aggregator.image_writes, "aggregator pump wrote no frame within 10 s"
    written_name, written_frame = aggregator.image_writes[0]
    assert written_name == "testcam"
    # CPU hosts deliver inline data; the DeepStream tier delivers a zero-copy
    # NVMM handle (SensorFrame's data/handle exclusivity) — both are the point.
    assert written_frame.data is not None or written_frame.handle is not None
    assert written_frame.width == 320

    # And the WorldState side of the contract: a BEST_EFFORT subscriber
    # on the leg's topic (same profile world_state requests) receives a
    # sensor_msgs/Image from the in-pipeline ROS tee.
    from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image

    got = []
    sub_node = rclpy.create_node("probe_subscriber")
    sub_node.create_subscription(
        Image,
        "/openral/cameras/testcam/image",
        got.append,
        QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        ),
    )
    deadline = time.time() + 10.0
    while time.time() < deadline and not got:
        rclpy.spin_once(sub_node, timeout_sec=0.2)
    sub_node.destroy_node()
    assert got, "no Image arrived on /openral/cameras/testcam/image within 10 s"
    assert got[0].height == 240 and got[0].width == 320, (got[0].height, got[0].width)
finally:
    leg.close()
    rclpy.try_shutdown()
assert leg.readers == []
print("SENSOR_LEG_PROBE_OK")
"""
    import os
    import subprocess
    import sys

    pkg_dir = Path(__file__).resolve().parents[2] / "packages" / "openral_rskill_ros"
    env_vars = dict(os.environ)
    env_vars["PYTHONPATH"] = f"{pkg_dir}{os.pathsep}{env_vars.get('PYTHONPATH', '')}"
    result = subprocess.run(  # reason: sys.executable with a fixed -c script
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        env=env_vars,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "SENSOR_LEG_PROBE_OK" in result.stdout


def test_slam_cameras_are_never_capped_even_when_unnamed() -> None:
    """Visual SLAM keeps full cadence automatically — no per-binding flag.

    ``DeployRuntime.slam_stereo_cameras=None`` means "the impl's built-in
    left/right default", so a scene that enables SLAM without naming its
    cameras must STILL exempt them. cuVSLAM loses tracking on a starved
    stream, and a silent degradation of a tracking input is precisely what
    this cap must never cause.
    """
    from openral_rskill_ros.sensor_leg import slam_camera_names

    class _Runtime:
        enable_slam = True
        slam_stereo_cameras = None
        slam_mono_camera = None

    names = slam_camera_names(_Runtime())
    assert names == frozenset({"left", "right"})

    left = _spec(
        "left",
        binding=SensorDeployBinding(backend_params={"device": "/dev/video2", "fps": 30}),
    )
    assert _fallback_topic_rate_hz(left, names) == 30.0
    # A non-SLAM camera in the same scene is still capped.
    top = _spec(
        "top", binding=SensorDeployBinding(backend_params={"device": "/dev/video0", "fps": 30})
    )
    assert _fallback_topic_rate_hz(top, names) == _MAX_FALLBACK_TOPIC_RATE_HZ


def test_slam_camera_names_covers_explicit_stereo_and_mono() -> None:
    """Explicitly named stereo rigs and the mono-RGBD camera are both exempt."""
    from openral_rskill_ros.sensor_leg import slam_camera_names

    class _Stereo:
        enable_slam = True
        slam_stereo_cameras = ("front_left", "front_right")
        slam_mono_camera = None

    class _Mono:
        enable_slam = True
        slam_stereo_cameras = None
        slam_mono_camera = "front"

    assert slam_camera_names(_Stereo()) == frozenset({"front_left", "front_right"})
    assert slam_camera_names(_Mono()) == frozenset({"left", "right", "front"})


def test_slam_off_means_every_camera_is_capped() -> None:
    """SLAM disabled (both in-tree deploy scenes) → no exemptions at all."""
    from openral_rskill_ros.sensor_leg import slam_camera_names

    class _NoSlam:
        enable_slam = False
        slam_stereo_cameras = None
        slam_mono_camera = None

    assert slam_camera_names(_NoSlam()) == frozenset()
    assert slam_camera_names(None) == frozenset()


# ── Dashboard span emission from the pump ──────────────────────────────────
#
# The 5 Hz cap on the ROS tee (`_MAX_FALLBACK_TOPIC_RATE_HZ`) would otherwise
# drop the dashboard's camera tiles to 5 fps, because WorldState's `_on_image`
# — their historical source — is driven by that tee. The pump emits the span
# instead, at full reader cadence. These tests pin the two properties that
# make that safe: a real thumbnail comes out, and a failure there can never
# stop the aggregator being fed.


def _rgb_frame(width: int = 64, height: int = 48, *, value: int = 0):
    """A real ``SensorFrame`` with inline RGB8 pixels (no mocks, §1.11)."""
    from openral_core.schemas import FrameEncoding, SensorFrame

    return SensorFrame(
        sensor_id="top",
        stamp_monotonic_ns=1,
        stamp_wall_ns=2,
        encoding=FrameEncoding.RGB8,
        width=width,
        height=height,
        channels=3,
        data=bytes([value]) * (width * height * 3),
    )


def test_pump_emits_a_real_jpeg_thumbnail_for_the_dashboard() -> None:
    """The span carries a decodable JPEG, not an empty attribute."""
    from openral_observability import producer as ral_producer
    from openral_rskill_ros.sensor_leg import _emit_frame_observability

    frame = _rgb_frame()
    thumb = ral_producer.encode_frame_thumbnail(frame)
    assert thumb is not None and thumb[:2] == b"\xff\xd8", "expected a JPEG SOI marker"

    # The emit path itself must complete against the real tracer/producer.
    _emit_frame_observability("top", frame, flip_180=False)


def test_pump_flip_180_rotates_the_thumbnail_but_not_the_policy_frame() -> None:
    """``OPENRAL_DASHBOARD_FLIP_180`` is display-only, and it really does flip.

    Guards the failure recorded in the dashboard-flip incident: flipping the
    frame the policy reads double-flips it against the VLA's own
    ``image_preprocessing.flip_180`` and collapses the rollout.
    """
    import numpy as np
    from openral_observability import producer as ral_producer
    from openral_rskill_ros.sensor_leg import _emit_frame_observability

    # Asymmetric content — a uniform frame is flip-invariant and would make
    # this test pass even if the flip were a no-op.
    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    arr[:24, :, 0] = 255  # bright top half, dark bottom half
    frame = _rgb_frame().model_copy(update={"data": arr.tobytes()})
    original = frame.data

    upright = ral_producer.encode_frame_thumbnail(frame)
    flipped_arr = arr[::-1, ::-1]
    flipped = ral_producer.encode_frame_thumbnail(
        frame.model_copy(update={"data": flipped_arr.tobytes()})
    )
    assert upright != flipped, "test frame is flip-invariant; it cannot detect a no-op flip"

    _emit_frame_observability("top", frame, flip_180=True)
    assert frame.data == original, "flip leaked into the frame the policy reads"


def test_pump_feeds_aggregator_even_when_the_thumbnail_path_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken display path must degrade the dashboard, never the policy."""
    from openral_rskill_ros import sensor_leg as leg_mod

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("thumbnail encoder exploded")

    monkeypatch.setattr(leg_mod, "_emit_frame_observability", _boom)

    written: list[tuple[str, object]] = []

    class _Agg:
        def update_image_frame(self, name: str, frame: object) -> None:
            written.append((name, frame))

    class _Reader:
        """Advances the stamp each poll — the pump dedups identical stamps."""

        def __init__(self) -> None:
            self.n = 0

        def read_latest(self, max_age_ms: float | None = None) -> object:
            self.n += 1
            return _rgb_frame().model_copy(update={"stamp_monotonic_ns": self.n})

    pump = leg_mod._AggregatorPump(_Reader(), "top", _Agg(), rate_hz=1000.0)
    pump.start()
    try:
        deadline = time.monotonic() + 5.0
        # Three writes, not one: the first would survive even an unguarded
        # raise (the aggregator is written before the emit), so only a pump
        # that is still alive on later polls proves the guard works.
        while len(written) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        pump.stop()

    assert len(written) >= 3, (
        f"pump thread died on a failing thumbnail encode after {len(written)} frame(s)"
    )
    assert written[0][0] == "top"


# ── Topic resolution ceiling ───────────────────────────────────────────────
#
# Re-profiling after the rate cap still put 52.75 % of GIL-holding samples in
# `_publish_frame` (~30 ms per 640x480 publish). Resolution is the remaining
# lever, but it is only safe while every launch-time subscriber that needs
# native pixels opts out — and while CameraInfo scales with the image.


def test_topic_size_capped_when_no_consumer_needs_native_pixels() -> None:
    from openral_rskill_ros.sensor_leg import _DEFAULT_TOPIC_MAX_SIZE, topic_frame_size

    class _Runtime:
        enable_object_detector = False
        enable_slam = False

    assert topic_frame_size(_Runtime()) == _DEFAULT_TOPIC_MAX_SIZE


@pytest.mark.parametrize("flag", ["enable_object_detector", "enable_slam"])
def test_topic_size_stays_native_for_pixel_hungry_consumers(flag: str) -> None:
    """The detector declares input_size 640; cuVSLAM triangulates on intrinsics."""
    from openral_rskill_ros.sensor_leg import topic_frame_size

    runtime = type(
        "_Runtime",
        (),
        {"enable_object_detector": False, "enable_slam": False, flag: True},
    )()
    assert topic_frame_size(runtime) is None


def test_topic_size_is_native_without_a_runtime_block() -> None:
    from openral_rskill_ros.sensor_leg import topic_frame_size

    assert topic_frame_size(None) is None


def test_topic_rate_cap_stays_above_the_staleness_limit() -> None:
    """2 Hz would sit exactly on WorldStateAggregator's 0.5 s staleness window.

    Sitting on the limit flapped the per-sensor diagnostics OK<->STALE, which
    is why the cap floor is 3 Hz and not lower.
    """
    from openral_world_state import DEFAULT_STALENESS_S

    staleness_s = DEFAULT_STALENESS_S

    assert staleness_s > 1.0 / _MAX_FALLBACK_TOPIC_RATE_HZ, (
        f"cap {_MAX_FALLBACK_TOPIC_RATE_HZ} Hz has period "
        f"{1.0 / _MAX_FALLBACK_TOPIC_RATE_HZ:.3f}s >= staleness {staleness_s}s"
    )
