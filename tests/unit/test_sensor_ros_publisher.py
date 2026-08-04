"""Unit tests for :class:`openral_sensors.ros_publisher.SensorRosPublisher`.

Two test tiers:

* **Construction / validation** — no rclpy required. Asserts the
  publisher rejects bad inputs (non-absolute topic, non-positive rate)
  and stays unstarted until :meth:`start`.
* **Live publish/subscribe** — gated on rclpy via
  ``pytest.importorskip``. Drives a real
  :class:`SensorRosPublisher` against a fake-but-real
  :class:`SensorReader` (no MagicMock: a small in-memory reader that
  returns a precomputed :class:`SensorFrame`), then opens an rclpy
  subscriber in the same process and asserts the round-trip arrives.

Per CLAUDE.md §1.11 — fake doubles live at the **process / network
boundary** (rclpy is the network boundary here, and we use a real
in-process rclpy publisher + subscriber, not a mock). The reader
double is a real Python class implementing the
:class:`SensorReaderLike` Protocol, not a ``MagicMock``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from openral_core import FrameEncoding, SensorFrame

# ── In-process fake reader (satisfies SensorReaderLike) ──────────────────────


@dataclass
class _FakeReader:
    """Real ``SensorReader``-shaped object that yields a precomputed frame.

    NOT a mock: an explicit, named class implementing the
    :class:`SensorReaderLike` Protocol. The publisher's contract is
    duck-typed on ``open / close / read_latest / sensor_id / is_open``.
    """

    sensor_id: str
    frame: SensorFrame
    is_open: bool = True

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read_latest(self, max_age_ms: int | None = None) -> SensorFrame:
        return self.frame


def _make_frame(width: int = 16, height: int = 12) -> SensorFrame:
    """Build an RGB8 :class:`SensorFrame` with a known repeating pixel pattern.

    Tiles ``0..255`` to the exact byte count the geometry implies. The size is
    a parameter because a frame must be able to match a real manifest's
    intrinsics — ``_publish_camera_info`` scales fx/fy/cx/cy from the
    calibrated resolution to the published one, so a mismatched pair no longer
    round-trips unchanged.
    """
    n_bytes = width * height * 3
    pattern = bytes(range(256))
    payload = (pattern * (n_bytes // 256 + 1))[:n_bytes]
    return SensorFrame(
        sensor_id="wrist_rgb",
        stamp_monotonic_ns=time.monotonic_ns(),
        stamp_wall_ns=time.time_ns(),
        encoding=FrameEncoding.RGB8,
        width=width,
        height=height,
        channels=3,
        data=payload,
    )


# ── Construction / validation (no rclpy) ─────────────────────────────────────


def test_publisher_rejects_relative_topic() -> None:
    """Topic must start with '/' (ROS-absolute)."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    with pytest.raises(ValueError, match=r"topic must be absolute"):
        SensorRosPublisher(reader=reader, topic="not_absolute", rate_hz=30.0)


def test_publisher_rejects_non_positive_rate() -> None:
    """rate_hz must be > 0."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    with pytest.raises(ValueError, match=r"rate_hz must be > 0"):
        SensorRosPublisher(reader=reader, topic="/x", rate_hz=0.0)
    with pytest.raises(ValueError, match=r"rate_hz must be > 0"):
        SensorRosPublisher(reader=reader, topic="/x", rate_hz=-10.0)


def test_publisher_rejects_non_positive_qos_depth() -> None:
    """qos_depth must be > 0."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    with pytest.raises(ValueError, match=r"qos_depth must be > 0"):
        SensorRosPublisher(reader=reader, topic="/x", rate_hz=30.0, qos_depth=0)


def test_publisher_construction_does_not_touch_ros() -> None:
    """Constructor stays import-safe on hosts without rclpy."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    pub = SensorRosPublisher(reader=reader, topic="/cam/image_raw", rate_hz=30.0)
    assert pub.is_started is False
    assert pub.n_published == 0
    assert pub.topic == "/cam/image_raw"
    assert pub.info_topic == "/cam/image_raw/camera_info"


def test_publisher_accepts_external_node_without_owning_its_lifecycle() -> None:
    """Composed runtimes retain node ownership across publisher stop."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    node = object()
    pub = SensorRosPublisher(
        reader=reader,
        topic="/cam/image_raw",
        rate_hz=30.0,
        node=node,  # type: ignore[arg-type] # reason: constructor is ROS-I/O-free
    )

    assert pub._node is node
    assert pub._owns_node is False


def test_publisher_start_without_rclpy_raises_runtime_error() -> None:
    """When rclpy is genuinely absent, start() raises RuntimeError with install hint.

    Skipped when rclpy is actually present in this venv — the rclpy-
    present case is covered by the live test below.
    """
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("rclpy is installed; this test covers the rclpy-absent branch")

    from openral_sensors.ros_publisher import SensorRosPublisher

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    pub = SensorRosPublisher(reader=reader, topic="/cam/image_raw", rate_hz=30.0)
    with pytest.raises(RuntimeError, match=r"rclpy"):
        pub.start()


# ── Live publish/subscribe (rclpy-gated) ────────────────────────────────────


def _rclpy_available() -> bool:
    """True iff rclpy + sensor_msgs are importable in this venv."""
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / sensor_msgs not on PYTHONPATH; source a ROS 2 install to run live tests",
)
def test_publisher_round_trip_via_real_subscriber() -> None:
    """Start a publisher, subscribe in the same process, assert at least one frame arrives.

    Real rclpy publisher + real rclpy subscriber + real in-process
    reader. Mirrors the GStreamer ROS-tee live test pattern.
    """
    import rclpy
    from openral_sensors.ros_publisher import SensorRosPublisher
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image

    reader = _FakeReader(sensor_id="wrist_rgb", frame=_make_frame())
    pub = SensorRosPublisher(reader=reader, topic="/cam/image_raw", rate_hz=60.0)
    pub.start()
    try:
        # The publisher uses BEST_EFFORT QoS per CLAUDE.md §5.3 (sensor
        # streams); the subscriber MUST match or DDS reports a QoS
        # incompatibility and no messages flow.
        sub_node = rclpy.create_node("sensor_ros_publisher_test_subscriber")
        sub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        received: list[Image] = []
        sub_node.create_subscription(Image, "/cam/image_raw", received.append, sub_qos)

        # Spin both nodes for up to 2 s, exiting as soon as we get one frame.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            rclpy.spin_once(sub_node, timeout_sec=0.05)
        sub_node.destroy_node()

        assert received, "no Image messages received within 2 s"
        msg = received[0]
        assert msg.encoding == "rgb8"
        assert msg.width == reader.frame.width
        assert msg.height == reader.frame.height
        assert msg.header.frame_id == "wrist_rgb"
        # Pump thread should have published at least one frame.
        assert pub.n_published >= 1
    finally:
        pub.stop()


@pytest.mark.skipif(
    not _rclpy_available(),
    reason="rclpy / sensor_msgs not on PYTHONPATH; source a ROS 2 install to run live tests",
)
def test_publisher_camera_info_companion_round_trip() -> None:
    """Manifest intrinsics → companion ``CameraInfo`` on the OpenRAL sibling topic.

    This is exactly the surface the deploy sensor leg drives for real-hardware
    mono visual SLAM: ``camera_info=spec.intrinsics`` from a REAL robot manifest
    (so101_follower's calibrated wrist camera), ``frame_id=spec.frame_id`` (the
    TF frame cuVSLAM adopts as its rig), and the ``/openral/cameras/<name>/
    camera_info`` layout (NOT the ``<topic>/camera_info`` default) matching the
    sim HAL.
    """
    import rclpy
    from openral_core import RobotDescription
    from openral_sensors.ros_publisher import SensorRosPublisher
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo

    desc = RobotDescription.from_yaml("robots/so101_follower/robot.yaml")
    spec = next(s for s in desc.sensors if s.modality == "rgb" and s.intrinsics is not None)

    # Frame geometry MUST match the declared intrinsics, as a real camera's
    # does: `_publish_camera_info` scales fx/fy/cx/cy from the calibrated
    # resolution to the published one, so pairing 640x480 intrinsics with a
    # 16x12 frame would assert the very desync this publisher exists to avoid.
    assert spec.intrinsics is not None  # reason: filtered above; narrows for mypy
    reader = _FakeReader(
        sensor_id=spec.name,
        frame=_make_frame(width=spec.intrinsics.width, height=spec.intrinsics.height),
    )
    pub = SensorRosPublisher(
        reader=reader,
        topic=f"/openral/cameras/{spec.name}/image",
        rate_hz=60.0,
        frame_id=spec.frame_id,
        camera_info=spec.intrinsics,
        info_topic=f"/openral/cameras/{spec.name}/camera_info",
    )
    assert pub.info_topic == f"/openral/cameras/{spec.name}/camera_info"
    pub.start()
    try:
        sub_node = rclpy.create_node("sensor_ros_publisher_info_test_subscriber")
        # CameraInfo is RELIABLE (camera_info_manager convention).
        info_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        received: list[CameraInfo] = []
        sub_node.create_subscription(CameraInfo, pub.info_topic, received.append, info_qos)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not received:
            rclpy.spin_once(sub_node, timeout_sec=0.05)
        sub_node.destroy_node()

        assert received, "no CameraInfo messages received within 2 s"
        info = received[0]
        assert info.header.frame_id == spec.frame_id
        assert (info.width, info.height) == (spec.intrinsics.width, spec.intrinsics.height)
        k = list(info.k)
        assert k[0] == spec.intrinsics.fx
        assert k[4] == spec.intrinsics.fy
        assert k[2] == spec.intrinsics.cx
        assert k[5] == spec.intrinsics.cy
    finally:
        pub.stop()


# ── Resolution ceiling + lockstep intrinsics ────────────────────────────────
#
# `max_size` exists because every publish hands rclpy a full-resolution buffer
# whose Python->C conversion holds the GIL for the whole copy (~30 ms per
# 640x480 frame, 30 % of wall time even at the sensor leg's capped rate). The
# hazard it introduces is silent: resize the pixels without rescaling
# CameraInfo and every geometric consumer is quietly wrong.


def test_max_size_rejects_non_positive_dimensions() -> None:
    from openral_sensors.ros_publisher import SensorRosPublisher

    with pytest.raises(ValueError, match="max_size must be positive"):
        SensorRosPublisher(
            reader=_FakeReader(sensor_id="wrist_rgb", frame=_make_frame()),
            topic="/openral/cameras/wrist/image",
            rate_hz=3.0,
            max_size=(0, 240),
        )


def test_downscale_factor_only_ever_shrinks() -> None:
    """A frame already inside the ceiling is published untouched."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    pub = SensorRosPublisher(
        reader=_FakeReader(sensor_id="wrist_rgb", frame=_make_frame()),
        topic="/openral/cameras/wrist/image",
        rate_hz=3.0,
        max_size=(320, 240),
    )
    assert pub._downscale_factor(640, 480) == 0.5
    assert pub._downscale_factor(160, 120) == 1.0, "must never upscale"
    assert pub._downscale_factor(320, 240) == 1.0


def test_downscale_preserves_aspect_ratio_and_buffer_length() -> None:
    """A wrong-length buffer would corrupt every subscriber, not just slow it."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    pub = SensorRosPublisher(
        reader=_FakeReader(sensor_id="wrist_rgb", frame=_make_frame()),
        topic="/openral/cameras/wrist/image",
        rate_hz=3.0,
        max_size=(320, 240),
    )
    raw = bytes(640 * 480 * 3)
    scale = pub._downscale_factor(640, 480)
    data, w, h = pub._downscale(raw, 640, 480, 3, scale)
    assert (w, h) == (320, 240)
    assert w / h == 640 / 480, "aspect ratio changed — intrinsics would need x/y split"
    assert len(data) == w * h * 3, "buffer length must match the declared geometry"


def test_downscale_falls_back_on_a_mismatched_buffer() -> None:
    """Publishing full-res is merely slow; publishing a mis-sized buffer corrupts."""
    from openral_sensors.ros_publisher import SensorRosPublisher

    pub = SensorRosPublisher(
        reader=_FakeReader(sensor_id="wrist_rgb", frame=_make_frame()),
        topic="/openral/cameras/wrist/image",
        rate_hz=3.0,
        max_size=(320, 240),
    )
    truncated = bytes(1000)  # not 640*480*3
    data, w, h = pub._downscale(truncated, 640, 480, 3, 0.5)
    assert (w, h) == (640, 480)
    assert data == truncated


def test_camera_info_intrinsics_scale_with_a_downscaled_image() -> None:
    """fx/fy/cx/cy must track the published pixels, or geometry breaks silently.

    Guards the concrete hazard: `_publish_camera_info` historically took
    width/height from the frame but k/p verbatim from the manifest, so a
    resized image would ship full-resolution intrinsics against reduced
    dimensions with nothing in the graph to flag it.
    """
    from openral_core import IntrinsicsPinhole
    from openral_sensors.ros_publisher import SensorRosPublisher

    intr = IntrinsicsPinhole(width=640, height=480, fx=480.0, fy=480.0, cx=320.0, cy=240.0)
    pub = SensorRosPublisher(
        reader=_FakeReader(sensor_id="wrist_rgb", frame=_make_frame()),
        topic="/openral/cameras/wrist/image",
        rate_hz=3.0,
        camera_info=intr,
        max_size=(320, 240),
    )

    published: list[object] = []

    class _Sink:
        def publish(self, msg: object) -> None:
            published.append(msg)

    pub._info_publisher = _Sink()  # type: ignore[assignment]  # reason: duck-typed sink
    pub._publish_camera_info(320, 240, stamp=None)

    info = published[0]
    assert (info.width, info.height) == (320, 240)
    # Half the pixels → half the focal length and half the principal point.
    assert info.k[0] == pytest.approx(240.0), "fx did not scale with the image"
    assert info.k[4] == pytest.approx(240.0), "fy did not scale with the image"
    assert info.k[2] == pytest.approx(160.0), "cx did not scale with the image"
    assert info.k[5] == pytest.approx(120.0), "cy did not scale with the image"
    assert info.p[0] == pytest.approx(info.k[0]) and info.p[2] == pytest.approx(info.k[2])
