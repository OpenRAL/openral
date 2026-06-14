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
    """Build a small RGB8 :class:`SensorFrame` with a known pixel pattern."""
    payload = bytes(range(width * height * 3 % 256)) * (
        (width * height * 3) // (width * height * 3 % 256 or 1) + 1
    )
    payload = payload[: width * height * 3]
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
