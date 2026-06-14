"""Unit tests for the GStreamer ROS-tee image publisher (ADR-0010 PR I/4).

Exercises the real :class:`RosImagePublisher` against a real GStreamer
pipeline + a real rclpy subscriber in the same process. No mocks.

Skips wholesale when either PyGObject or rclpy is unavailable.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

gi = pytest.importorskip(
    "gi", reason="PyGObject not installed (pip install openral-runner[gstreamer])"
)
gi.require_version("Gst", "1.0")
rclpy = pytest.importorskip(
    "rclpy",
    reason="rclpy not on PYTHONPATH — source a ROS 2 install before running ROS-tee tests",
)
sensor_msgs = pytest.importorskip("sensor_msgs.msg", reason="sensor_msgs not available")

# End-to-end tests below initialise rclpy + a DDS rmw implementation in the
# same Python process. On hosts where torch / pyarrow have already linked
# their own glib (the conftest pulls them in transitively), the rclpy DDS
# load conflicts with the already-loaded gi glib and the process aborts.
# Gate the live publish/subscribe round-trip behind an opt-in env var; the
# pure-construction + factory tests above stay on by default. The Docker
# image runs without the torch-pyarrow stack at import time and exercises
# the live path natively.
_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) to exercise"
)

# Imports kept below the gates so the module skips cleanly when ROS is absent.
from gi.repository import Gst  # noqa: E402
from openral_core import SensorReaderConfig  # noqa: E402
from openral_core.exceptions import ROSConfigError  # noqa: E402
from openral_runner.backends.gstreamer import (  # noqa: E402
    GStreamerSensorReader,
    PipelineSpec,
    Platform,
    Source,
)
from openral_runner.backends.gstreamer.ros_tee import RosImagePublisher  # noqa: E402
from openral_runner.factory import _make_gstreamer_reader  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

# ── RosImagePublisher pure-construction validation ───────────────────────────


def test_ros_publisher_rejects_relative_topic() -> None:
    """Topics must be absolute (start with '/')."""
    Gst.init(None)
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=1 ! videoconvert ! "
        "video/x-raw,format=BGR,width=80,height=60 ! appsink name=ros_sink"
    )
    appsink = pipeline.get_by_name("ros_sink")
    try:
        with pytest.raises(ValueError, match="topic must be absolute"):
            RosImagePublisher(sensor_id="x", appsink=appsink, topic="bad_relative")
    finally:
        pipeline.set_state(Gst.State.NULL)


def test_ros_publisher_rejects_non_positive_rate() -> None:
    """rate_hz must be > 0 or None."""
    Gst.init(None)
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=1 ! videoconvert ! "
        "video/x-raw,format=BGR,width=80,height=60 ! appsink name=ros_sink"
    )
    appsink = pipeline.get_by_name("ros_sink")
    try:
        with pytest.raises(ValueError, match="rate_hz must be"):
            RosImagePublisher(sensor_id="x", appsink=appsink, topic="/test/img", rate_hz=0)
    finally:
        pipeline.set_state(Gst.State.NULL)


# ── End-to-end with a real subscriber ────────────────────────────────────────


def _subscribe_once_and_wait(
    topic: str,
    node_name: str,
    received: list[Image],
    received_event: threading.Event,
    spin_seconds: float,
) -> None:
    """Helper: spin a rclpy subscriber for ``spin_seconds`` and collect Images."""
    # rclpy.init may already be active if a prior test in this module ran it.
    if not rclpy.ok():
        rclpy.init()
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
    node = rclpy.create_node(node_name)

    def callback(msg: Image) -> None:
        received.append(msg)
        received_event.set()

    node.create_subscription(Image, topic, callback, qos)
    deadline = time.monotonic() + spin_seconds
    while time.monotonic() < deadline and not received_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_reader_ros_tee_publishes_image_messages() -> None:
    """End-to-end: reader's ROS tee publishes real sensor_msgs/Image on the topic."""
    topic = "/bh_test/ros_tee/image_raw"
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=160,
        height=120,
        fps=30,
        enable_nvmm=False,
        enable_ros_tee=True,
    )
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        spec=spec,
        platform=Platform.CPU_ONLY,
        ros_topic=topic,
        ros_rate_hz=10.0,
        default_max_age_ms=500,
    )
    received: list[Image] = []
    received_event = threading.Event()
    subscriber = threading.Thread(
        target=_subscribe_once_and_wait,
        args=(topic, "bh_test_subscriber", received, received_event, 4.0),
        daemon=True,
    )
    with reader:
        subscriber.start()
        # Give the subscriber a moment to discover the publisher.
        subscriber.join(timeout=5.0)
    assert received, (
        f"No Image messages received on {topic!r} within 4 s — publisher likely never fired."
    )
    msg = received[0]
    assert msg.width == 160
    assert msg.height == 120
    assert msg.encoding == "bgr8"
    assert msg.step == 160 * 3
    assert len(msg.data) == 160 * 120 * 3
    assert msg.header.frame_id == "cam0"


def test_reader_ros_tee_rejects_missing_topic() -> None:
    """A spec with enable_ros_tee=True but no ros_topic must raise on construction."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=80,
        height=60,
        fps=30,
        enable_nvmm=False,
        enable_ros_tee=True,
    )
    with pytest.raises(ROSConfigError, match="no ``ros_topic``"):
        GStreamerSensorReader(sensor_id="cam0", spec=spec)


# ── Factory wiring ───────────────────────────────────────────────────────────


def test_schema_rejects_publish_to_ros_without_topic() -> None:
    """publish_to_ros=True must come with publish_topic — enforced at the schema layer."""
    from pydantic_core import ValidationError

    with pytest.raises(ValidationError, match="publish_topic is unset"):
        SensorReaderConfig.model_validate(
            {
                "sensor_id": "cam0",
                "backend": "gstreamer",
                "backend_params": {
                    "source": "testsrc",
                    "width": 80,
                    "height": 60,
                    "fps": 30,
                },
                "publish_to_ros": True,
                # publish_topic intentionally omitted
            }
        )


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_factory_builds_ros_tee_reader_end_to_end() -> None:
    """Factory wires SensorReaderConfig(publish_to_ros=True, publish_topic=...) cleanly."""
    cfg = SensorReaderConfig.model_validate(
        {
            "sensor_id": "cam0",
            "backend": "gstreamer",
            "backend_params": {
                "source": "testsrc",
                "width": 80,
                "height": 60,
                "fps": 30,
                "enable_nvmm": False,
            },
            "publish_to_ros": True,
            "publish_topic": "/bh_test/factory/image_raw",
            "publish_rate_hz": 5.0,
        }
    )
    reader = _make_gstreamer_reader(cfg)
    assert isinstance(reader, GStreamerSensorReader)
    # open() must succeed (ROS env is sourced) and start the publisher branch.
    with reader:
        time.sleep(0.2)
        assert reader._ros_publisher is not None
        assert reader._ros_publisher.is_started
