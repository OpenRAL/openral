# SPDX-License-Identifier: Apache-2.0
"""``ros2_image`` reader — the backend for streams a device cannot emit.

A StereoLabs ZED presents ONE side-by-side UVC node over USB; its depth is
computed on the host GPU by the ZED SDK and only ever *published*, on
``/<name>/depth/depth_registered``. The catalog's ``stereolabs/zed_mini``
bundle has always declared that depth stream — but with no ``ros2_image``
backend registered, nothing could subscribe to it, so the declaration was
undeliverable. Same for RealSense aligned depth and any other driver-owned
stream.

These tests use a **real rclpy publisher and a real subscription** — a
``sensor_msgs/Image`` genuinely crossing DDS — rather than duck-typing the
message, because the parts most worth pinning (QoS compatibility, the buffer,
the byte layout of a 32FC1 depth frame) only exist on the wire.

The graph is confined to its own DDS domain with LOCALHOST discovery. That is
not ceremony: on 2026-09-05 a sim left on domain 0 with subnet multicast
reached a **live bimanual OpenArm on another host** (#227), and a test that
publishes ``/openral/...``-shaped images is exactly the kind that must not.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from openral_core import FrameEncoding, SensorReaderConfig
from openral_core.exceptions import ROSConfigError, ROSPerceptionStale
from openral_runner.backends.ros2_image import (
    Ros2ImageSensorReader,
    _depth32f_to_depth16,
)
from openral_runner.factory import make_sensor_readers

rclpy = pytest.importorskip("rclpy", reason="ros2_image needs a ROS 2 install")
pytest.importorskip("sensor_msgs", reason="ros2_image needs sensor_msgs")

from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

_TEST_DOMAIN = "91"
_SETTLE_TIMEOUT_S = 10.0


@pytest.fixture
def isolated_ros(monkeypatch: pytest.MonkeyPatch):
    """A private DDS domain, LOCALHOST-only, torn down after the test."""
    monkeypatch.setenv("ROS_DOMAIN_ID", _TEST_DOMAIN)
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    context = rclpy.Context()
    context.init()
    yield context
    context.try_shutdown()


def _publisher(context, topic: str):
    from rclpy.node import Node

    node = Node("ros2_image_test_pub", context=context)
    qos = QoSProfile(
        depth=5, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT
    )
    return node, node.create_publisher(Image, topic, qos)


def _rgb_msg(height: int = 2, width: int = 3) -> Image:
    msg = Image()
    msg.height, msg.width = height, width
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = width * 3
    # Values double as their flat index so a reshape/stride error is visible.
    msg.data = bytes(range(height * width * 3))
    return msg


def _depth32f_msg(values: list[list[float]]) -> Image:
    array = np.asarray(values, dtype=np.float32)
    msg = Image()
    msg.height, msg.width = array.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = array.shape[1] * 4
    msg.data = array.tobytes()
    return msg


class TestLiveRoundTrip:
    def test_an_rgb_image_crosses_dds_into_a_sensor_frame(self, isolated_ros) -> None:
        topic = "/test_zed/left/image_rect_color"
        node, publisher = _publisher(isolated_ros, topic)
        sub_node = _sub_node(isolated_ros)
        reader = Ros2ImageSensorReader(sensor_id="head_rgb", topic=topic, node=sub_node)
        reader.open()
        try:
            msg = _rgb_msg()
            _pump_until_frame(reader, sub_node, node, publisher, msg, isolated_ros)
            frame = reader.read_latest(max_age_ms=5_000)
            assert frame.encoding is FrameEncoding.RGB8
            assert (frame.height, frame.width, frame.channels) == (2, 3, 3)
            assert frame.data == bytes(range(18))
        finally:
            reader.close()
            sub_node.destroy_node()
            node.destroy_node()

    def test_zed_style_32fc1_depth_arrives_as_depth16_millimetres(self, isolated_ros) -> None:
        """The stream the ZED bundle declares and nothing could previously read."""
        topic = "/test_zed/depth/depth_registered"
        node, publisher = _publisher(isolated_ros, topic)
        sub_node = _sub_node(isolated_ros)
        reader = Ros2ImageSensorReader(sensor_id="head_depth", topic=topic, node=sub_node)
        reader.open()
        try:
            # 1.5 m in range; NaN is a stereo matcher's "no match".
            msg = _depth32f_msg([[1.5, float("nan")], [0.25, 15.0]])
            _pump_until_frame(reader, sub_node, node, publisher, msg, isolated_ros)
            frame = reader.read_latest(max_age_ms=5_000)
            assert frame.encoding is FrameEncoding.DEPTH16
            assert frame.channels == 1
            mm = np.frombuffer(frame.data, dtype="<u2").reshape(2, 2)
            assert mm.tolist() == [[1500, 0], [250, 15000]]
        finally:
            reader.close()
            sub_node.destroy_node()
            node.destroy_node()

    def test_a_pitch_aligned_row_stride_is_sliced_off(self, isolated_ros) -> None:
        """`step` > the packed row length is a real publisher, not a malformed one.

        Isaac/NITROS hand out pitch-aligned buffers and an ROI crop keeps its
        parent's stride. Reading the payload as `height x width` pixels then
        raises, and `_on_image` downgrades that to a WARN — so the sensor goes
        permanently `ROSPerceptionStale` and reads exactly like a dead camera.
        """
        topic = "/test_isaac/left/image_rect_color"
        node, publisher = _publisher(isolated_ros, topic)
        sub_node = _sub_node(isolated_ros)
        reader = Ros2ImageSensorReader(sensor_id="head_rgb", topic=topic, node=sub_node)
        reader.open()
        try:
            msg = _rgb_msg()
            msg.step = 3 * 3 + 5  # 5 pad bytes per row, as a 4-byte-aligned driver emits
            msg.data = bytes(
                b"".join(bytes(range(r * 9, r * 9 + 9)) + b"\xff" * 5 for r in range(2))
            )
            _pump_until_frame(reader, sub_node, node, publisher, msg, isolated_ros)
            frame = reader.read_latest(max_age_ms=5_000)
            assert (frame.height, frame.width, frame.channels) == (2, 3, 3)
            assert frame.data == bytes(range(18))  # padding sliced, not folded in
        finally:
            reader.close()
            sub_node.destroy_node()
            node.destroy_node()


def _sub_node(context):
    from rclpy.node import Node

    return Node("ros2_image_test_sub", context=context)


def _pump_until_frame(reader, sub_node, pub_node, publisher, msg, context) -> None:
    """Republish + spin both nodes until the reader buffers a frame.

    BEST_EFFORT drops anything published before discovery completes, so a
    single publish races the subscription match — republish rather than sleep
    on a guess. The executor is bound to the test's private context;
    ``rclpy.spin_once`` would use the (uninitialised) default one.
    """
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor(context=context)
    executor.add_node(pub_node)
    executor.add_node(sub_node)
    try:
        deadline = time.monotonic() + _SETTLE_TIMEOUT_S
        while time.monotonic() < deadline:
            publisher.publish(msg)
            executor.spin_once(timeout_sec=0.05)
            try:
                reader.read_latest(max_age_ms=5_000)
            except ROSPerceptionStale:
                continue
            return
    finally:
        executor.shutdown()
    pytest.fail(f"no frame reached the reader within {_SETTLE_TIMEOUT_S}s")


class TestDepthConversion:
    """`32FC1` metres → `DEPTH16` millimetres, with the invalid cases pinned.

    A wrapped uint16 would report a confident, wrong, NEAR distance for
    something far away, so everything non-finite or out of range must land on
    0 — the ROS "no reading" value.
    """

    def test_metres_become_millimetres(self) -> None:
        out = _depth32f_to_depth16(np.array([[0.1, 1.5, 15.0]], np.float32))
        assert out.tolist() == [[100, 1500, 15000]]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0, 1e6])
    def test_invalid_samples_become_no_reading(self, value: float) -> None:
        out = _depth32f_to_depth16(np.array([[value]], np.float32))
        assert out.tolist() == [[0]]

    def test_the_uint16_ceiling_is_not_wrapped(self) -> None:
        # 70 m > 65.535 m. Wrapping would report ~4.4 m — a confident near miss.
        out = _depth32f_to_depth16(np.array([[70.0]], np.float32))
        assert out.tolist() == [[0]]


class TestStalenessContract:
    def test_read_on_a_closed_reader_raises(self) -> None:
        reader = Ros2ImageSensorReader(sensor_id="x", topic="/t")
        with pytest.raises(RuntimeError, match="closed reader"):
            reader.read_latest()

    def test_no_frame_yet_names_the_qos_trap(self, isolated_ros) -> None:
        """The message must name the QoS mismatch — it reads like a dead camera."""
        sub_node = _sub_node(isolated_ros)
        reader = Ros2ImageSensorReader(sensor_id="x", topic="/never_published", node=sub_node)
        reader.open()
        try:
            with pytest.raises(ROSPerceptionStale, match="no frame received yet"):
                reader.read_latest()
        finally:
            reader.close()
            sub_node.destroy_node()


class TestTeardown:
    """`close` releases what `open` built, including from a failed `open`."""

    def test_close_releases_a_node_from_an_open_that_failed_late(self, isolated_ros) -> None:
        """`open` sets `is_open` last, so a late failure leaves it false.

        An `is_open` early return in `close` would then skip teardown entirely
        and strand the node — the docstring's "safe after a partially-failed
        open" has to hold against the flag, not because of it. A subscription
        to an invalid topic name fails after the node exists.
        """
        sub_node = _sub_node(isolated_ros)
        reader = Ros2ImageSensorReader(sensor_id="x", topic="/not a topic", node=sub_node)
        with pytest.raises(Exception):  # noqa: B017  # reason: rclpy's own name error
            reader.open()
        assert reader.is_open is False
        reader.close()
        reader.close()  # idempotent without the flag guarding it
        sub_node.destroy_node()


class TestConfigValidation:
    def test_an_empty_topic_is_refused(self) -> None:
        with pytest.raises(ROSConfigError, match="non-empty"):
            Ros2ImageSensorReader(sensor_id="x", topic="")

    def test_an_unknown_reliability_is_refused(self) -> None:
        with pytest.raises(ROSConfigError, match="best_effort"):
            Ros2ImageSensorReader(sensor_id="x", topic="/t", reliability="sometimes")

    def test_the_factory_builds_the_backend_from_a_config(self) -> None:
        """`ros2_image` must be registered — it was in the enum but not the registry."""
        cfg = SensorReaderConfig(
            sensor_id="head_depth",
            backend="ros2_image",
            backend_params={"topic": "/zed/depth/depth_registered", "qos_depth": 10},
            max_age_ms=250,
        )
        (reader,) = make_sensor_readers([cfg])
        assert isinstance(reader, Ros2ImageSensorReader)
        assert reader.sensor_id == "head_depth"
        assert reader.is_open is False

    def test_the_factory_refuses_a_config_with_no_topic(self) -> None:
        cfg = SensorReaderConfig(sensor_id="head_depth", backend="ros2_image", backend_params={})
        with pytest.raises(ROSConfigError, match=r"backend_params\.topic"):
            make_sensor_readers([cfg])
