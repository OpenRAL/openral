"""Unit + integration tests for the GStreamer perception event tee (ADR-0018 F6).

Exercises the real :class:`PerceptionEventPublisher` against a real
GStreamer pipeline. The live publish/subscribe round-trip is gated on
``OPENRAL_TEST_ROS_LIVE=1`` (matching the convention in
``tests/unit/test_gstreamer_ros_tee.py``) — gates exist because
``rclpy`` + DDS can clash with a prior glib pulled in transitively by
``torch`` / ``pyarrow`` during the regular pytest run.

No mocks. The two pure-Python detector tests run unconditionally and
exercise :class:`MotionDetector` / :class:`SceneChangeDetector` over a
hand-crafted BGR byte buffer — closer to the real GStreamer payload than
any fake would be.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from openral_core import (
    MotionMetadata,
    PerceptionEventMetadata,
    SceneChangeMetadata,
)
from openral_runner.backends.gstreamer.perception_tee import (
    TOPIC_PREFIX,
    EventDetector,
    MotionDetector,
    PerceptionEventPublisher,
    SceneChangeDetector,
)
from openral_runner.backends.gstreamer.pipeline import (
    TEE_NAME,
    PipelineSpec,
    Platform,
    Source,
    build_pipeline_string,
)

# ── Pipeline-string builder (pure Python — no GStreamer at runtime) ──────────


def test_pipeline_string_event_tee_only() -> None:
    """Enabling only event_tee gives a 2-leg tee: policy + event branches."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=160,
        height=120,
        fps=30,
        enable_nvmm=False,
        enable_event_tee=True,
        event_rate_hz=5.0,
    )
    out = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert f"tee name={TEE_NAME}" in out
    assert "appsink name=bh_sink" in out
    assert "appsink name=event_sink" in out
    # The event leg pins format=BGR and rate=5/1.
    assert "format=BGR" in out
    assert "framerate=5/1" in out
    # No ROS branch.
    assert "ros_sink" not in out


def test_pipeline_string_three_leg_tee_when_both_tees_enabled() -> None:
    """Enabling ros + event together gives a 3-leg tee (policy + ros + event)."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=160,
        height=120,
        fps=30,
        enable_nvmm=False,
        enable_ros_tee=True,
        enable_event_tee=True,
    )
    out = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert "appsink name=bh_sink" in out
    assert "appsink name=ros_sink" in out
    assert "appsink name=event_sink" in out
    # All three legs gate behind a leaky downstream queue so a stalled
    # observability / detector branch can't backpressure the policy.
    assert out.count("queue leaky=downstream") >= 3


def test_pipeline_string_linear_when_neither_tee_enabled() -> None:
    """Backwards-compat: spec with no tees still produces a linear pipeline."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=160,
        height=120,
        fps=30,
        enable_nvmm=False,
    )
    out = build_pipeline_string(spec, platform=Platform.CPU_ONLY)
    assert "tee" not in out
    assert "appsink name=bh_sink" in out


def test_pipeline_spec_event_appsink_name_validated() -> None:
    """A pathological event_appsink_name is rejected at model construction."""
    with pytest.raises(ValueError, match="not a valid GStreamer element name"):
        PipelineSpec(
            source=Source.TESTSRC,
            enable_event_tee=True,
            event_appsink_name="bad name with spaces",
        )


def test_pipeline_spec_event_rate_must_be_positive() -> None:
    """event_rate_hz must be > 0 (pydantic Field constraint)."""
    with pytest.raises(ValueError):
        PipelineSpec(source=Source.TESTSRC, enable_event_tee=True, event_rate_hz=0.0)


# ── Pure-Python detector unit tests ──────────────────────────────────────────


def _solid_bgr(width: int, height: int, value: int) -> bytes:
    """Return a width*height*3 BGR buffer filled with a constant 0..255 value."""
    return bytes([value]) * (width * height * 3)


def test_motion_detector_returns_none_on_first_frame() -> None:
    """First frame has no previous to diff against — must return ``None``."""
    det = MotionDetector(threshold=0.02)
    out = det.detect(_solid_bgr(80, 60, 100), 80, 60, "cam0")
    assert out is None


def test_motion_detector_fires_on_large_delta() -> None:
    """A black→white step trips the threshold; metadata fills out correctly."""
    det = MotionDetector(threshold=0.05)
    det.detect(_solid_bgr(80, 60, 0), 80, 60, "cam0")  # prime
    out = det.detect(_solid_bgr(80, 60, 255), 80, 60, "cam0")
    assert isinstance(out, MotionMetadata)
    assert out.sensor_id == "cam0"
    assert out.threshold == 0.05
    assert out.magnitude > 0.5  # near-full delta


def test_motion_detector_quiet_on_identical_frames() -> None:
    """Two identical frames have zero delta and do not fire."""
    det = MotionDetector(threshold=0.01)
    det.detect(_solid_bgr(80, 60, 128), 80, 60, "cam0")  # prime
    out = det.detect(_solid_bgr(80, 60, 128), 80, 60, "cam0")
    assert out is None


def test_motion_detector_rejects_bad_threshold() -> None:
    """``threshold`` outside [0, 1] is rejected at construction."""
    with pytest.raises(ValueError, match="threshold must be in"):
        MotionDetector(threshold=1.5)


def test_scene_change_detector_fires_on_distinct_frames() -> None:
    """A black→white frame pair is two non-overlapping histograms; distance >> 0."""
    det = SceneChangeDetector(threshold=0.1)
    det.detect(_solid_bgr(80, 60, 0), 80, 60, "cam0")  # prime
    out = det.detect(_solid_bgr(80, 60, 255), 80, 60, "cam0")
    assert isinstance(out, SceneChangeMetadata)
    assert out.sensor_id == "cam0"
    assert out.metric == "chisqr_alt"
    assert out.distance > out.threshold


def test_scene_change_detector_quiet_on_identical_frames() -> None:
    """Identical frames → distance = 0 → no event."""
    det = SceneChangeDetector(threshold=0.01)
    det.detect(_solid_bgr(80, 60, 128), 80, 60, "cam0")  # prime
    out = det.detect(_solid_bgr(80, 60, 128), 80, 60, "cam0")
    assert out is None


def test_motion_summarise_includes_bbox_when_localised() -> None:
    """The summary line carries the bbox when the detector localised the motion."""
    det = MotionDetector(threshold=0.01)
    det.detect(_solid_bgr(80, 60, 0), 80, 60, "cam0")
    metadata = det.detect(_solid_bgr(80, 60, 255), 80, 60, "cam0")
    assert metadata is not None
    summary = det.summarise(metadata)
    assert "motion magnitude=" in summary
    assert "bbox=" in summary
    assert "cam0" in summary


# ── PerceptionEventPublisher pure-construction validation ────────────────────


class _NullDetector(EventDetector):
    """Detector that never fires; used to exercise constructor validation."""

    kind: str = "null"

    def detect(
        self,
        frame_bgr: bytes,
        width: int,
        height: int,
        sensor_id: str,
    ) -> PerceptionEventMetadata | None:
        return None

    def summarise(self, metadata: PerceptionEventMetadata) -> str:  # pragma: no cover
        return f"null on {metadata.sensor_id}"


def test_publisher_rejects_duplicate_kinds() -> None:
    """Two detectors with the same kind would route to the same topic."""
    with pytest.raises(ValueError, match="duplicate detector kinds"):
        PerceptionEventPublisher(
            sensor_id="cam0",
            appsink=object(),  # not started; never touched by ctor
            detectors=[MotionDetector(), MotionDetector()],
        )


def test_publisher_rejects_empty_detector_list() -> None:
    """A publisher with zero detectors has no work — disallow."""
    with pytest.raises(ValueError, match="at least one detector"):
        PerceptionEventPublisher(
            sensor_id="cam0",
            appsink=object(),
            detectors=[],
        )


def test_publisher_rejects_relative_topic_prefix() -> None:
    """Topic prefixes must be absolute (start with '/')."""
    with pytest.raises(ValueError, match="topic_prefix must be absolute"):
        PerceptionEventPublisher(
            sensor_id="cam0",
            appsink=object(),
            detectors=[MotionDetector()],
            topic_prefix="openral/perception",
        )


def test_publisher_rejects_non_positive_rate() -> None:
    """rate_hz must be > 0."""
    with pytest.raises(ValueError, match="rate_hz must be > 0"):
        PerceptionEventPublisher(
            sensor_id="cam0",
            appsink=object(),
            detectors=[MotionDetector()],
            rate_hz=0.0,
        )


def test_topic_prefix_locked_to_adr_path() -> None:
    """The constant must match the contract in ADR-0018 §1."""
    assert TOPIC_PREFIX == "/openral/perception"


# ── Live integration: real GStreamer + real rclpy ────────────────────────────
#
# Gates live behind ``OPENRAL_TEST_ROS_LIVE=1`` to match the convention in
# ``tests/unit/test_gstreamer_ros_tee.py`` (the live rclpy + DDS init can
# clash with a glib pulled in by torch/pyarrow during the regular pytest
# run). Importorskips live *inside* the test function so the pure-Python
# tests above stay collectible on hosts without PyGObject / rclpy.

_LIVE_ROS = bool(os.getenv("OPENRAL_TEST_ROS_LIVE"))
_LIVE_ROS_REASON = (
    "live rclpy publish/subscribe — set OPENRAL_TEST_ROS_LIVE=1 in a clean shell "
    "(no torch import) to exercise"
)


@pytest.mark.skipif(not _LIVE_ROS, reason=_LIVE_ROS_REASON)
def test_perception_publisher_publishes_motion_events_end_to_end() -> None:
    """End-to-end: videotestsrc → MotionDetector → PromptStamped on /openral/perception/motion."""
    gi = pytest.importorskip(
        "gi",
        reason="PyGObject not installed (pip install openral-runner[gstreamer])",
    )
    gi.require_version("Gst", "1.0")
    rclpy = pytest.importorskip(
        "rclpy",
        reason="rclpy not on PYTHONPATH — source a ROS 2 install before live tests",
    )
    pytest.importorskip(
        "openral_msgs.msg",
        reason="openral_msgs not built (run `just ros2-build`)",
    )

    from gi.repository import Gst
    from openral_msgs.msg import PromptStamped
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    Gst.init(None)
    # videotestsrc with pattern=ball gives a moving target — the motion
    # detector reliably fires every frame past frame 0.
    pipeline = Gst.parse_launch(
        "videotestsrc is-live=true pattern=ball num-buffers=60 ! "
        "videoconvert ! video/x-raw,format=BGR,width=160,height=120,framerate=30/1 ! "
        "videorate ! video/x-raw,framerate=10/1 ! "
        "appsink name=event_sink emit-signals=true max-buffers=1 drop=true sync=false",
    )
    appsink = pipeline.get_by_name("event_sink")
    publisher = PerceptionEventPublisher(
        sensor_id="cam0",
        appsink=appsink,
        detectors=[MotionDetector(threshold=0.005)],
        rate_hz=10.0,
    )
    received: list[PromptStamped] = []
    received_event = threading.Event()

    def _subscribe_until_received() -> None:
        if not rclpy.ok():
            rclpy.init()
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node = rclpy.create_node("openral_perception_tee_subscriber")

        def callback(msg: PromptStamped) -> None:
            received.append(msg)
            received_event.set()

        node.create_subscription(
            PromptStamped,
            f"{TOPIC_PREFIX}/motion",
            callback,
            qos,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()

    subscriber = threading.Thread(target=_subscribe_until_received, daemon=True)
    publisher.start()
    try:
        pipeline.set_state(Gst.State.PLAYING)
        subscriber.start()
        subscriber.join(timeout=6.0)
    finally:
        pipeline.set_state(Gst.State.NULL)
        publisher.stop()

    assert received, (
        "No PromptStamped received on /openral/perception/motion within 5 s — "
        "publisher likely never fired."
    )
    msg = received[0]
    assert msg.header.frame_id == "cam0"
    assert "motion" in msg.text
    # metadata_json round-trips through the discriminator.
    from pydantic import TypeAdapter

    metadata = TypeAdapter(PerceptionEventMetadata).validate_json(msg.metadata_json)
    assert isinstance(metadata, MotionMetadata)
    assert metadata.sensor_id == "cam0"
