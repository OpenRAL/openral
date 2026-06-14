r"""Live perception-tee round-trip — runs inside the x86-ros Docker image.

Exercises the GStreamer perception event tee (ADR-0018 F6) end-to-end
without pytest / conftest / torch in the loop:

1. Build a ``videotestsrc pattern=ball`` → ``tee`` pipeline directly via
   ``Gst.parse_launch`` with an ``event_sink`` appsink. The event leg is
   the same shape the :func:`build_pipeline_string` builder emits when
   ``PipelineSpec.enable_event_tee=True``.
2. Construct a :class:`PerceptionEventPublisher` with a real
   :class:`MotionDetector` (threshold 0.005 — sensitive enough for the
   bouncing ball pattern).
3. Start the publisher; play the pipeline.
4. Spin a real ``rclpy`` subscriber on ``/openral/perception/motion``;
   assert at least one ``openral_msgs/PromptStamped`` arrives whose
   ``metadata_json`` decodes through the
   :data:`openral_core.PerceptionEventMetadata` discriminated union
   into a :class:`MotionMetadata` carrying ``sensor_id="cam0"``.

Exits 0 on success, non-zero with an error message otherwise.

Designed to run as:

    docker run --rm --gpus all \
        -v "$(pwd)/docker/inference/smoke_perception_tee.py:/workspace/smoke_perception_tee.py:ro" \
        openral:x86-latest python /workspace/smoke_perception_tee.py
"""

from __future__ import annotations

# ruff: noqa: E402, I001  reason: import order matters — the gstreamer
# subpackage must be imported BEFORE any rclpy module so Gst.init runs
# first (same ordering constraint as smoke_ros_tee.py).
import os
import sys
import threading
import time

from openral_runner.backends.gstreamer.perception_tee import (
    MotionDetector,
    PerceptionEventPublisher,
    TOPIC_PREFIX,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import rclpy
from openral_core import MotionMetadata, PerceptionEventMetadata
from openral_msgs.msg import PromptStamped
from pydantic import TypeAdapter
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

SENSOR_ID = "cam0"
TOPIC = f"{TOPIC_PREFIX}/motion"
WIDTH, HEIGHT = 160, 120
EVENT_RATE_HZ = 10

# Mirrors the pipeline assembled by
# ``build_pipeline_string(spec, platform=Platform.CPU_ONLY)`` when
# ``enable_event_tee=True``. Kept inline here so the smoke script does
# not depend on the YAML/factory path — the goal is to validate the
# event leg + publisher in isolation.
PIPELINE = (
    "videotestsrc is-live=true pattern=ball ! "
    f"videoconvert ! video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate=30/1 ! "
    "tee name=openral_cam_tee "
    "  openral_cam_tee. ! queue leaky=downstream max-size-buffers=2 ! fakesink sync=false "
    "  openral_cam_tee. ! queue leaky=downstream max-size-buffers=2 ! videorate ! "
    f"video/x-raw,format=BGR,framerate={EVENT_RATE_HZ}/1 ! "
    "appsink name=event_sink emit-signals=true max-buffers=1 drop=true sync=false"
)


def run() -> int:
    """Drive the round-trip and return a CLI exit status."""
    Gst.init(None)
    pipeline = Gst.parse_launch(PIPELINE)
    appsink = pipeline.get_by_name("event_sink")
    if appsink is None:  # pragma: no cover — parse_launch already raises on bad pipelines
        print("[smoke] FAIL — pipeline missing event_sink appsink", file=sys.stderr)
        return 3

    publisher = PerceptionEventPublisher(
        sensor_id=SENSOR_ID,
        appsink=appsink,
        detectors=[MotionDetector(threshold=0.005)],
        rate_hz=float(EVENT_RATE_HZ),
    )

    received: list[PromptStamped] = []
    received_event = threading.Event()

    print(f"[smoke] starting perception publisher on topic={TOPIC!r} ...", flush=True)
    publisher.start()
    try:
        if not rclpy.ok():
            rclpy.init()
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node = rclpy.create_node("openral_smoke_perception_subscriber")

        def cb(msg: PromptStamped) -> None:
            print(
                f"[smoke] cb! frame_id={msg.header.frame_id!r} text={msg.text!r}",
                flush=True,
            )
            received.append(msg)
            received_event.set()

        node.create_subscription(PromptStamped, TOPIC, cb, qos)
        print("[smoke] subscriber created; pipeline -> PLAYING; spinning ...", flush=True)
        pipeline.set_state(Gst.State.PLAYING)

        deadline = time.monotonic() + 6.0
        spins = 0
        while time.monotonic() < deadline and not received_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            spins += 1
        print(f"[smoke] spun {spins} times; received={len(received)}", flush=True)
        node.destroy_node()
    finally:
        pipeline.set_state(Gst.State.NULL)
        publisher.stop()

    if not received:
        print(
            f"[smoke] FAIL — no PromptStamped on {TOPIC!r} within 6 s",
            file=sys.stderr,
        )
        return 1

    msg = received[0]
    if msg.header.frame_id != SENSOR_ID:
        print(
            f"[smoke] FAIL — wrong header.frame_id "
            f"(want {SENSOR_ID!r}, got {msg.header.frame_id!r})",
            file=sys.stderr,
        )
        return 2
    metadata = TypeAdapter(PerceptionEventMetadata).validate_json(msg.metadata_json)
    if not isinstance(metadata, MotionMetadata):
        print(
            f"[smoke] FAIL — wrong PerceptionEventMetadata kind "
            f"(want motion, got {type(metadata).__name__})",
            file=sys.stderr,
        )
        return 2
    if metadata.sensor_id != SENSOR_ID:
        print(
            f"[smoke] FAIL — metadata.sensor_id mismatch "
            f"(want {SENSOR_ID!r}, got {metadata.sensor_id!r})",
            file=sys.stderr,
        )
        return 2

    print(
        f"[smoke] OK — received {len(received)} PromptStamped on {TOPIC!r}",
        flush=True,
    )
    print(
        f"[smoke]   text={msg.text!r}\n"
        f"[smoke]   metadata.magnitude={metadata.magnitude:.4f} "
        f"threshold={metadata.threshold} region_bbox={metadata.region_bbox!r}",
        flush=True,
    )
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    rc = run()
    # Skip Python's atexit/finaliser teardown — pydantic-Rust /
    # GStreamer / cyclonedds C-extension interaction segfaults at
    # shutdown even when the round-trip succeeded (same workaround as
    # docker/inference/smoke_ros_tee.py).
    os._exit(rc)
