r"""Live ROS-tee round-trip — runs inside the x86-ros Docker image.

Exercises the GStreamerSensorReader's ROS tee branch end-to-end without
pytest / conftest / torch in the loop:

1. Build a videotestsrc → tee pipeline via :class:`PipelineSpec`.
2. Open the reader; the reader starts a :class:`RosImagePublisher`
   on the ``ros_sink`` appsink.
3. Spin a real ``rclpy`` subscriber in a thread; assert at least one
   ``sensor_msgs/Image`` arrives with the expected shape.

Exits 0 on success, non-zero with an error message otherwise.

Designed to run as:

    docker run --rm --gpus all \
        -v "$(pwd)/docker/inference/smoke_ros_tee.py:/workspace/smoke_ros_tee.py:ro" \
        openral:x86-ros-latest python /workspace/smoke_ros_tee.py
"""

from __future__ import annotations

# ruff: noqa: I001  reason: import order matters — the gstreamer
# subpackage must be imported BEFORE any rclpy module, because importing
# the reader triggers ``Gst.init()`` at module load and rclpy.Node()
# segfaults later when Gst hasn't been initialised first (see PR I/8).
import sys
import threading
import time

from openral_runner.backends.gstreamer import (
    GStreamerSensorReader,
    PipelineSpec,
    Platform,
    Source,
)

import rclpy
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image

TOPIC = "/bh_smoke/ros_tee/image_raw"
WIDTH, HEIGHT = 160, 120


def run() -> int:
    """Drive the round-trip and return a CLI exit status."""
    spec = PipelineSpec(
        source=Source.TESTSRC,
        width=WIDTH,
        height=HEIGHT,
        fps=30,
        enable_nvmm=False,
        enable_ros_tee=True,
    )
    reader = GStreamerSensorReader(
        sensor_id="cam0",
        spec=spec,
        platform=Platform.CPU_ONLY,
        ros_topic=TOPIC,
        ros_rate_hz=10.0,
        default_max_age_ms=500,
    )

    received: list[Image] = []
    received_event = threading.Event()

    print(f"[smoke] opening reader on topic={TOPIC!r} ...", flush=True)
    with reader:
        # Subscribe on the MAIN thread. Spawning a second rclpy node
        # creation on a worker thread is intermittently unsafe in this
        # image (cyclonedds RMW + thread-context interplay observed
        # mid-PR I/8); the publisher already runs on a Gst streaming
        # thread, so the main thread is free to host the subscriber.
        if not rclpy.ok():
            rclpy.init()
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        node = rclpy.create_node("bh_smoke_subscriber")

        def cb(msg: Image) -> None:
            print(
                f"[smoke] cb! width={msg.width} height={msg.height} encoding={msg.encoding!r}",
                flush=True,
            )
            received.append(msg)
            received_event.set()

        node.create_subscription(Image, TOPIC, cb, qos)
        print("[smoke] subscriber created; spinning ...", flush=True)
        deadline = time.monotonic() + 5.0
        spins = 0
        while time.monotonic() < deadline and not received_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
            spins += 1
        print(f"[smoke] spun {spins} times; received={len(received)}", flush=True)
        node.destroy_node()

    if not received:
        print(
            f"[smoke] FAIL — no sensor_msgs/Image messages on {TOPIC!r} within 4 s",
            file=sys.stderr,
        )
        return 1

    msg = received[0]
    print(f"[smoke] OK — received {len(received)} Image(s) on {TOPIC!r}")
    print(
        f"[smoke]   width={msg.width} height={msg.height} encoding={msg.encoding!r} "
        f"step={msg.step} frame_id={msg.header.frame_id!r} data_len={len(msg.data)}"
    )
    expected_data_len = WIDTH * HEIGHT * 3
    if (
        msg.width != WIDTH
        or msg.height != HEIGHT
        or msg.encoding != "bgr8"
        or len(msg.data) != expected_data_len
    ):
        print(
            f"[smoke] FAIL — message shape mismatch "
            f"(want {WIDTH}x{HEIGHT} bgr8 / {expected_data_len} bytes)",
            file=sys.stderr,
        )
        return 2
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    rc = run()
    # Skip Python's atexit/finaliser teardown — the pydantic-Rust /
    # GStreamer / cyclonedds C-extension interaction segfaults at
    # shutdown even when the live round-trip succeeded. We've already
    # verified the round-trip; honour that with the correct exit code
    # via os._exit which bypasses Python's cleanup.
    import os

    os._exit(rc)
