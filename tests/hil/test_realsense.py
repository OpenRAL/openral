"""HIL smoke test — Intel RealSense D435 connected and publishing.

Guards
------
- Skipped if ``pyrealsense2`` is not installed (no SDK).
- Skipped if no RealSense device is detected by the SDK.
- Skipped if ROS 2 is not sourced (no ``ROS_DISTRO`` env var).

Run on the lab runner with::

    pytest tests/hil/test_realsense.py -v --timeout=60

The test verifies:
1. At least one RealSense device is visible to the SDK.
2. An RGB frame and a depth frame can be captured within 5 s.
3. RGB and depth resolutions match the SensorSpec declared for the D435
   (640x480 nominal).
4. The IMU stream delivers accelerometer data.
"""

from __future__ import annotations

import os
import time

import pytest

# ── Skip guards ───────────────────────────────────────────────────────────────

pyrealsense2 = pytest.importorskip(
    "pyrealsense2",
    reason="pyrealsense2 SDK not installed; skipping RealSense HIL tests.",
)

if not os.environ.get("ROS_DISTRO"):
    pytest.skip(
        "ROS_DISTRO not set — source your ROS 2 installation first.",
        allow_module_level=True,
    )

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rs_pipeline():  # type: ignore[no-untyped-def]
    """Yield an active RealSense pipeline; skip if no device is connected."""
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        pytest.skip("No RealSense device detected; skipping HIL tests.")

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.accel)

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    yield pipeline, profile
    pipeline.stop()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_device_detected(rs_pipeline: object) -> None:  # type: ignore[no-untyped-def]
    """At least one RealSense device must be present."""
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    ctx = rs.context()
    assert len(ctx.query_devices()) >= 1


def test_rgb_frame_received(rs_pipeline: object) -> None:  # type: ignore[no-untyped-def]
    """An RGB frame must arrive within 5 seconds."""
    pipeline, _ = rs_pipeline  # type: ignore[misc]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            color = frames.get_color_frame()
            if color:
                assert color.get_width() == 640
                assert color.get_height() == 480
                return
    pytest.fail("No RGB frame received within 5 s.")


def test_depth_frame_received(rs_pipeline: object) -> None:  # type: ignore[no-untyped-def]
    """A depth frame must arrive within 5 seconds."""
    pipeline, _ = rs_pipeline  # type: ignore[misc]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            depth = frames.get_depth_frame()
            if depth:
                assert depth.get_width() == 640
                assert depth.get_height() == 480
                return
    pytest.fail("No depth frame received within 5 s.")


def test_imu_accel_received(rs_pipeline: object) -> None:  # type: ignore[no-untyped-def]
    """Accelerometer data must arrive within 2 seconds."""
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    pipeline, _ = rs_pipeline  # type: ignore[misc]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            for i in range(frames.size()):
                f = frames[i]
                if f.profile.stream_type() == rs.stream.accel:
                    data = f.as_motion_frame().get_motion_data()
                    # Gravity should register on at least one axis
                    assert abs(data.x) + abs(data.y) + abs(data.z) > 0.1
                    return
    pytest.fail("No IMU accelerometer frame received within 2 s.")


def test_rgb_rate_at_least_25hz(rs_pipeline: object) -> None:  # type: ignore[no-untyped-def]
    """RGB must publish at ≥ 25 Hz sustained over 2 seconds (SensorSpec: 30 Hz)."""
    pipeline, _ = rs_pipeline  # type: ignore[misc]
    frames_received = 0
    start = time.monotonic()
    duration = 2.0
    deadline = start + duration
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames and frames.get_color_frame():
            frames_received += 1
        time.sleep(0.005)
    actual_rate = frames_received / duration
    assert actual_rate >= 25.0, f"RGB rate too low: {actual_rate:.1f} Hz (need ≥ 25 Hz)"
