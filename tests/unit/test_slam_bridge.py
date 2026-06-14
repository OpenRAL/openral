"""ADR-0025 — unit coverage for the rclpy → OTLP SLAM map bridge.

Two layers:

1. :func:`encode_occupancy_grid_png` is a pure function — tested
   directly against synthetic ``nav_msgs/OccupancyGrid``-shaped data.
2. The dashboard store handler for ``slam.occupancy_grid`` spans is
   tested by manufacturing a minimal OTLP ``Span`` payload and
   asserting ``_topics["slam"]`` ends up populated correctly.

Both layers run without rclpy / a ROS 2 workspace; the live
``SlamMapBridge`` exercise lives in the integration tier alongside
the slam_toolbox launch test.
"""

from __future__ import annotations

import base64
import io
import math

import pytest


def test_encode_grid_round_trips_through_pil() -> None:
    """PNG decodes back to a real image with the right dimensions."""
    pytest.importorskip("PIL")
    from openral_runner.slam_bridge import encode_occupancy_grid_png

    # 3×2 grid: free / occupied / unknown / 50%-prob / occupied / free
    data = [0, 100, -1, 50, 100, 0]
    encoded = encode_occupancy_grid_png(width=3, height=2, data=data)

    raw = base64.b64decode(encoded)
    # PNG magic bytes
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    assert img.size == (3, 2)
    assert img.mode == "L"


def test_encode_grid_rejects_size_mismatch() -> None:
    from openral_runner.slam_bridge import encode_occupancy_grid_png

    with pytest.raises(ValueError, match="width\\*height"):
        encode_occupancy_grid_png(width=3, height=2, data=[0, 0, 0])


def test_encode_grid_unknown_cells_become_mid_grey() -> None:
    """Unknown (-1) cells render to 128 so they're visually distinct."""
    pytest.importorskip("PIL")
    from openral_runner.slam_bridge import encode_occupancy_grid_png

    encoded = encode_occupancy_grid_png(width=2, height=1, data=[-1, -1])
    raw = base64.b64decode(encoded)
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("L")
    # Whole image is mid-grey.
    pixels = list(img.getdata())
    assert pixels == [128, 128]


def test_yaw_from_quaternion_identity_is_zero() -> None:
    from openral_runner.slam_bridge import yaw_from_quaternion_xyzw

    assert yaw_from_quaternion_xyzw(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)


def test_yaw_from_quaternion_ninety_deg_about_z() -> None:
    from openral_runner.slam_bridge import yaw_from_quaternion_xyzw

    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    assert yaw_from_quaternion_xyzw(0.0, 0.0, s, c) == pytest.approx(math.pi / 2)


def test_yaw_from_quaternion_negative_ninety_deg_about_z() -> None:
    from openral_runner.slam_bridge import yaw_from_quaternion_xyzw

    s = math.sin(-math.pi / 4)
    c = math.cos(-math.pi / 4)
    assert yaw_from_quaternion_xyzw(0.0, 0.0, s, c) == pytest.approx(-math.pi / 2)


def test_robot_pose_from_transform_extracts_xy_and_yaw() -> None:
    from openral_runner.slam_bridge import robot_pose_from_transform

    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    x, y, yaw = robot_pose_from_transform(
        translation_xyz=(1.5, -2.25, 0.0),
        rotation_xyzw=(0.0, 0.0, s, c),
    )
    assert x == pytest.approx(1.5)
    assert y == pytest.approx(-2.25)
    assert yaw == pytest.approx(math.pi / 2)


def test_dashboard_store_picks_up_slam_occupancy_grid_span() -> None:
    """End-to-end synthetic-span path matches the bridge's attribute shape.

    Builds a single ``slam.occupancy_grid`` OTLP span by hand and feeds
    it through ``TelemetryStore.ingest_spans``. The ``_topics["slam"]`` slot
    must come out populated with every documented attribute.
    """
    pytest.importorskip("opentelemetry.proto")
    from openral_observability.dashboard.store import TelemetryStore
    from opentelemetry.proto.common.v1.common_pb2 import (
        AnyValue,
        ArrayValue,
        KeyValue,
    )
    from opentelemetry.proto.trace.v1.trace_pb2 import (
        ResourceSpans,
        ScopeSpans,
        Span,
    )

    span = Span(
        trace_id=b"0" * 16,
        span_id=b"0" * 8,
        name="slam.occupancy_grid",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_000_100_000,
        attributes=[
            KeyValue(key="openral.slam.frame_id", value=AnyValue(string_value="map")),
            KeyValue(key="openral.slam.width", value=AnyValue(int_value=128)),
            KeyValue(key="openral.slam.height", value=AnyValue(int_value=64)),
            KeyValue(key="openral.slam.resolution_m", value=AnyValue(double_value=0.05)),
            KeyValue(key="openral.slam.origin_x", value=AnyValue(double_value=-3.2)),
            KeyValue(key="openral.slam.origin_y", value=AnyValue(double_value=-1.6)),
            KeyValue(
                key="openral.slam.png_b64",
                value=AnyValue(string_value="iVBORw0KGgo"),  # truncated; just a marker
            ),
            KeyValue(
                key="openral.slam.source_node",
                value=AnyValue(string_value="openral_slam_toolbox"),
            ),
            KeyValue(key="openral.slam.robot_x", value=AnyValue(double_value=0.75)),
            KeyValue(key="openral.slam.robot_y", value=AnyValue(double_value=-0.40)),
            KeyValue(key="openral.slam.robot_yaw", value=AnyValue(double_value=1.5708)),
            KeyValue(key="openral.slam.base_frame", value=AnyValue(string_value="base_link")),
            KeyValue(
                key="openral.slam.footprint_radius_m",
                value=AnyValue(double_value=0.30),
            ),
            KeyValue(
                key="openral.slam.footprint_polygon_xy",
                value=AnyValue(
                    array_value=ArrayValue(
                        values=[
                            AnyValue(double_value=0.35),
                            AnyValue(double_value=0.25),
                            AnyValue(double_value=-0.35),
                            AnyValue(double_value=0.25),
                            AnyValue(double_value=-0.35),
                            AnyValue(double_value=-0.25),
                            AnyValue(double_value=0.35),
                            AnyValue(double_value=-0.25),
                        ]
                    )
                ),
            ),
        ],
    )
    scope_spans = ScopeSpans(spans=[span])
    resource_spans = ResourceSpans(scope_spans=[scope_spans])

    store = TelemetryStore()
    store.ingest_spans([resource_spans])
    snapshot = store.snapshot()

    slam = snapshot["topics"]["slam"]
    assert slam["frame_id"] == "map"
    assert slam["width"] == 128
    assert slam["height"] == 64
    assert slam["resolution_m"] == pytest.approx(0.05)
    assert slam["origin_x"] == pytest.approx(-3.2)
    assert slam["origin_y"] == pytest.approx(-1.6)
    assert slam["png_b64"] == "iVBORw0KGgo"
    assert slam["source_node"] == "openral_slam_toolbox"
    assert "ts_unix" in slam
    assert slam["robot_x"] == pytest.approx(0.75)
    assert slam["robot_y"] == pytest.approx(-0.40)
    assert slam["robot_yaw"] == pytest.approx(1.5708)
    assert slam["base_frame"] == "base_link"
    assert slam["footprint_radius_m"] == pytest.approx(0.30)
    assert slam["footprint_polygon"] == [
        [0.35, 0.25],
        [-0.35, 0.25],
        [-0.35, -0.25],
        [0.35, -0.25],
    ]


def _footprint_to_pixels_ref(
    polygon, robot_x, robot_y, yaw, origin_x, origin_y, resolution, height
):
    """Python reference for the JS footprintToPixels: rotate base-frame
    vertices by yaw, translate to the robot world pose, then world->pixel
    (with the vertical flip py = (height-1) - row)."""
    import math

    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for bx, by in polygon:
        wx = robot_x + bx * c - by * s
        wy = robot_y + bx * s + by * c
        col = (wx - origin_x) / resolution
        row = (wy - origin_y) / resolution
        out.append((col, (height - 1) - row))
    return out


def test_footprint_to_pixels_rotation_and_flip() -> None:
    import math

    # yaw=0: base +x (front) maps east (col+), +y maps up (smaller py).
    pts = _footprint_to_pixels_ref([(1.0, 0.0), (0.0, 1.0)], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0)
    assert pts[0] == (1.0, 9.0)  # (1,0) -> col 1, row 0, py 9
    assert pts[1] == (0.0, 8.0)  # (0,1) -> col 0, row 1, py 8

    # yaw=+90deg: base +x rotates to world +y (north -> up).
    pts = _footprint_to_pixels_ref([(1.0, 0.0)], 0.0, 0.0, math.pi / 2, 0.0, 0.0, 1.0, 10.0)
    assert pts[0][0] == pytest.approx(0.0)
    assert pts[0][1] == pytest.approx(8.0)
