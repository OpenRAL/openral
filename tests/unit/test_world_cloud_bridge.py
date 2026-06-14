"""ADR-0030 — unit coverage for the robot-perspective world-cloud bridge.

Two layers, both runnable without rclpy / a ROS 2 workspace (mirrors
``tests/unit/test_slam_bridge.py``):

1. The pure render core (:func:`crop_points_to_box`,
   :func:`distance_to_rgb`, :func:`encode_world_cloud_png`,
   :func:`world_cloud_span_attributes`) — tested directly against
   synthetic ``(N, 3)`` arrays.
2. The dashboard store handler for ``world.pointcloud`` spans — a
   hand-built OTLP span fed through ``TelemetryStore.ingest_spans``,
   asserting ``_topics["pointcloud"]`` comes out populated.

The live ``WorldCloudBridge`` rclpy exercise belongs in the integration
tier alongside the octomap launch test.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from openral_runner.world_cloud_bridge import (
    WORLD_CLOUD_TOPIC_DEFAULT,
    _project_chase_view,
    crop_points_to_box,
    distance_to_rgb,
    encode_world_cloud_png,
    world_cloud_span_attributes,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decode(png_b64: str) -> bytes:
    return base64.b64decode(png_b64.encode("ascii"))


def test_topic_default_is_octomap_centers() -> None:
    assert WORLD_CLOUD_TOPIC_DEFAULT == "/octomap_point_cloud_centers"


def test_crop_drops_points_outside_box() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],  # keep
            [1.0, -1.0, 0.5],  # keep
            [5.0, 0.0, 0.0],  # drop: |x| > xy_m
            [0.0, 0.0, 3.0],  # drop: z > z_max
            [0.0, 0.0, -1.0],  # drop: z < z_min
        ],
        dtype=np.float32,
    )
    kept = crop_points_to_box(pts, xy_m=2.0, z_min=-0.2, z_max=2.0)
    assert kept.shape == (2, 3)
    assert kept.dtype == np.float32


def test_crop_empty_input_returns_empty() -> None:
    kept = crop_points_to_box(np.zeros((0, 3), dtype=np.float32), xy_m=2.0, z_min=-0.2, z_max=2.0)
    assert kept.shape == (0, 3)


def test_encode_returns_valid_png() -> None:
    pytest.importorskip("PIL")
    pts = np.random.default_rng(0).uniform(-1.5, 1.5, size=(2000, 3)).astype(np.float32)
    png_b64 = encode_world_cloud_png(pts, range_max_m=4.0, image_w=480, image_h=360)
    raw = _decode(png_b64)
    assert raw.startswith(_PNG_MAGIC)
    assert len(raw) > 100


def test_encode_empty_cloud_is_valid_background_png() -> None:
    pytest.importorskip("PIL")
    pts = np.zeros((0, 3), dtype=np.float32)
    png_b64 = encode_world_cloud_png(pts, range_max_m=4.0)
    assert _decode(png_b64).startswith(_PNG_MAGIC)


def test_color_varies_with_distance() -> None:
    pytest.importorskip("PIL")
    assert distance_to_rgb(0.3, range_max_m=4.0) != distance_to_rgb(1.9, range_max_m=4.0)
    # Both endpoints encode without error.
    encode_world_cloud_png(np.array([[0.3, 0.0, 0.0]], dtype=np.float32), range_max_m=4.0)
    encode_world_cloud_png(np.array([[1.9, 0.0, 0.0]], dtype=np.float32), range_max_m=4.0)


def test_chase_view_preserves_vertical_structure() -> None:
    """A full-height scene must spread across the frame, not pancake to the floor.

    ADR-0030 regression: the chase camera sits behind+above the robot and looks
    forward-and-down, so a cloud filling the crop box in height (floor → ~2 m)
    must project onto a wide band of image rows in height order — not collapse
    onto the bottom edge (the "flattened octomap" symptom).
    """
    image_w, image_h = 480, 360
    # A grid filling the local crop box: x/y in [-1.5, 1.5], z over the full
    # [-0.2, 2.0] height band the renderer keeps.
    axis = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
    zs = np.linspace(-0.2, 2.0, 12, dtype=np.float32)
    grid = np.array([[x, y, z] for x in axis for y in axis for z in zs], dtype=np.float32)

    us, vs, _depth, _order = _project_chase_view(grid, image_w=image_w, image_h=image_h)
    on_screen = us >= 0
    rows = image_h - 1 - vs[on_screen]  # same row flip the renderer applies

    # Not flattened: at most a small fraction may pile on the bottom two rows.
    bottom_clip = np.mean(rows >= image_h - 2)
    assert bottom_clip < 0.25, f"{bottom_clip:.0%} of points clipped to the bottom edge"
    # Vertical structure is visible: rows span a wide band, not a thin slab.
    assert int(rows.max() - rows.min()) > 120

    # Height order preserved: a vertical column maps monotonically up the image.
    # Kept within the frame (z up to 1.5 at this range) so the test reads pure
    # ordering, not edge clipping (taller points correctly run off the top).
    col = np.array([[1.0, 0.0, z] for z in np.linspace(-0.2, 1.5, 10)], dtype=np.float32)
    cu, cv, _d, _o = _project_chase_view(col, image_w=image_w, image_h=image_h)
    visible = cu >= 0
    assert np.all(np.diff(cv[visible]) > 0), "higher points must project higher in the image"


def test_world_cloud_span_attributes_assembled() -> None:
    pytest.importorskip("PIL")
    pts = np.zeros((3, 3), dtype=np.float32)
    attrs = world_cloud_span_attributes(
        points_base=pts,
        frame_id="base_link",
        source_node="openral_octomap_server",
        range_max_m=4.0,
        xy_m=2.0,
        z_min=-0.2,
        z_max=2.0,
    )
    assert attrs["openral.world_cloud.frame_id"] == "base_link"
    assert attrs["openral.world_cloud.n_points"] == 3
    assert attrs["openral.world_cloud.source_node"] == "openral_octomap_server"
    png = attrs["openral.world_cloud.png_b64"]
    assert isinstance(png, str) and png
    assert attrs["openral.world_cloud.range_max_m"] == 4.0


def test_dashboard_store_picks_up_world_pointcloud_span() -> None:
    """Synthetic ``world.pointcloud`` OTLP span populates ``_topics['pointcloud']``."""
    pytest.importorskip("opentelemetry.proto")
    from openral_observability.dashboard.store import TelemetryStore
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.trace.v1.trace_pb2 import (
        ResourceSpans,
        ScopeSpans,
        Span,
    )

    def _kv(key: str, *, s: str | None = None, i: int | None = None) -> KeyValue:
        if s is not None:
            return KeyValue(key=key, value=AnyValue(string_value=s))
        return KeyValue(key=key, value=AnyValue(int_value=int(i)))  # type: ignore[arg-type]

    span = Span(
        name="world.pointcloud",
        start_time_unix_nano=1_000_000_000,
        end_time_unix_nano=1_000_500_000,
        attributes=[
            _kv("openral.world_cloud.frame_id", s="base_link"),
            _kv("openral.world_cloud.n_points", i=1234),
            _kv("openral.world_cloud.png_b64", s="iVBORw0KGgo="),
            _kv("openral.world_cloud.source_node", s="openral_octomap_server"),
        ],
    )
    rs = ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])

    store = TelemetryStore()
    store.ingest_spans([rs])
    snap = store.snapshot()
    pc = snap["topics"]["pointcloud"]
    assert pc["frame_id"] == "base_link"
    assert pc["n_points"] == 1234
    assert pc["png_b64"] == "iVBORw0KGgo="
    assert pc["source_node"] == "openral_octomap_server"
    assert pc["ts_unix"]
