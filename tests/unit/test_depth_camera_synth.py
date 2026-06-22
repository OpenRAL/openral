# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the simulated depth-camera → point-cloud synth.

ADR-0030 — `synthesize_depth_pointcloud` casts one `mj_multiRay` ray per
(strided) pixel through a pinhole model and returns the hit points in the
camera *optical* frame (REP-103: +x right, +y down, +z forward). It is the
3-D analogue of `synthesize_laser_scan_2d` and is robot-agnostic: any named
MJCF camera in any robot's scene works.

Tests build a real `mujoco.MjModel` (a camera facing a known-distance wall)
— no mocks, no GL context — and assert the returned cloud's geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# The synth itself defers `mujoco`, so it imports without it; the cast
# functions need a real model, hence the module-level importorskip.
from openral_sim.backends.depth_camera import (
    synthesize_depth_image,
    synthesize_depth_pointcloud,
)

mujoco = pytest.importorskip("mujoco")

# A camera at the world origin (default orientation → looks down world -Z,
# +x right, +y up) facing a large fronto-parallel wall whose near face sits
# at z = -1.9 m. Every ray that hits the wall must report an optical-frame
# z of ~1.9 m regardless of pixel (the wall is perpendicular to the optical
# axis).
_WALL_Z = -2.0
_WALL_HALF_THICK = 0.1
_EXPECTED_DEPTH_M = -(_WALL_Z + _WALL_HALF_THICK)  # 1.9 m to the near face

_DEPTH_CAM_MJCF = f"""
<mujoco model="depth_cam_test">
  <worldbody>
    <camera name="depth0" pos="0 0 0"/>
    <geom name="wall" type="box" pos="0 0 {_WALL_Z}" size="5 5 {_WALL_HALF_THICK}"/>
  </worldbody>
</mujoco>
"""

# Small pinhole — fast to ray-cast, still exercises off-axis pixels.
_W, _H = 32, 24
_FX = _FY = 20.0
_CX, _CY = 16.0, 12.0


def _model_data() -> tuple[object, object]:
    model = mujoco.MjModel.from_xml_string(_DEPTH_CAM_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth_cloud_recovers_fronto_parallel_wall() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    # Every pixel sees the wall, so the cloud is full (W*H points).
    assert points.shape == (_W * _H, 3)
    assert points.dtype == np.float32
    # Fronto-parallel wall → constant optical-frame depth across the image.
    assert np.allclose(points[:, 2], _EXPECTED_DEPTH_M, atol=1e-3)
    # Optical frame: +x right, +y down → both signs present, centred on 0.
    assert points[:, 0].min() < 0.0 < points[:, 0].max()
    assert points[:, 1].min() < 0.0 < points[:, 1].max()
    assert abs(float(points[:, 0].mean())) < 0.05
    assert abs(float(points[:, 1].mean())) < 0.05


def test_depth_cloud_back_projection_matches_pinhole() -> None:
    """A corner pixel back-projects to x=(u-cx)/fx * z, y=(v-cy)/fy * z."""
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    z = _EXPECTED_DEPTH_M
    # The (u=0, v=0) pixel is the first row-major ray.
    expected_x = (0.0 - _CX) / _FX * z
    expected_y = (0.0 - _CY) / _FY * z
    assert points[0, 0] == pytest.approx(expected_x, abs=2e-3)
    assert points[0, 1] == pytest.approx(expected_y, abs=2e-3)
    assert points[0, 2] == pytest.approx(z, abs=2e-3)


def test_depth_cloud_respects_max_range() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=1.0,  # wall is at 1.9 m → out of range → no points
    )
    assert points.shape == (0, 3)


def test_depth_cloud_respects_min_range() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        min_range_m=3.0,  # wall closer than 3 m → dropped
    )
    assert points.shape == (0, 3)


def test_depth_cloud_stride_subsamples() -> None:
    model, data = _model_data()
    points = synthesize_depth_pointcloud(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        stride=2,
    )
    expected = math.ceil(_W / 2) * math.ceil(_H / 2)
    assert points.shape == (expected, 3)


_SELF_FILTER_MJCF = """
<mujoco model="depth_self_filter">
  <worldbody>
    <camera name="depth0" pos="0 0 0"/>
    <body name="robot_arm" pos="0 0 -1.0">
      <geom name="arm" type="box" size="0.3 0.3 0.05"/>
    </body>
    <geom name="wall" type="box" pos="0 0 -2.0" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""


def test_depth_cloud_excludes_robot_body_hits() -> None:
    """exclude_body_ids drops hits on the robot's own bodies (self-filter).

    The camera sees a near 'robot_arm' box (z=-1) in front of the far wall
    (z=-2). Without exclusion the nearest hit is the arm (~1 m); excluding the
    arm body, every ray passes through to the wall (~1.9 m).
    """
    model = mujoco.MjModel.from_xml_string(_SELF_FILTER_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    arm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_arm")

    common = dict(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    without = synthesize_depth_pointcloud(**common)
    withf = synthesize_depth_pointcloud(**common, exclude_body_ids=frozenset({arm_body}))
    # The arm box is the closest thing in view, so without the filter the
    # nearest return is the arm near-face at ~0.95 m...
    assert float(without[:, 2].min()) == pytest.approx(0.95, abs=0.05)
    # ...with the arm excluded every ray passes through to the wall (~1.9 m),
    # and no return is closer than the wall.
    assert float(withf[:, 2].min()) == pytest.approx(1.9, abs=0.05)


def test_depth_cloud_unknown_camera_raises() -> None:
    from openral_core.exceptions import ROSConfigError

    model, data = _model_data()
    with pytest.raises(ROSConfigError, match="nonexistent"):
        synthesize_depth_pointcloud(
            model=model,
            data=data,
            camera_name="nonexistent",
            width=_W,
            height=_H,
            fx=_FX,
            fy=_FY,
            cx=_CX,
            cy=_CY,
            max_range_m=10.0,
        )


# ── synthesize_depth_image (ADR-0064 — dense raster for nvblox) ──────────────


def test_depth_image_dense_fronto_parallel_wall() -> None:
    """The image is dense ``(H, W)`` and a fronto-parallel wall reads constant Z."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    # Dense raster: (rows, cols) = (height, width) — every pixel kept.
    assert depth.shape == (_H, _W)
    assert depth.dtype == np.float32
    # Perpendicular optical-Z is constant across a fronto-parallel wall, and
    # equals the cloud synth's z-column (1.9 m to the near face).
    assert np.allclose(depth, _EXPECTED_DEPTH_M, atol=1e-3)


def test_depth_image_zero_for_misses() -> None:
    """Out-of-range / no-return pixels read exactly ``0.0`` (nvblox's sentinel)."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=1.0,  # wall at 1.9 m → every ray out of range
    )
    assert depth.shape == (_H, _W)
    assert np.count_nonzero(depth) == 0


def test_depth_image_stride_scales_raster() -> None:
    """``stride=2`` rasterises ``ceil(H/2) x ceil(W/2)`` (CameraInfo scales to match)."""
    model, data = _model_data()
    depth = synthesize_depth_image(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
        stride=2,
    )
    assert depth.shape == (math.ceil(_H / 2), math.ceil(_W / 2))
    assert np.allclose(depth, _EXPECTED_DEPTH_M, atol=1e-3)


def test_depth_image_matches_cloud_z_column() -> None:
    """The dense image's nonzero pixels equal the cloud synth's optical-Z, in order.

    Both synths share ``_cast_depth_rays``; on a wall every ray hits, so the
    raster flattened row-major must equal the cloud's ``z`` column exactly.
    """
    model, data = _model_data()
    common = dict(
        model=model,
        data=data,
        camera_name="depth0",
        width=_W,
        height=_H,
        fx=_FX,
        fy=_FY,
        cx=_CX,
        cy=_CY,
        max_range_m=10.0,
    )
    cloud = synthesize_depth_pointcloud(**common)  # (W*H, 3), full wall
    depth = synthesize_depth_image(**common)  # (H, W)
    assert np.allclose(depth.reshape(-1), cloud[:, 2], atol=1e-5)
